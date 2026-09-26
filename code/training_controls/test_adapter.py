"""CPU preparation tests; no model fitting. Remote can additionally run --legacy."""
import copy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import tempfile
import hashlib
import ast
import math
from types import SimpleNamespace
import pandas as pd
from adapter import scope_frames, prepare_group, comparison_contract, initialization_contract, SEEDS, CANDIDATES
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_controls.runner import require_boundary, expected_inputs
import training_controls.runner as runner

SOURCE = Path(os.environ.get('TC_SOURCE_ROOT', str(Path(__file__).resolve().parents[3] / '04_执行代码')))

class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame([dict(stable_record_id='a', model_head='h1', assigned_split='train', is_censored=False),
                                   dict(stable_record_id='b', model_head='h2', assigned_split='train', is_censored=False)])
        self.valid = pd.DataFrame([dict(stable_record_id='c', model_head='h1', assigned_split='valid', is_censored=False)])
    def test_train_scope_isolated_before_fit(self):
        tr, va = scope_frames(self.train, self.valid, 'STL_FULL', 'h1')
        self.assertEqual(tr.stable_record_id.tolist(), ['a'])
        self.assertEqual(va.stable_record_id.tolist(), ['c'])
        self.assertEqual(len(scope_frames(self.train, self.valid, 'MTL_FULL')[0]), 2)
    def test_reject_test_and_invalid_scope(self):
        with self.assertRaises(ValueError): scope_frames(self.train, self.valid.assign(assigned_split='test'), 'MTL_FULL')
        with self.assertRaises(ValueError): scope_frames(self.train, self.valid, 'MTL_FULL', 'h1')
        with self.assertRaises(ValueError): scope_frames(self.train, self.valid, 'STL_FULL')
        with self.assertRaises(ValueError): scope_frames(self.train, self.valid.assign(stable_record_id='a'), 'MTL_FULL')
    def test_original_seeds(self):
        for seed in SEEDS: self.assertEqual(initialization_contract(seed)['dataloader_seed'], seed)
        with self.assertRaises(ValueError): initialization_contract(17)
        self.assertEqual(len(CANDIDATES), 6)
    def test_physical_boundary_and_hash_binding(self):
        frame=self.train.assign(boundary_id='S3_PARENT')
        require_boundary(frame,'S3_PARENT')
        with self.assertRaises(ValueError): require_boundary(frame,'S1_CONDITION')
        with self.assertRaises(ValueError): require_boundary(self.train,'S3_PARENT')
        args=SimpleNamespace(train='train',valid='valid',expected_train_sha256='a',expected_valid_sha256='b')
        expected_inputs(args,{'train':'a','valid':'b'}.__getitem__)
        with self.assertRaises(ValueError): expected_inputs(args,{'train':'changed','valid':'b'}.__getitem__)
    def test_loss_call_matches_actual_legacy_signature(self):
        parsed=ast.parse((SOURCE/'qsar_tl/training/deep_experiment.py').read_text(encoding='utf-8'))
        fn=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=='batch_weighted_loss')
        allowed={n.arg for n in fn.args.args+fn.args.kwonlyargs}
        used=[]
        for node in ast.walk(ast.parse(Path(runner.__file__).read_text(encoding='utf-8'))):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='batch_weighted_loss':
                used.extend(k.arg for k in node.keywords if k.arg)
        self.assertTrue(used)
        self.assertTrue(set(used)<=allowed, f'Unsupported loss arguments: {set(used)-allowed}')
    def test_bundle_resume_never_refits_completed_head(self):
        # A fake fit verifies orchestration only; no network or optimizer exists.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root=Path(tmp)
            for name in ('train','valid','reference'): (root/name).write_text(name)
            def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
            def read(p): return json.loads(Path(p).read_text())
            def write(p,v): Path(p).write_text(json.dumps(v))
            frames={'train':self.train.assign(boundary_id='S3_PARENT'), 'valid':self.valid.assign(boundary_id='S3_PARENT')}
            torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda:None))
            base=SimpleNamespace(load_frame=lambda p,r,part:frames[part], encoder_for=lambda x:object(),
                 runtime=lambda:(None,pd,torch,None,None,None,None,None,None,None))
            args=SimpleNamespace(variant='STL_FULL',candidate='C2',seed=42,route='W00',boundary='S3_PARENT',head=None,
                train=str(root/'train'),valid=str(root/'valid'),reference_manifest=str(root/'reference'),
                expected_train_sha256=digest(root/'train'),expected_valid_sha256=digest(root/'valid'),epochs=30,
                out=str(root/'bundle'),resume=False)
            calls=[]
            def fake_fit(child):
                calls.append(child.head); p=Path(child.out); p.mkdir()
                if child.head=='h2' and calls.count('h2')==1: raise RuntimeError('injected failure')
                (p/'model.pt').write_text('checkpoint fixture')
                contract=read(root/'bundle/bundle_contract.json')
                m={k:v for k,v in contract.items() if k!='heads'}
                m.update(head=child.head,status='formal_complete',artifacts={'model.pt':digest(p/'model.pt')},
                         selection_status='fixed_final_epoch_not_comparison_eligible',train_n=1,valid_n=0)
                write(p/'manifest.json',m)
            with patch.object(runner,'runtime',return_value=(base,digest,read,write,None)), patch.object(runner,'identity',return_value={}), patch.object(runner,'fit',side_effect=fake_fit):
                with self.assertRaisesRegex(RuntimeError,'Bundle incomplete'): runner.fit_bundle(args)
                self.assertEqual(read(root/'bundle/bundle_manifest.json')['status'],'incomplete')
                args.resume=True; runner.fit_bundle(args)
                self.assertEqual(calls,['h1','h2','h2'])
                self.assertEqual(read(root/'bundle/bundle_manifest.json')['status'],'formal_complete')

class LegacyPreparationTests(unittest.TestCase):
    """Run in a valid PyTorch environment with --legacy; creates no weights."""
    def test_eight_tiny_forward_backward_contracts_no_parameter_updates(self):
        sys.path.insert(0,str(SOURCE))
        import torch
        from revision_pipeline import train as base
        from qsar_tl.training import deep_experiment as legacy
        from qsar_tl.training.deep_train import DeepTrainingConfig,collate_aggregated_task_batch,set_torch_seed
        from qsar_tl.modeling.network import DeepModelConfig,EcotoxMultiTaskNetwork,TOXICITY_BIN_LOGITS_KEY
        for route in ('W00','M00'):
            for variant in ('MTL_FULL','STL_FULL','MTL_MOL','STL_MOL'):
                set_torch_seed(42)
                heads=('h1','h2') if variant.startswith('MTL') else ('h1',)
                cats={} if variant.endswith('MOL') else {k:4 for k in legacy.CATEGORICAL_COLUMNS}
                mc=DeepModelConfig(numeric_dim=22,fingerprint_dim=512,task_heads=heads,categorical_cardinalities=cats,
                    descriptor_count=8,hidden_dims=(256,128),use_adapters=False,use_molecular_residual=True,
                    toxicity_bin_count=13,toxicity_binning_mode='aux_classification')
                model=EcotoxMultiTaskNetwork(mc)
                rows=[]
                for i in range(4):
                    rows.append(dict(molecular_numeric=[.1]*8+[0.]*14,fingerprint=[1.]+[0.]*511,
                        categorical_ids={k:1 for k in cats},adapter_id=0,task_head=heads[i%len(heads)],target_value=.3+i*.2,
                        toxicity_bin_index=i if i<3 else -1,censored_direction_id=0))
                batch=collate_aggregated_task_batch(rows)
                cfg=DeepTrainingConfig(task_weights={h:1. for h in heads},toxicity_bin_loss_weight=.025 if route=='W00' else 0.,
                    toxicity_binning_mode='aux_classification',huber_delta=1.,mse_loss_weight=0.,gradient_clip_norm=5.)
                before={k:v.detach().clone() for k,v in model.state_dict().items()}
                class NoUpdate:
                    def zero_grad(self,**kwargs): model.zero_grad(**kwargs)
                    def step(self): pass  # contract check, no training update
                loss=runner.epoch(model,[batch],cfg,torch,legacy,base,legacy.build_regression_loss(cfg),'cpu',NoUpdate())
                self.assertTrue(math.isfinite(loss),(route,variant))
                with torch.no_grad(): outputs=model(batch['molecular_numeric'],batch['fingerprint'],batch['categorical_ids'],adapter_ids=batch['adapter_id'])
                self.assertEqual(tuple(outputs[TOXICITY_BIN_LOGITS_KEY].shape),(4,13))
                grad=model.toxicity_bin_classifier.weight.grad
                if route=='W00': self.assertIsNotNone(grad); self.assertTrue(torch.isfinite(grad).all()); self.assertGreater(float(grad.abs().sum()),0.)
                else: self.assertTrue(grad is None or bool((grad==0).all()))
                self.assertTrue(all(torch.equal(before[k],v) for k,v in model.state_dict().items()))
    def test_real_legacy_independence_and_molecule_invariance(self):
        sys.path.insert(0, str(SOURCE))
        from revision_pipeline.train import encoder_for
        from revision_pipeline.train_contract import full_contract
        from qsar_tl.training import deep_experiment as legacy
        from qsar_tl.training.baseline import add_duration_nonlinear_features
        ref = Path(os.environ.get('TC_REFERENCE_MANIFEST', str(SOURCE.parent / '01_冻结来源/reference_models/W00/manifest.json')))
        contract = full_contract(ref, 'W00')
        def row(i, head, part, smiles, species, target):
            d = {k: 'train_value' for k in legacy.CATEGORICAL_COLUMNS}
            d.update(stable_record_id=i, aggregate_id=i, model_head=head, task_head=head,
                     assigned_split=part, split_part=part, is_censored=False, smiles=smiles,
                     latin_name=species, target=target, target_name='ptox_mol_l',
                     target_family='aquatic_pTox_mol_L', effect_level_x=50, duration_bin_h=24,
                     task_group='toxicity')
            return d
        tr = add_duration_nonlinear_features(pd.DataFrame([
            row('a','h1','train','CC','one',1.), row('b','h1','train','CCC','one',3.),
            row('d','h2','train','CCCC','other_head_only',100.)]))
        va = add_duration_nonlinear_features(pd.DataFrame([row('c','h1','valid','CC','valid_only',999.)]))
        enc = encoder_for(legacy)
        stl = prepare_group(tr, va, 'STL_FULL', contract, legacy, enc, head='h1')
        mtl = prepare_group(tr, va, 'MTL_FULL', contract, legacy, enc)
        self.assertTrue(comparison_contract(mtl['preprocessing'], stl['preprocessing']))
        pre = stl['preprocessing']
        self.assertEqual(pre['fit_ids'], ['a','b'])
        self.assertNotIn('valid_only', pre['categorical_maps']['latin_name'])
        self.assertNotIn('other_head_only', pre['categorical_maps']['latin_name'])
        va2 = va.assign(target=-999., latin_name='changed_valid', effect_level_x=999.)
        again = prepare_group(tr, va2, 'STL_FULL', contract, legacy, enc, head='h1')
        self.assertEqual(pre, again['preprocessing'])
        mol = prepare_group(tr, va, 'STL_MOL', contract, legacy, enc, head='h1')
        mol2 = prepare_group(tr, va2, 'STL_MOL', contract, legacy, enc, head='h1')
        self.assertEqual(mol['preprocessing']['categorical_maps'], {})
        self.assertEqual(mol['valid_samples'][0]['molecular_numeric'], mol2['valid_samples'][0]['molecular_numeric'])
        self.assertTrue(all(x == 0 for x in mol['valid_samples'][0]['molecular_numeric'][8:]))

if __name__ == '__main__':
    use_legacy = '--legacy' in sys.argv
    if use_legacy: sys.argv.remove('--legacy')
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ScopeTests)
    if use_legacy: suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(LegacyPreparationTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)

"""Sensitivity contracts and optional real feature/Gaussian tests; no training."""
from dataclasses import dataclass
import io
import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training_controls.sensitivity_runner import exact_anchor, spec_for, fingerprint_bits, fit_pre, samples_for, molecular_encoder,likelihood_collate,gaussian_epoch

@dataclass(frozen=True)
class Spec:
    name:str='full'
    use_medium_adapter:bool=True
    use_duration_features:bool=True
    use_effect_level_features:bool=True
    species_lifestage_columns:tuple|None=None

class Contracts(unittest.TestCase):
    def test_anchor_rejects_different_exact_supervision(self):
        anchor=pd.DataFrame([dict(stable_record_id='a',target=1.,model_head='h',smiles='CC',is_censored=False)])
        combined=pd.concat([anchor,pd.DataFrame([dict(stable_record_id='b',target=float('nan'),model_head='h',smiles='CCC',is_censored=True)])],ignore_index=True)
        exact_anchor(combined,anchor)
        with self.assertRaises(ValueError): exact_anchor(combined.assign(target=99.),anchor)
        with self.assertRaises(ValueError): exact_anchor(combined,combined)
    def test_registered_mask_flags_and_bits(self):
        legacy=SimpleNamespace(ABLATION_SPECS={'full':Spec(),'taxonomy_identity_only':Spec(species_lifestage_columns=('latin_name','organism_lifestage'))})
        s=spec_for(legacy,'E51_NO_TIME_EFFECT')
        self.assertFalse(s.use_duration_features); self.assertFalse(s.use_effect_level_features); self.assertFalse(s.use_medium_adapter)
        self.assertIn('latin_name',spec_for(legacy,'E50_TAXONOMY_IDENTITY').species_lifestage_columns)
        self.assertEqual(fingerprint_bits('E40_FP2048'),2048); self.assertEqual(fingerprint_bits('E30_C1'),512)

class RealNumerical(unittest.TestCase):
    def test_actual_preprocessors_masks_and_censor_labels(self):
        source=os.environ.get('TC_SOURCE_ROOT')
        if source: sys.path.insert(0,source)
        from qsar_tl.training import deep_experiment as legacy
        from qsar_tl.training.baseline import add_duration_nonlinear_features
        from revision_pipeline.train_contract import full_contract
        ref=os.environ['TC_REFERENCE_MANIFEST']; scheme=os.environ['TC_BIN_SCHEME']
        import json
        c=full_contract(ref,'W00'); rm=json.loads(Path(ref).read_text())
        c['toxicity_binning_config']={k:v for k,v in rm['toxicity_binning'].items() if k in legacy.ToxicityBinningConfig.__dataclass_fields__}
        c['toxicity_bin_scheme']=legacy.load_toxicity_bin_scheme(scheme)
        def row(i,part,target,censored=False):
            d={k:'train_value' for k in legacy.CATEGORICAL_COLUMNS}
            d.update(stable_record_id=i,aggregate_id=i,model_head='h',task_head='h',target=target,is_censored=censored,
                assigned_split=part,split_part=part,smiles='CC' if i=='a' else 'CCC',effect_level_x=50.,duration_bin_h=24.,
                target_name='ptox_mol_l',target_family='aquatic_pTox_mol_L',task_group='toxicity',
                unit_family_v2='water_mol_l',standard_value_mg_l=1.,standard_value_mol_l=.001,
                bound_lower=2. if censored else float('nan'),bound_upper=float('nan'))
            return d
        anchor=add_duration_nonlinear_features(pd.DataFrame([row('a','train',1.),row('b','train',3.)]))
        censor=add_duration_nonlinear_features(pd.DataFrame([row('c','train',float('nan'),True)]))
        valid=add_duration_nonlinear_features(pd.DataFrame([row('v','valid',2.)]))
        enc=molecular_encoder(legacy,512)
        pre1=fit_pre(anchor,c,legacy,enc,'E30_C1'); pre2=fit_pre(anchor,c,legacy,enc,'E30_C2')
        for key in ('numeric_stats','target_scaler','categorical_maps','zscore_correction','fit_ids'):
            self.assertEqual(pre1[key],pre2[key])
        cs=samples_for(censor,pre2,legacy,enc)[0]
        self.assertEqual(cs['toxicity_bin_index'],-1); self.assertIsNone(cs['target_value_raw'])
        self.assertTrue(cs['bound_lower_scaled']<cs['bound_upper_scaled'])
        ps=samples_for(anchor,pre1,legacy,enc)
        self.assertTrue(any(s['toxicity_bin_index']>=0 for s in ps))
        e51=fit_pre(anchor,c,legacy,enc,'E51_NO_TIME_EFFECT')
        varied=add_duration_nonlinear_features(valid.assign(effect_level_x=5.,duration_bin_h=720.))
        a=samples_for(valid,e51,legacy,enc)[0]; b=samples_for(varied,e51,legacy,enc)[0]
        self.assertEqual(a['molecular_numeric'],b['molecular_numeric'])
        self.assertTrue(all(v==0 for v in a['molecular_numeric'][8:]))
        e50=fit_pre(anchor,c,legacy,enc,'E50_TAXONOMY_IDENTITY')
        self.assertIn('latin_name',e50['categorical_maps']); self.assertIn('organism_lifestage',e50['categorical_maps'])
        self.assertTrue(set(e50['categorical_maps']).isdisjoint({'genus','kingdom','family','species','taxon_group_l1'}))
        enc2048=molecular_encoder(legacy,2048)
        self.assertEqual(enc.encode('CC')[0],enc2048.encode('CC')[0])
        self.assertEqual(len(enc.encode('CC')[1]),512); self.assertEqual(len(enc2048.encode('CC')[1]),2048)
    def test_sigma_checkpoint_and_tail_gradients(self):
        import torch
        from censor_gaussian import HeadSigma,gaussian_nll
        sigma=HeadSigma(['h1','h2']); buf=io.BytesIO(); torch.save(sigma.state_dict(),buf); buf.seek(0)
        restored=HeadSigma(['h1','h2']); restored.load_state_dict(torch.load(buf,weights_only=True),strict=True)
        self.assertTrue(torch.equal(sigma(['h2','h1']),restored(['h2','h1'])))
        mu=torch.tensor([0.,0.,0.],requires_grad=True)
        sd=sigma(['h1','h2','h1'])
        lo=torch.tensor([1.,-float('inf'),40.]); hi=torch.tensor([1.,-40.,float('inf')]); exact=torch.tensor([True,False,False])
        loss=gaussian_nll(mu,sd,lo,hi,exact).sum(); loss.backward()
        self.assertTrue(torch.isfinite(loss)); self.assertTrue(torch.isfinite(mu.grad).all()); self.assertTrue(torch.isfinite(sigma.raw.grad).all())
        batch=likelihood_collate(torch,lambda rows:{})([dict(bound_lower_scaled=1.,bound_upper_scaled=1.+1e-10)])
        self.assertEqual(batch['bound_lower_scaled'].dtype,torch.float64)
        self.assertGreater(float(batch['bound_upper_scaled'][0]-batch['bound_lower_scaled'][0]),0.)
    def test_gaussian_runner_one_batch_no_parameter_update(self):
        import torch
        from censor_gaussian import HeadSigma
        from revision_pipeline import train as base
        from qsar_tl.modeling.network import DeepModelConfig,EcotoxMultiTaskNetwork
        from qsar_tl.training.deep_train import DeepTrainingConfig,collate_aggregated_task_batch
        network=EcotoxMultiTaskNetwork(DeepModelConfig(numeric_dim=22,fingerprint_dim=512,task_heads=('h1','h2'),
            descriptor_count=8,hidden_dims=(32,16),use_adapters=False,toxicity_bin_count=13,toxicity_binning_mode='aux_classification'))
        sigma=HeadSigma(['h1','h2'])
        rows=[]
        for i in range(2):
            rows.append(dict(molecular_numeric=[.1]*22,fingerprint=[0.]*512,categorical_ids={},adapter_id=0,task_head=f'h{i+1}',
                target_value=.5 if i==0 else 0.,censored_direction_id=i,toxicity_bin_index=0 if i==0 else -1,
                bound_lower_scaled=.5 if i==0 else 40.,bound_upper_scaled=.5 if i==0 else float('inf')))
        batch=likelihood_collate(torch,collate_aggregated_task_batch)(rows)
        cfg=DeepTrainingConfig(task_weights={'h1':1.,'h2':1.},toxicity_bin_loss_weight=.025,gradient_clip_norm=5.)
        before={k:v.clone() for k,v in network.state_dict().items()}; raw_before=sigma.raw.detach().clone()
        class NoUpdate:
            def zero_grad(self,**kw): network.zero_grad(**kw); sigma.zero_grad(**kw)
            def step(self): pass
        value=gaussian_epoch(network,sigma,[batch],cfg,torch,base,'cpu',NoUpdate())
        self.assertTrue(torch.isfinite(torch.tensor(value)))
        self.assertTrue(torch.isfinite(sigma.raw.grad).all())
        self.assertTrue(all(torch.equal(before[k],v) for k,v in network.state_dict().items()))
        self.assertTrue(torch.equal(raw_before,sigma.raw.detach()))

if __name__=='__main__':
    extended='--legacy' in sys.argv
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    if extended: suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(RealNumerical))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)

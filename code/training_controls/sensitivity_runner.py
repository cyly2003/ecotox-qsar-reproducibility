"""Isolated E30/E40/E50/E51 runner. Frozen core modules are never modified.

C0/C1/C2 have an explicit common exact-train preprocessing anchor. Only C2
loads censored training supervision; all early selection uses exact validation.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import platform
import sys
from . import runner as core
from .adapter import CANDIDATES, SEEDS

ARMS=('E30_C0','E30_C1','E30_C2','E40_FP2048','E50_TAXONOMY_IDENTITY','E51_NO_TIME_EFFECT')

def spec_for(legacy,arm):
    if arm not in ARMS: raise ValueError('Unknown registered sensitivity arm')
    if arm=='E50_TAXONOMY_IDENTITY':
        return replace(legacy.ABLATION_SPECS['taxonomy_identity_only'],use_medium_adapter=False)
    spec=replace(legacy.ABLATION_SPECS['full'],use_medium_adapter=False)
    if arm=='E51_NO_TIME_EFFECT':
        return replace(spec,name='no_time_and_effect_all_encodings',use_duration_features=False,use_effect_level_features=False)
    return spec

def fingerprint_bits(arm): return 2048 if arm=='E40_FP2048' else 512

def sensitivity_identity(digest):
    values=core.identity(digest)
    root=Path(__file__).parents[1]
    for p in (Path(__file__),root/'censor_gaussian.py'):
        values[str(p.relative_to(root)).replace('\\','/')]=digest(p)
    return values

def molecular_encoder(legacy,bits):
    from rdkit import Chem
    enc=legacy.MolecularFeatureBuilder(fingerprint_size=bits)
    if enc.source!='rdkit': raise RuntimeError('Real RDKit required')
    cache={}
    def encode(smiles):
        value=str(smiles)
        if value not in cache:
            if Chem.MolFromSmiles(value) is None: raise ValueError('Unparseable admitted structure')
            cache[value]=enc._encode_rdkit(value)
        return cache[value]
    enc.encode=encode
    return enc

def exact_anchor(train,anchor,feature_columns=()):
    """C2 exact rows must be identical to the common exact preprocessing anchor."""
    import pandas as pd
    if anchor.is_censored.astype(bool).any(): raise ValueError('Preprocessing anchor must be exact only')
    observed=train.loc[~train.is_censored.astype(bool)]
    columns=list(dict.fromkeys(['stable_record_id','target','model_head','smiles',*feature_columns]))
    lhs=observed[columns].sort_values('stable_record_id').reset_index(drop=True)
    rhs=anchor[columns].sort_values('stable_record_id').reset_index(drop=True)
    try: pd.testing.assert_frame_equal(lhs,rhs,check_dtype=False,check_exact=True)
    except AssertionError as exc: raise ValueError('C2 exact rows differ from common exact anchor') from exc

def fit_pre(anchor,contract,legacy,encoder,arm):
    if anchor.is_censored.any() or set(anchor.assigned_split.astype(str))!={'train'}:
        raise ValueError('Only common exact train may fit preprocessing')
    point=anchor.copy(); point['target_value']=point.target.astype(float)
    spec=spec_for(legacy,arm); names=tuple(contract['descriptor_names'])
    cats=legacy.fit_categorical_maps(point,ablation=spec,min_count=contract['categorical_min_count'])
    stats=legacy.fit_numeric_stats(point,encoder,descriptor_names=names,ablation=spec)
    z=legacy.fit_zscore_correction(point,encoder,numeric_stats=stats,
        feature_names=legacy.build_numeric_feature_names(len(names),descriptor_names=names),descriptor_names=names,
        config=legacy.ZScoreCorrectionConfig(enabled=contract['zscore_enabled'],threshold=contract['zscore_threshold']),ablation=spec)
    scaler=legacy.fit_target_scaler(point,target_column='target_value',mode='per_task_target',fit_indices=list(range(len(point))))
    return dict(arm=arm,variant='MTL_FULL',categorical_maps=cats,numeric_stats=stats,zscore_correction=z.to_manifest(),
        target_scaler=scaler.to_manifest(),descriptor_names=list(names),task_heads=sorted(point.model_head.astype(str).unique()),
        fit_ids=sorted(point.stable_record_id.astype(str)),fit_partition='common_exact_train_only',
        toxicity_binning_config=contract.get('toxicity_binning_config'),toxicity_bin_scheme=contract.get('toxicity_bin_scheme'),
        fingerprint_bits=fingerprint_bits(arm),numeric_dim=len(names)+len(legacy.CONTEXT_NUMERIC_COLUMNS))

def samples_for(frame,pre,legacy,encoder):
    from revision_pipeline.train_contract import bounds_contract
    spec=spec_for(legacy,pre['arm']); scaler=legacy.TargetScaler(**pre['target_scaler'])
    work=frame.copy(); work['target_value']=work.target.where(~work.is_censored,0.)
    z=legacy.ZScoreCorrection(**{k:v for k,v in pre['zscore_correction'].items() if k!='method'})
    samples=legacy.build_deep_samples(work,encoder=encoder,descriptor_names=tuple(pre['descriptor_names']),
        categorical_maps=pre['categorical_maps'],adapter_map={},numeric_stats=pre['numeric_stats'],target_column='target_value',
        target_scaler=scaler,zscore_correction=z,ablation=spec)
    for sample,row in zip(samples,frame.to_dict('records')):
        key=legacy.target_scale_key(row|{'task_head':row['model_head']},scaler.mode)
        if key not in scaler.stats: raise ValueError('Head has no common exact-train target scaler')
        lo,hi=bounds_contract(row['is_censored'],row['target'],row['bound_lower'],row['bound_upper'])
        sample.update(stable_record_id=str(row['stable_record_id']),is_censored=bool(row['is_censored']),
            target_value_raw=None if row['is_censored'] else float(row['target']),
            bound_lower_scaled=scaler.transform(key,lo) if math.isfinite(lo) else -math.inf,
            bound_upper_scaled=scaler.transform(key,hi) if math.isfinite(hi) else math.inf,
            censored_direction_id=int(row['is_censored']),toxicity_bin_index=-1)
        # Never assign a point-bin label to a censor bound, even temporarily.
        if not row['is_censored'] and pre.get('toxicity_binning_config'):
            raw=encoder.encode(row['smiles'])[0]
            mw=raw[pre['descriptor_names'].index('MolWt')]
            sample.update(legacy.assign_toxicity_bin(row,pre['toxicity_bin_scheme'],
                config=legacy.ToxicityBinningConfig(**pre['toxicity_binning_config']),descriptor_mol_weight=mw).as_sample_fields())
    return samples

def gaussian_epoch(network,sigma,loader,cfg,torch,base,device,optimizer=None):
    from censor_gaussian import task_balanced_nll
    from qsar_tl.training.deep_train import toxicity_bin_classification_loss
    network.train(optimizer is not None); sigma.train(optimizer is not None)
    weighted=0.; count=0
    for batch in loader:
        b=base.move_batch(batch,torch,device)
        if optimizer is not None: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            outputs=network(b['molecular_numeric'],b['fingerprint'],b['categorical_ids'],adapter_ids=b['adapter_id'])
            mu=torch.stack([outputs[h][i] for i,h in enumerate(b['task_head'])])
            loss=task_balanced_nll(mu,sigma(b['task_head']),b['bound_lower_scaled'],b['bound_upper_scaled'],
                b['censored_direction_id']==0,b['task_head'],cfg.task_weights)
            aux=toxicity_bin_classification_loss(outputs,toxicity_bin_index=b['toxicity_bin_index'],mode='aux_classification')
            if aux is not None: loss=loss+cfg.toxicity_bin_loss_weight*aux
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite Gaussian objective')
            if optimizer is not None:
                loss.backward(); norm=torch.nn.utils.clip_grad_norm_(list(network.parameters())+list(sigma.parameters()),cfg.gradient_clip_norm)
                if not bool(torch.isfinite(norm)): raise FloatingPointError('Nonfinite Gaussian gradient')
                optimizer.step()
        n=len(b['task_head']); weighted+=float(loss.detach().cpu())*n; count+=n
    return weighted/count if count else None

def likelihood_collate(torch,collate):
    def fn(rows):
        batch=collate(rows)
        # Keep original bound precision until the float64 likelihood; do not
        # round close intervals through the legacy float32 target collator.
        batch['bound_lower_scaled']=torch.tensor([r['bound_lower_scaled'] for r in rows],dtype=torch.float64)
        batch['bound_upper_scaled']=torch.tensor([r['bound_upper_scaled'] for r in rows],dtype=torch.float64)
        return batch
    return fn

def fit(args):
    if platform.system()!='Linux': raise RuntimeError('Training is remote Linux only')
    base,digest,read,write,full=core.runtime()
    np,pd,torch,rdkit,legacy,Training,collate,seed_fn,Config,Network=base.runtime()
    if not args.device.startswith('cuda') or not torch.cuda.is_available(): raise RuntimeError('Authorized CUDA required')
    if args.epochs not in (1,30): raise ValueError('1-epoch smoke or 30-epoch budget only')
    core.expected_inputs(args,digest)
    if digest(args.parent_selection_lock)!=args.expected_parent_selection_sha256: raise ValueError('Parent selection lock changed')
    lock=read(args.parent_selection_lock)
    boundary='S1_CONDITION' if args.arm=='E51_NO_TIME_EFFECT' else 'S3_PARENT'
    if args.boundary!=boundary: raise ValueError('Sensitivity arm on wrong boundary')
    if lock.get('test_used_for_selection') is not False or lock.get('variant')!='MTL_FULL' or lock.get('route')!=args.route or lock.get('boundary')!=boundary:
        raise ValueError('Require matching E10 MTL_FULL development-only selection lock')
    candidate=lock.get('selected_candidate')
    if candidate not in CANDIDATES: raise ValueError('Parent recipe candidate unresolved')
    args.candidate=candidate; contract=core.build_contract(args,legacy,full,read)
    train=base.load_frame(args.train,args.route,'train'); valid=base.load_frame(args.valid,args.route,'valid')
    core.require_boundary(train,boundary); core.require_boundary(valid,boundary)
    if valid.is_censored.any(): raise ValueError('Validation must be common exact point cohort')
    if set(train.stable_record_id)&set(valid.stable_record_id) or core.group_keys(train)&core.group_keys(valid) or base.source_ids(train)&base.source_ids(valid):
        raise ValueError('Development partition leakage')
    is_e30=args.arm.startswith('E30_')
    if is_e30:
        if not args.exact_train or not args.expected_exact_train_sha256 or digest(args.exact_train)!=args.expected_exact_train_sha256:
            raise ValueError('E30 requires registered common exact-train preprocessing anchor')
        anchor=base.load_frame(args.exact_train,args.route,'train'); core.require_boundary(anchor,boundary)
        exact_anchor(train,anchor,(*legacy.CATEGORICAL_COLUMNS,*legacy.CONTEXT_NUMERIC_COLUMNS))
        if args.arm!='E30_C2' and train.is_censored.any(): raise ValueError('C0/C1 training must be exact only')
    else:
        if train.is_censored.any(): raise ValueError('Feature sensitivity retains point supervision only')
        anchor=train.copy()
    out=Path(args.out or os.environ['REVISION_ATTEMPT_DIR']); out.mkdir(parents=True,exist_ok=False)
    seed_fn(args.seed); np.random.seed(args.seed)
    enc=molecular_encoder(legacy,fingerprint_bits(args.arm)); pre=fit_pre(anchor,contract,legacy,enc,args.arm)
    excluded_train=int((~train.model_head.astype(str).isin(pre['task_heads'])).sum())
    train=train.loc[train.model_head.astype(str).isin(pre['task_heads'])].reset_index(drop=True)
    valid=valid.loc[valid.model_head.astype(str).isin(pre['task_heads'])].reset_index(drop=True)
    if valid.empty: raise ValueError('No exact validation targets supported by common exact train')
    samples=samples_for(train,pre,legacy,enc); vsamples=samples_for(valid,pre,legacy,enc)
    exact_samples=samples_for(anchor,pre,legacy,enc)
    weights=legacy.resolve_task_weights(exact_samples,train_indices=list(range(len(exact_samples))) if args.route=='W00' else [],
        train_cfg=contract['task_weighting_config'],task_heads=tuple(pre['task_heads']))
    cfg=Training(epochs=args.epochs,batch_size=contract['batch_size'],learning_rate=contract['learning_rate'],weight_decay=contract['weight_decay'],
        gradient_clip_norm=contract['gradient_clip_norm'],scheduler=contract['scheduler'],huber_delta=contract['huber_delta'],mse_loss_weight=0.,
        task_weights=weights,device=args.device,seed=args.seed,toxicity_bin_loss_weight=contract['toxicity_bin_loss_weight'],toxicity_binning_mode='aux_classification')
    from .adapter import model_config
    mc=replace(model_config(pre,contract,Config),fingerprint_dim=fingerprint_bits(args.arm))
    network=Network(mc).to(args.device)
    gaussian=args.arm in ('E30_C1','E30_C2'); sigma=None
    if gaussian:
        from censor_gaussian import HeadSigma
        sigma=HeadSigma(pre['task_heads'],sigma_floor=1e-4).to(args.device)
    params=[{'params':list(network.parameters()),'weight_decay':cfg.weight_decay}]
    if sigma is not None: params.append({'params':list(sigma.parameters()),'weight_decay':0.})
    optimizer=torch.optim.AdamW(params,lr=cfg.learning_rate)
    scheduler=legacy.build_scheduler(optimizer,cfg,args.epochs); loss_fn=legacy.build_regression_loss(cfg)
    collate_fn=likelihood_collate(torch,collate)
    loader=torch.utils.data.DataLoader(samples,batch_size=cfg.batch_size,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(args.seed),collate_fn=collate_fn)
    vl=torch.utils.data.DataLoader(vsamples,batch_size=cfg.batch_size,shuffle=False,num_workers=0,collate_fn=collate_fn)
    best=math.inf; stale=0; best_epoch=0; history=[]; state=None; sigstate=None
    for e in range(1,args.epochs+1):
        if gaussian:
            tl=gaussian_epoch(network,sigma,loader,cfg,torch,base,args.device,optimizer)
            value=gaussian_epoch(network,sigma,vl,cfg,torch,base,args.device)
        else:
            tl=core.epoch(network,loader,cfg,torch,legacy,base,loss_fn,args.device,optimizer)
            value=core.epoch(network,vl,cfg,torch,legacy,base,loss_fn,args.device)
        if value<best-contract['min_delta']:
            best=value; stale=0; best_epoch=e
            state={k:v.detach().cpu().clone() for k,v in network.state_dict().items()}
            if sigma is not None: sigstate={k:v.detach().cpu().clone() for k,v in sigma.state_dict().items()}
        else: stale+=1
        history.append(dict(epoch=e,train_loss=tl,valid_exact_loss=value,best_epoch=best_epoch,
            train_sample_visits=len(samples),exact_sample_visits=len(exact_samples),censored_sample_visits=sum(s['is_censored'] for s in samples),optimizer_updates=len(loader)))
        print(json.dumps(history[-1]),flush=True); legacy.step_scheduler(scheduler,value)
        if stale>=contract['patience']: break
    network.load_state_dict(state,strict=True); torch.save(state,out/'model.pt')
    if sigma is not None: sigma.load_state_dict(sigstate,strict=True); torch.save(sigstate,out/'sigma.pt')
    write(out/'preprocessing.json',pre)
    audit=dict(train_ids=sorted(train.stable_record_id.astype(str)),valid_ids=sorted(valid.stable_record_id.astype(str)),
        development_groups=sorted(core.group_keys(train)|core.group_keys(valid)),development_sources=sorted(base.source_ids(train)|base.source_ids(valid)))
    write(out/'boundary_audit.json',audit); pd.DataFrame(history).to_csv(out/'history.csv',index=False)
    prediction=base.predict_rows(network,valid,vsamples,pre,args.device,torch,pd,collate,cfg.batch_size)
    prediction.to_parquet(out/'valid_predictions.parquet',index=False)
    write(out/'manifest.json',dict(schema='sensitivity_v1',status='smoke_complete' if args.epochs==1 else 'formal_complete',
        arm=args.arm,variant='MTL_FULL',route=args.route,boundary=boundary,seed=args.seed,candidate=candidate,gaussian=gaussian,
        parent_selection_sha256=digest(args.parent_selection_lock),parent_selection=lock,
        contract=contract,model_config=asdict(mc),training_config=asdict(cfg),sigma_floor=1e-4 if gaussian else None,
        sigma_initial=1.0 if gaussian else None,sigma_parameterization='per_head_softplus_plus_floor' if gaussian else None,
        sigma_weight_decay=0. if gaussian else None,train_n=len(samples),exact_train_n=len(exact_samples),valid_n=len(valid),
        excluded_no_exact_train_head_rows=excluded_train,best_epoch=best_epoch,max_epochs=args.epochs,
        parameter_count=sum(p.numel() for p in network.parameters())+(sum(p.numel() for p in sigma.parameters()) if sigma else 0),
        train_sha256=digest(args.train),valid_sha256=digest(args.valid),exact_train_sha256=digest(args.exact_train) if is_e30 else digest(args.train),
        preprocessing_fit='common_exact_train_only' if is_e30 else 'point_train_only',task_weights_fit='same_common_exact_train',
        target_scale='ptox_mol_l' if args.route=='W00' else 'neg_log10_mol_kg',
        early_stopping='same_exact_validation_huber' if not gaussian else 'same_exact_validation_gaussian_nll',
        code_identity=sensitivity_identity(digest),artifacts={p.name:digest(p) for p in out.iterdir() if p.is_file()},
        valid_metrics=base.score(prediction),test_loaded_rows=0,test_labels_used_for_selection=False,
        environment=dict(python=sys.version,torch=torch.__version__,rdkit=rdkit.__version__,platform=platform.platform())))

def infer(args):
    base,digest,read,write,full=core.runtime()
    np,pd,torch,rdkit,legacy,Training,collate,seed_fn,Config,Network=base.runtime()
    run=Path(args.run); m=read(run/'manifest.json'); lock=read(args.selection_lock)
    if m['status']!='formal_complete' or lock.get('test_used_for_selection') is not False or digest(run/'manifest.json') not in lock.get('approved_manifest_sha256',[]):
        raise ValueError('Formal pre-test approval required')
    if any(lock.get(k)!=m[k] for k in ('route','boundary')): raise ValueError('Inference lock route/boundary mismatch')
    approved=lock.get('input_sha256_by_partition',{}).get(args.part,[])
    if isinstance(approved,str): approved=[approved]
    if digest(args.data) not in approved: raise ValueError('Unapproved evaluation cohort hash')
    if m['code_identity']!=sensitivity_identity(digest) or m['environment']['rdkit']!=rdkit.__version__: raise ValueError('Source/environment changed')
    for name,sha in m['artifacts'].items():
        if digest(run/name)!=sha: raise ValueError('Training artifact changed')
    raw=base.load_frame(args.data,m['route'],args.part); core.require_boundary(raw,m['boundary'])
    audit=read(run/'boundary_audit.json'); pre=read(run/'preprocessing.json')
    if args.part=='test':
        if set(raw.stable_record_id.astype(str))&set(audit['train_ids']+audit['valid_ids']) or core.group_keys(raw)&set(audit['development_groups']) or base.source_ids(raw)&set(audit['development_sources']):
            raise ValueError('Heldout test overlaps development')
    elif digest(args.data)!=m[args.part+'_sha256']: raise ValueError('Development file changed')
    known=raw.model_head.astype(str).isin(pre['task_heads']); frame=raw.loc[known].reset_index(drop=True)
    enc=molecular_encoder(legacy,pre['fingerprint_bits']); samples=samples_for(frame,pre,legacy,enc) if len(frame) else []
    network=Network(Config(**m['model_config'])).to(args.device)
    network.load_state_dict(torch.load(run/'model.pt',map_location=args.device,weights_only=True),strict=True)
    if len(frame):
        prediction=base.predict_rows(network,frame,samples,pre,args.device,torch,pd,collate,m['contract']['batch_size'])
        output=raw.merge(prediction[['stable_record_id','prediction']],on='stable_record_id',how='left',validate='one_to_one')
    else: output=raw.copy(); output['prediction']=float('nan')
    output['prediction_status']=np.where(output.prediction.notna(),'PREDICTED','NOT_ESTIMABLE_NO_EXACT_TRAIN_LABEL')
    if m['gaussian']:
        from censor_gaussian import HeadSigma
        sigma=HeadSigma(pre['task_heads'],sigma_floor=m['sigma_floor']).to(args.device)
        sigma.load_state_dict(torch.load(run/'sigma.pt',map_location=args.device,weights_only=True),strict=True)
        with torch.no_grad(): values=sigma(pre['task_heads']).cpu().numpy()
        mapping=dict(zip(pre['task_heads'],values)); scaler=legacy.TargetScaler(**pre['target_scaler'])
        sig=[]
        for row in output.to_dict('records'):
            h=str(row['model_head'])
            key=legacy.target_scale_key(row|{'task_head':h},scaler.mode)
            sig.append(float(mapping[h])*float(scaler.stats[key]['std']) if h in mapping else float('nan'))
        output['sigma_native']=sig
    output['censor_violation_distance']=float('nan')
    for idx,row in output.loc[output.is_censored & output.prediction.notna()].iterrows():
        lo=float(row.bound_lower) if pd.notna(row.bound_lower) else -math.inf
        hi=float(row.bound_upper) if pd.notna(row.bound_upper) else math.inf
        output.at[idx,'censor_violation_distance']=max(lo-row.prediction,0.)+max(row.prediction-hi,0.)
    out=Path(args.out)
    if out.exists(): raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True); output.to_parquet(out,index=False)
    points=output.loc[~output.is_censored & output.prediction.notna()]
    censor=output.loc[output.is_censored & output.prediction.notna()]
    write(out.with_suffix('.manifest.json'),dict(training_manifest_sha256=digest(run/'manifest.json'),input_sha256=digest(args.data),
        output_sha256=digest(out),part=args.part,preprocessing_refit=False,input_cohort_n=len(output),predicted_n=int(output.prediction.notna().sum()),
        point_metrics=base.score(points),censored_n=len(censor),censor_violation_rate=float((censor.censor_violation_distance>0).mean()) if len(censor) else None))

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True); f=sub.add_parser('fit')
    for name in ('train','valid','expected-train-sha256','expected-valid-sha256','reference-manifest','parent-selection-lock','expected-parent-selection-sha256','boundary'):
        f.add_argument('--'+name,required=True)
    f.add_argument('--arm',choices=ARMS,required=True); f.add_argument('--route',choices=('W00','M00'),required=True)
    f.add_argument('--seed',type=int,choices=SEEDS,required=True); f.add_argument('--exact-train'); f.add_argument('--expected-exact-train-sha256')
    f.add_argument('--bin-scheme'); f.add_argument('--epochs',type=int,default=30); f.add_argument('--out'); f.add_argument('--device',default='cuda:0')
    i=sub.add_parser('predict')
    for name in ('run','selection-lock','data','out'): i.add_argument('--'+name,required=True)
    i.add_argument('--part',choices=('train','valid','test'),required=True); i.add_argument('--device',default='cuda:0')
    args=p.parse_args(); fit(args) if args.command=='fit' else infer(args)

if __name__=='__main__': main()

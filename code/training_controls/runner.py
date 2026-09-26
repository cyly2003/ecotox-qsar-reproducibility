"""Single-unit E10 runner. Caller exclusively owns queue, locks and HPO selection.

fit accepts train/valid files only. One STL head per unit ensures independent
preprocessing, initialization, optimizer and checkpoint. Inference takes a signed
selection lock and never fits; output directories/files may not already exist.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform
import sys
from .adapter import CANDIDATES, SEEDS, VARIANTS, prepare_group, transform, model_config

def runtime():
    from revision_pipeline import train as base
    from revision_pipeline.train_contract import digest, read_json, write_json, full_contract
    return base, digest, read_json, write_json, full_contract

def identity(digest):
    import qsar_tl
    import revision_pipeline.train as base
    paths = list(Path(qsar_tl.__file__).parent.rglob('*.py'))
    paths += [Path(__file__).with_name(name) for name in ('__init__.py','adapter.py','runner.py')]
    paths += [Path(base.__file__), Path(base.__file__).with_name('train_contract.py')]
    return {str(p.relative_to(Path(qsar_tl.__file__).parents[2])).replace('\\','/'): digest(p) for p in paths}

def group_keys(frame):
    if 'split_group_id' not in frame or frame.split_group_id.isna().any():
        raise ValueError('A registered split_group_id is required')
    return set(frame.split_group_id.astype(str))

def require_boundary(frame, boundary):
    if 'boundary_id' not in frame or frame.boundary_id.isna().any() or set(frame.boundary_id.astype(str)) != {boundary}:
        raise ValueError('Physical boundary_id differs from registered --boundary')

def expected_inputs(args, digest):
    for part in ('train', 'valid'):
        expected = getattr(args, f'expected_{part}_sha256')
        if not expected or digest(getattr(args,part)) != expected:
            raise ValueError(f'{part} does not match preregistered SHA-256')

def build_contract(args, legacy, full_contract, read_json):
    c = full_contract(args.reference_manifest, args.route)
    c.update(CANDIDATES[args.candidate])
    ref = read_json(args.reference_manifest)
    if args.route == 'W00':
        if not args.bin_scheme:
            raise ValueError('Original W00 requires explicit frozen toxicity-bin scheme')
        fields = legacy.ToxicityBinningConfig.__dataclass_fields__
        c['toxicity_binning_config'] = {k:v for k,v in ref['toxicity_binning'].items() if k in fields}
        c['toxicity_bin_scheme'] = legacy.load_toxicity_bin_scheme(args.bin_scheme)
        c['toxicity_bin_count'] = legacy.toxicity_bin_class_count(c['toxicity_bin_scheme'])
        if c['toxicity_bin_count'] != ref['toxicity_binning']['class_count']:
            raise ValueError('Bin scheme class count differs from submitted model')
        c['toxicity_bin_loss_weight'] = c['toxicity_binning_config']['loss_weight']
    else:
        # Submitted M00 retains the classifier module, but Stage3 gives it zero
        # objective weight. Keep the module/parameter count and RNG draw order.
        c.update(toxicity_bin_count=ref['toxicity_binning']['class_count'], toxicity_bin_loss_weight=0.)
    return c

def epoch(model, loader, config, torch, legacy, base, loss_fn, device, optimizer=None):
    model.train(optimizer is not None)
    weighted = 0.; count = 0
    for batch in loader:
        b = base.move_batch(batch, torch, device)
        if optimizer is not None: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            outputs = model(b['molecular_numeric'], b['fingerprint'], b['categorical_ids'], adapter_ids=b['adapter_id'])
            loss = legacy.batch_weighted_loss(lambda *a, **kw: outputs,
                b['molecular_numeric'], b['fingerprint'], b['categorical_ids'], b['adapter_id'],
                b['target_value'], b['task_head'], loss_fn, config, device,
                toxicity_bin_index=b.get('toxicity_bin_index'),
                toxicity_bin_loss_weight=config.toxicity_bin_loss_weight)
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Non-finite loss')
            if optimizer is not None:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                if not bool(torch.isfinite(norm)): raise FloatingPointError('Non-finite gradient')
                optimizer.step()
        n = len(b['task_head']); weighted += float(loss.detach().cpu()) * n; count += n
    return weighted / count if count else None

def fit(args):
    if platform.system() != 'Linux': raise RuntimeError('Training is remote Linux only')
    base,digest,read,write,full = runtime()
    np,pd,torch,rdkit,legacy,Training,collate,seed_fn,Config,Network = base.runtime()
    if not args.device.startswith('cuda') or not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    if args.epochs not in (1,30): raise ValueError('Use 1 epoch smoke or 30 epoch registered max')
    expected_inputs(args, digest)
    out = Path(args.out or os.environ['REVISION_ATTEMPT_DIR']); out.mkdir(parents=True, exist_ok=False)
    contract = build_contract(args, legacy, full, read)
    train,valid = getattr(args, '_loaded_frames', (None,None))
    if train is None: train = base.load_frame(args.train, args.route, 'train')
    if valid is None: valid = base.load_frame(args.valid, args.route, 'valid')
    require_boundary(train,args.boundary); require_boundary(valid,args.boundary)
    if group_keys(train) & group_keys(valid): raise ValueError('Train/valid registered group overlap')
    if base.source_ids(train) & base.source_ids(valid): raise ValueError('Train/valid source overlap')
    audit = {'train_ids':sorted(train.stable_record_id.astype(str)), 'valid_ids': sorted(valid.stable_record_id.astype(str)),
             'development_groups': sorted(group_keys(train) | group_keys(valid)),
             'development_sources': sorted(base.source_ids(train) | base.source_ids(valid))}
    seed_fn(args.seed); np.random.seed(args.seed)
    encoder = getattr(args, '_molecular_encoder', None) or base.encoder_for(legacy)
    group = prepare_group(train, valid, args.variant, contract, legacy, encoder, args.head)
    pre = group['preprocessing']; samples = group['train_samples']; vsamples = group['valid_samples']
    if args.route == 'W00' and args.variant.startswith('MTL_') and not any(s.get('toxicity_bin_index', -1) >= 0 for s in samples):
        raise ValueError('W00 auxiliary classification has zero eligible train labels; audit schema')
    weights = legacy.resolve_task_weights(samples, train_indices=list(range(len(samples))) if args.route=='W00' else [],
        train_cfg=contract['task_weighting_config'], task_heads=tuple(pre['task_heads']))
    cfg = Training(epochs=args.epochs, batch_size=contract['batch_size'], learning_rate=contract['learning_rate'],
        weight_decay=contract['weight_decay'], gradient_clip_norm=contract['gradient_clip_norm'],
        scheduler=contract['scheduler'], huber_delta=contract['huber_delta'], mse_loss_weight=0.,
        task_weights=weights, device=args.device, seed=args.seed,
        toxicity_bin_loss_weight=contract['toxicity_bin_loss_weight'], toxicity_binning_mode='aux_classification')
    mc = model_config(pre, contract, Config); model = Network(mc).to(args.device)
    loader = torch.utils.data.DataLoader(samples, batch_size=cfg.batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(args.seed), collate_fn=collate)
    vl = torch.utils.data.DataLoader(vsamples, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate) if vsamples else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = legacy.build_scheduler(optimizer,cfg,args.epochs); loss_fn=legacy.build_regression_loss(cfg)
    best=math.inf; best_epoch=0; stale=0; history=[]; state=None
    for e in range(1,args.epochs+1):
        tl=epoch(model,loader,cfg,torch,legacy,base,loss_fn,args.device,optimizer)
        val=epoch(model,vl,cfg,torch,legacy,base,loss_fn,args.device) if vl is not None else None
        if val is None or val < best-contract['min_delta']:
            if val is not None: best=val
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; best_epoch=e; stale=0
        else: stale+=1
        history.append(dict(epoch=e,train_loss=tl,valid_loss=val,best_epoch=best_epoch,
                            train_sample_visits=len(samples),optimizer_updates=len(loader)))
        print(json.dumps(history[-1]),flush=True)
        legacy.step_scheduler(scheduler,val if val is not None else tl)
        if vl is not None and stale>=contract['patience']: break
    model.load_state_dict(state,strict=True); torch.save(state,out/'model.pt')
    write(out/'preprocessing.json',pre); write(out/'boundary_audit.json',audit)
    pd.DataFrame(history).to_csv(out/'history.csv',index=False)
    metrics=None
    if vsamples:
        predictions=base.predict_rows(model,group['valid_frame'],vsamples,pre,args.device,torch,pd,collate,cfg.batch_size)
        predictions.to_parquet(out/'valid_predictions.parquet',index=False); metrics=base.score(predictions)
    manifest=dict(schema='e10_training_controls_v1',status='smoke_complete' if args.epochs==1 else 'formal_complete',
        route=args.route,boundary=args.boundary,variant=args.variant,seed=args.seed,candidate=args.candidate,head=args.head,
        target_scale='ptox_mol_l' if args.route=='W00' else 'neg_log10_mol_kg',
        contract=contract,model_config=asdict(mc),training_config=asdict(cfg),
        train_n=len(samples),valid_n=len(vsamples),max_epochs=args.epochs,best_epoch=best_epoch,
        auxiliary_bin_eligible_train_n=sum(s.get('toxicity_bin_index',-1)>=0 for s in samples),
        parameter_count=sum(p.numel() for p in model.parameters()),valid_metrics=metrics,
        selection_status=group['selection_status'],test_loaded_rows=0,test_labels_used_for_selection=False,
        train_sha256=digest(args.train),valid_sha256=digest(args.valid),reference_sha256=digest(args.reference_manifest),
        expected_train_sha256=args.expected_train_sha256,expected_valid_sha256=args.expected_valid_sha256,
        code_identity=identity(digest),artifacts={p.name:digest(p) for p in out.iterdir() if p.is_file()},
        environment=dict(python=sys.version,torch=torch.__version__,rdkit=rdkit.__version__,platform=platform.platform()))
    write(out/'manifest.json',manifest)

def fit_bundle(args):
    """One process, independently fitted STL tasks. Cached rows are not fit state."""
    import copy
    import hashlib
    import gc
    if not args.variant.startswith('STL_'):
        raise ValueError('fit-bundle is for STL_FULL/STL_MOL only')
    if args.head is not None: raise ValueError('Bundle discovers all point-train heads; --head forbidden')
    base,digest,read,write,full=runtime()
    expected_inputs(args,digest)
    train=base.load_frame(args.train,args.route,'train'); valid=base.load_frame(args.valid,args.route,'valid')
    require_boundary(train,args.boundary); require_boundary(valid,args.boundary)
    np,pd,torch,rdkit,legacy,Training,collate,seed_fn,Config,Network=base.runtime()
    encoder=base.encoder_for(legacy)  # deterministic RDKit cache only
    heads=sorted(train.loc[~train.is_censored,'model_head'].astype(str).unique())
    out=Path(args.out or os.environ['REVISION_ATTEMPT_DIR'])
    if args.resume:
        if not out.is_dir(): raise ValueError('--resume requires an existing bundle directory')
    else:
        out.mkdir(parents=True,exist_ok=False)
    contract={'variant':args.variant,'candidate':args.candidate,'seed':args.seed,'route':args.route,'boundary':args.boundary,
              'train_sha256':args.expected_train_sha256,'valid_sha256':args.expected_valid_sha256,
              'reference_sha256':digest(args.reference_manifest),'max_epochs':args.epochs,'code_identity':identity(digest),'heads':heads}
    header=out/'bundle_contract.json'
    if args.resume and header.exists():
        if read(header)!=contract: raise ValueError('Existing bundle has a different run contract')
    else:
        if args.resume: raise ValueError('Cannot resume bundle without its original contract')
        if any(out.iterdir()): raise ValueError('Unidentified nonempty bundle output')
        write(header,contract)
    records=[]; outputs=[]; failed=[]
    for head in heads:
        slug=hashlib.sha256(head.encode()).hexdigest()[:20]; task_root=out/'heads'/slug
        task_root.mkdir(parents=True,exist_ok=True)
        chosen=None
        for prior in sorted(task_root.glob('attempt_*/manifest.json')):
            m=read(prior)
            if m['status'] not in ('formal_complete','smoke_complete'): continue
            for key in ('variant','candidate','seed','route','boundary','train_sha256','valid_sha256','max_epochs','code_identity'):
                if m[key]!=contract[key]: raise ValueError(f'Completed head contract mismatch: {head} {key}')
            if m['head']!=head: raise ValueError('Stored head differs from stable bundle head identity')
            if all(digest(prior.parent/name)==sha for name,sha in m['artifacts'].items()):
                chosen=prior.parent; break
        if chosen is None:
            attempts=sorted(task_root.glob('attempt_*'))
            if len(attempts)>=3:
                failed.append({'head':head,'reason':'EXHAUSTED_INITIAL_PLUS_TWO_RETRIES','attempts':len(attempts)})
                write(out/'bundle_progress.json',{'completed_heads':len(records),'total_heads':len(heads),'models':records,'failed':failed})
                continue
            chosen=task_root/f'attempt_{len(attempts)+1:02d}'
            child=copy.copy(args); child.head=head; child.out=str(chosen)
            child._loaded_frames=(train,valid); child._molecular_encoder=encoder
            try:
                fit(child)
            except Exception as exc:
                chosen.mkdir(parents=True,exist_ok=True)
                failure={'head':head,'reason':f'{type(exc).__name__}: {exc}','attempt':len(attempts)+1,
                         'status':'FAILED','retry_available':len(attempts)+1<3}
                write(chosen/'FAILED.json',failure); failed.append(failure)
                write(out/'bundle_progress.json',{'completed_heads':len(records),'total_heads':len(heads),'models':records,'failed':failed})
                gc.collect(); torch.cuda.empty_cache()
                continue
        manifest=read(chosen/'manifest.json')
        records.append({'head':head,'run':str(chosen.relative_to(out)),'manifest_sha256':digest(chosen/'manifest.json'),
                        'selection_status':manifest['selection_status'],'train_n':manifest['train_n'],'valid_n':manifest['valid_n']})
        if (chosen/'valid_predictions.parquet').exists(): outputs.append(pd.read_parquet(chosen/'valid_predictions.parquet'))
        write(out/'bundle_progress.json',{'completed_heads':len(records),'total_heads':len(heads),'models':records,'failed':failed})
        gc.collect(); torch.cuda.empty_cache()
    metrics=None
    if outputs:
        pred=pd.concat(outputs,ignore_index=True).sort_values('stable_record_id').reset_index(drop=True)
        if pred.stable_record_id.duplicated().any(): raise ValueError('Duplicate STL validation prediction identities')
        pred.to_parquet(out/'valid_predictions.parquet',index=False); metrics=base.score(pred)
    write(out/'bundle_manifest.json',{'schema':'e10_stl_bundle_v1','status':'incomplete' if failed else ('smoke_complete' if args.epochs==1 else 'formal_complete'),
        **contract,'models':records,'failed':failed,'completed_heads':len(records),'total_heads':len(heads),
        'valid_metrics':None if failed else metrics,'partial_valid_metrics_not_selection_eligible':metrics if failed else None,
        'test_loaded_rows':0,'test_labels_used_for_selection':False,
        'validation_predictions_sha256':digest(out/'valid_predictions.parquet') if outputs else None,
        'fitted_state_shared_between_heads':False,'molecular_cache_is_deterministic_unsupervised':True})
    if failed: raise RuntimeError(f'Bundle incomplete: {len(failed)} failed heads; resume the SAME directory with --resume')

def infer(args):
    base,digest,read,write,full = runtime()
    np,pd,torch,rdkit,legacy,Training,collate,seed_fn,Config,Network=base.runtime()
    run=Path(args.run); manifest=read(run/'manifest.json'); lock=read(args.selection_lock)
    if manifest['status']!='formal_complete': raise ValueError('Smoke cannot enter formal inference')
    if lock.get('test_used_for_selection') is not False or digest(run/'manifest.json') not in lock.get('approved_manifest_sha256',[]):
        raise ValueError('Manifest absent from pre-test selection lock')
    if any(lock.get(k)!=manifest[k] for k in ('route','boundary')):
        raise ValueError('Selection lock route/boundary mismatch')
    approved=lock.get('input_sha256_by_partition',{}).get(args.part,[])
    if isinstance(approved,str): approved=[approved]
    if digest(args.data) not in approved: raise ValueError('Inference cohort input hash absent from selection lock')
    if manifest['code_identity'] != identity(digest): raise ValueError('Source drift')
    if manifest['environment']['rdkit'] != rdkit.__version__: raise ValueError('RDKit drift')
    for name,sha in manifest['artifacts'].items():
        if digest(run/name)!=sha: raise ValueError(f'Artifact drift: {name}')
    pre=read(run/'preprocessing.json'); audit=read(run/'boundary_audit.json')
    frame=base.load_frame(args.data,manifest['route'],args.part)
    require_boundary(frame,manifest['boundary'])
    if args.part=='test':
        if set(frame.stable_record_id.astype(str)) & set(audit['train_ids']+audit['valid_ids']): raise ValueError('Test identity overlap')
        if group_keys(frame) & set(audit['development_groups']): raise ValueError('Test registered group overlap')
        if base.source_ids(frame) & set(audit['development_sources']): raise ValueError('Test source overlap')
    elif digest(args.data)!=manifest[args.part+'_sha256']:
        raise ValueError('Development data differs from training inputs')
    if manifest['variant'].startswith('STL_'):
        frame=frame.loc[frame.model_head.astype(str).eq(str(manifest['head']))].reset_index(drop=True)
    raw=frame.copy()
    known=raw.model_head.astype(str).isin(pre['task_heads']) & ~raw.is_censored
    frame=raw.loc[known].reset_index(drop=True)
    samples=transform(frame,pre,legacy,base.encoder_for(legacy)) if len(frame) else []
    model=Network(Config(**manifest['model_config'])).to(args.device)
    model.load_state_dict(torch.load(run/'model.pt',map_location=args.device,weights_only=True),strict=True)
    if len(frame):
        calculated=base.predict_rows(model,frame,samples,pre,args.device,torch,pd,collate,manifest['contract']['batch_size'],embeddings=args.command=='export')
        columns=['stable_record_id','prediction']+[c for c in calculated if c.startswith('z_')]
        output=raw.merge(calculated[columns],on='stable_record_id',how='left',validate='one_to_one')
    else:
        output=raw.copy(); output['prediction']=float('nan')
        if args.command=='export':
            for k in range(128): output[f'z_{k:03d}']=float('nan')
    output['prediction_status']='NOT_ESTIMABLE_NO_TRAIN_LABEL'
    output.loc[output.is_censored,'prediction_status']='NOT_POINT_EVALUATION'
    output.loc[output.prediction.notna(),'prediction_status']='PREDICTED'
    metric_rows=output.loc[output.prediction.notna() & ~output.is_censored]
    out=Path(args.out)
    if out.exists(): raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True); output.to_parquet(out,index=False)
    write(out.with_suffix('.manifest.json'),dict(training_manifest_sha256=digest(run/'manifest.json'),
        selection_lock_sha256=digest(args.selection_lock),input_sha256=digest(args.data),output_sha256=digest(out),
        part=args.part,preprocessing_refit=False,metrics=base.score(metric_rows),input_cohort_n=len(raw),
        predicted_n=int(output.prediction.notna().sum()),
        no_train_label_n=int(output.prediction_status.eq('NOT_ESTIMABLE_NO_TRAIN_LABEL').sum()),
        censored_n=int(output.is_censored.sum()),n=len(output),
        output_scope='single_head_cohort_requires_full_STL_assembly' if manifest['variant'].startswith('STL_') else 'complete_route_cohort'))

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    for command in ('fit','fit-bundle'):
        f=sub.add_parser(command)
        for name in ('train','valid','reference-manifest','boundary','expected-train-sha256','expected-valid-sha256'): f.add_argument('--'+name,required=True)
        f.add_argument('--route',choices=['W00','M00'],required=True); f.add_argument('--variant',choices=VARIANTS,required=True)
        f.add_argument('--seed',type=int,choices=SEEDS,required=True); f.add_argument('--candidate',choices=CANDIDATES,required=True)
        f.add_argument('--head'); f.add_argument('--bin-scheme'); f.add_argument('--epochs',type=int,default=30)
        f.add_argument('--out'); f.add_argument('--device',default='cuda:0')
        if command=='fit-bundle': f.add_argument('--resume',action='store_true')
    for command in ('predict','export'):
        i=sub.add_parser(command)
        for name in ('run','selection-lock','data','out'): i.add_argument('--'+name,required=True)
        i.add_argument('--part',choices=['train','valid','test'],required=True); i.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    if args.command=='fit': fit(args)
    elif args.command=='fit-bundle': fit_bundle(args)
    else: infer(args)

if __name__=='__main__': main()

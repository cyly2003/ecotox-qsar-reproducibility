"""Revision-only thin runner. Fit never accepts test data; inference never refits.

Architecture, input processing, exact loss and task-balanced objective reuse qsar_tl.
Censor extension adds squared distance to a p-scale interval (hinge, not Tobit).
Imports of torch occur only in fit/predict commands; contract tests remain torch-free.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import math
import json
from pathlib import Path
import platform
import sys
from .train_contract import bounds_contract, canonical_hash, digest, full_contract, read_json, write_json


def runtime():
    import numpy as np
    import pandas as pd
    import torch
    import rdkit
    from qsar_tl.training import deep_experiment as legacy
    from qsar_tl.training.deep_train import DeepTrainingConfig, collate_aggregated_task_batch, set_torch_seed
    from qsar_tl.modeling.network import DeepModelConfig, EcotoxMultiTaskNetwork
    return np, pd, torch, rdkit, legacy, DeepTrainingConfig, collate_aggregated_task_batch, set_torch_seed, DeepModelConfig, EcotoxMultiTaskNetwork


def code_identity():
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / 'qsar_tl').rglob('*.py'))
    files += [root / 'revision_pipeline' / name for name in ('train.py', 'train_contract.py')]
    return {str(p.relative_to(root)).replace('\\', '/'): digest(p) for p in files}


def load_frame(path, route, part):
    import pandas as pd
    from qsar_tl.training.deep_experiment import CATEGORICAL_COLUMNS
    from qsar_tl.training.baseline import add_duration_nonlinear_features
    frame = pd.read_parquet(path)  # physical split files: there is no all-data/test handle in fit
    required = {'route', 'stable_record_id', 'model_head', 'assigned_split', 'is_censored',
                'target', 'bound_lower', 'bound_upper', 'smiles', 'canonical_parent', 'latin_name',
                'target_name', 'target_family', 'effect_level_x', 'duration_bin_h',
                'source_result_ids_json', *CATEGORICAL_COLUMNS}
    missing = required - set(frame)
    if route=='W00' and 'task_group' not in frame:
        missing.add('task_group')
    if missing:
        raise ValueError(f'Missing input contract columns: {sorted(missing)}')
    if frame.empty or set(frame.route) != {route} or set(frame.assigned_split) != {part}:
        raise ValueError(f'Expected nonempty physical {route}/{part} file.')
    if frame.stable_record_id.isna().any() or frame.stable_record_id.duplicated().any():
        raise ValueError('Stable identities must be non-null and unique.')
    if frame.model_head.isna().any() or frame.model_head.astype(str).str.strip().eq('').any():
        raise ValueError('Missing model_head.')
    if not frame.is_censored.isin([True, False, 0, 1]).all():
        raise ValueError('is_censored must be boolean or 0/1, never a string.')
    frame['is_censored'] = frame.is_censored.astype(bool)
    expected = 'ptox_mol_l' if route == 'W00' else 'neg_log10_mol_kg'
    if set(frame.target_name) != {expected}:
        raise ValueError(f'Wrong target scale: expected {expected}')
    for row in frame.itertuples():
        bounds_contract(row.is_censored, row.target, row.bound_lower, row.bound_upper)
    frame['task_head'] = frame.model_head.astype(str)
    frame['split_part'] = part
    frame['aggregate_id'] = frame.stable_record_id.astype(str)
    return add_duration_nonlinear_features(frame).reset_index(drop=True)


def encoder_for(legacy):
    """Reuse exact RDKit8 calculation, forbid the legacy text-fingerprint fallback."""
    from rdkit import Chem
    encoder = legacy.MolecularFeatureBuilder(fingerprint_size=512)
    if encoder.source != 'rdkit':
        raise RuntimeError('RDKit is mandatory; fallback fingerprints are forbidden.')
    cache = {}
    def encode(smiles):
        text = str(smiles)
        if text not in cache:
            if Chem.MolFromSmiles(text) is None:
                raise ValueError(f'Unparseable admitted SMILES: {text}')
            cache[text] = encoder._encode_rdkit(text)
        return cache[text]
    encoder.encode = encode
    return encoder


def fit_preprocessing(train, contract, legacy, encoder):
    point = train.loc[~train.is_censored].copy()
    if point.empty:
        raise ValueError('No uncensored training rows available to fit preprocessing.')
    point['target_value'] = point.target.astype(float)
    spec = replace(legacy.ABLATION_SPECS['full'], use_medium_adapter=False)
    names = tuple(contract['descriptor_names'])
    category = legacy.fit_categorical_maps(point, ablation=spec, min_count=contract['categorical_min_count'])
    stats = legacy.fit_numeric_stats(point, encoder, descriptor_names=names, ablation=spec)
    zscore = legacy.fit_zscore_correction(point,encoder,numeric_stats=stats,
        feature_names=legacy.build_numeric_feature_names(8,descriptor_names=names),descriptor_names=names,
        config=legacy.ZScoreCorrectionConfig(enabled=contract['zscore_enabled'],threshold=contract['zscore_threshold']),ablation=spec)
    scaler = legacy.fit_target_scaler(point, target_column='target_value', mode='per_task_target', fit_indices=list(range(len(point))))
    heads = sorted(point.model_head.unique().tolist())
    return {'categorical_maps': category, 'numeric_stats': stats, 'target_scaler': scaler.to_manifest(),
            'descriptor_names': list(names), 'task_heads': heads,
            'fit_ids': sorted(point.stable_record_id.astype(str).tolist()),
            'zscore_correction':zscore.to_manifest(),
            'zscore_enabled': contract['zscore_enabled'], 'zscore_threshold': contract['zscore_threshold'],
            'fit_partition': 'train_uncensored_point_only'}


def make_samples(frame, pre, legacy, encoder):
    spec = replace(legacy.ABLATION_SPECS['full'], use_medium_adapter=False)
    scaler = legacy.TargetScaler(**pre['target_scaler'])
    zfields={k:v for k,v in pre['zscore_correction'].items() if k!='method'}
    zscore = legacy.ZScoreCorrection(**zfields)
    work = frame.copy()
    # build_deep_samples needs a scalar; this placeholder never enters exact loss or metrics.
    work['target_value'] = work.target.where(~work.is_censored, 0.0)
    samples = legacy.build_deep_samples(work, encoder=encoder, descriptor_names=tuple(pre['descriptor_names']),
        categorical_maps=pre['categorical_maps'], adapter_map={}, numeric_stats=pre['numeric_stats'],
        target_column='target_value', target_scaler=scaler, zscore_correction=zscore, ablation=spec)
    for sample, row in zip(samples, frame.to_dict('records')):
        key = legacy.target_scale_key(row | {'task_head': row['model_head']}, scaler.mode)
        if key not in scaler.stats:
            raise ValueError(f'Task has no uncensored-train target scaler: {key}')
        lo, hi = bounds_contract(row['is_censored'], row['target'], row['bound_lower'], row['bound_upper'])
        sample['bound_lower_scaled'] = scaler.transform(key, lo) if math.isfinite(lo) else -math.inf
        sample['bound_upper_scaled'] = scaler.transform(key, hi) if math.isfinite(hi) else math.inf
        sample['censored_direction_id'] = int(row['is_censored'])  # exclusion mask only; not direction/input
        sample['is_censored'] = row['is_censored']
        sample['stable_record_id'] = row['stable_record_id']
        sample['target_value_raw'] = None if row['is_censored'] else float(row['target'])
    return samples


def make_collate(torch, base_collate):
    def collate(rows):
        batch = base_collate(rows)
        batch['bound_lower_scaled'] = torch.tensor([r['bound_lower_scaled'] for r in rows], dtype=torch.float32)
        batch['bound_upper_scaled'] = torch.tensor([r['bound_upper_scaled'] for r in rows], dtype=torch.float32)
        return batch
    return collate


def model_config(pre, contract, heads, Config):
    return Config(numeric_dim=22, fingerprint_dim=512, task_heads=tuple(heads), descriptor_count=8,
                  categorical_cardinalities={k: max(v.values()) + 1 for k,v in pre['categorical_maps'].items()},
                  hidden_dims=tuple(contract['hidden_dims']), dropout=contract['dropout'],
                  effect_level_numeric_indices=(8,9,10,11), use_molecular_residual=True,
                  use_adapters=False, adapter_count=0, fusion_mode='concat')


def move_batch(batch, torch, device):
    result = {}
    for k,v in batch.items():
        result[k] = v.to(device) if torch.is_tensor(v) else ({a:b.to(device) for a,b in v.items()} if k == 'categorical_ids' else v)
    return result


def epoch_step(model, loader, config, torch, legacy, loss_fn, device, optimizer=None, alpha=0.0):
    model.train(optimizer is not None)
    weighted, count = 0.0, 0
    for batch in loader:
        b = move_batch(batch, torch, device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            outputs = model(b['molecular_numeric'], b['fingerprint'], b['categorical_ids'], adapter_ids=b['adapter_id'])
            # Reuse original task-balanced exact objective; censor IDs never enter forward.
            loss = legacy.batch_weighted_loss(lambda *a, **kw: outputs, b['molecular_numeric'], b['fingerprint'], b['categorical_ids'],
                b['adapter_id'], b['target_value'], b['task_head'], loss_fn, config, device,
                censored_direction_id=b['censored_direction_id'], censored_loss_weight=0.0)
            mask_c = b['censored_direction_id'] != 0
            if alpha > 0 and bool(mask_c.any()):
                parts = []
                for head in sorted(set(b['task_head'])):
                    mask = mask_c & torch.tensor([x == head for x in b['task_head']], device=device)
                    if bool(mask.any()):
                        p = outputs[head][mask]
                        low, high = b['bound_lower_scaled'][mask], b['bound_upper_scaled'][mask]
                        parts.append((torch.relu(low-p).square() + torch.relu(p-high).square()).mean())
                loss = loss + alpha * torch.stack(parts).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Non-finite loss.')
            if optimizer is not None:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError('Non-finite gradient.')
                optimizer.step()
        n = len(b['task_head']); weighted += float(loss.detach().cpu()) * n; count += n
    return weighted / count if count else None


def predict_rows(model, frame, samples, pre, device, torch, pd, base_collate, batch_size, embeddings=False):
    from qsar_tl.training.deep_experiment import TargetScaler, target_scale_key
    scaler = TargetScaler(**pre['target_scaler']); pred=[]; zs=[]
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            subset = samples[start:start+batch_size]
            b = move_batch(base_collate(subset), torch, device)
            outputs = model(b['molecular_numeric'], b['fingerprint'], b['categorical_ids'], adapter_ids=b['adapter_id'])
            if embeddings:
                zs.extend(model.encode_shared(b['molecular_numeric'], b['fingerprint'], b['categorical_ids'], adapter_ids=b['adapter_id']).cpu().numpy().tolist())
            for i, sample in enumerate(subset):
                row = frame.iloc[start+i]
                key = target_scale_key({'task_head': row.model_head, 'target_family': row.target_family, 'target_name': row.target_name}, scaler.mode)
                pred.append(scaler.inverse_transform(key, float(outputs[sample['task_head']][i].cpu())))
    result = frame[['route','stable_record_id','canonical_parent','latin_name','model_head','assigned_split','source_result_ids_json','is_censored','target','bound_lower','bound_upper']].copy()
    result['prediction'] = pred
    if embeddings:
        for k in range(128):
            result[f'z_{k:03d}'] = [z[k] for z in zs]
    return result


def score(frame):
    import numpy as np
    point = frame.loc[~frame.is_censored].copy()
    if 'comparison_eligible' in point:
        point=point.loc[point.comparison_eligible]
    if point.empty:
        return {'n': 0, 'task_macro_mae': None, 'mae': None}
    point['ae'] = (point.prediction - point.target).abs()
    return {'n': len(point), 'heads': int(point.model_head.nunique()),
            'task_macro_mae': float(point.groupby('model_head').ae.mean().mean()),
            'mae': float(point.ae.mean()), 'rmse': float(np.sqrt(((point.prediction-point.target)**2).mean()))}


def boundary_keys(frame, boundary):
    label = boundary.lower()
    if 'combo' in label or 'combination' in label:
        if frame[['canonical_parent','latin_name','model_head']].isna().any().any():
            raise ValueError('Combination boundary requires resolved structural, species and head identities.')
        return [canonical_hash(list(row)) for row in frame[['canonical_parent','latin_name','model_head']].astype(str).itertuples(index=False,name=None)]
    if 'parent' in label and 'random' not in label:
        if frame.canonical_parent.isna().any() or frame.canonical_parent.astype(str).str.strip().eq('').any():
            raise ValueError('Parent boundary requires resolved canonical parents.')
        return frame.canonical_parent.astype(str).tolist()
    if 'random' in label:
        return frame.stable_record_id.astype(str).tolist()
    raise ValueError(f'Unknown registered boundary: {boundary}')


def source_ids(frame):
    identities=set()
    for raw in frame.source_result_ids_json:
        values=json.loads(raw) if isinstance(raw,str) else raw
        if not isinstance(values,(list,tuple)) or not values:
            raise ValueError('source_result_ids_json must be a nonempty list.')
        for value in values:
            if value is None or str(value).strip()=='': raise ValueError('Missing source result identity.')
            text=str(value).strip()
            # ECOTOX result IDs are numeric in some source tables and strings in others.
            try:
                number=int(text)
                text=str(number)
            except ValueError:
                if isinstance(value,float) and value.is_integer(): text=str(int(value))
            identities.add(text)
    return identities


def fit(args):
    if platform.system()!='Linux':
        raise RuntimeError('Training is authorized on the remote Linux host only.')
    np,pd,torch,rdkit,legacy,Training,base_collate,seed_fn,Config,Network = runtime()
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Revision fit requires the authorized remote CUDA environment.')
    if args.seed != 42 or args.epochs not in (1,30):
        raise ValueError('Locked seed42; epochs=1 smoke or 30 formal only.')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    contract=full_contract(args.reference_manifest,args.route)
    train=load_frame(args.data,args.route,'train'); valid=load_frame(args.valid,args.route,'valid')
    if set(train.stable_record_id) & set(valid.stable_record_id):
        raise ValueError('Train/valid identity overlap.')
    train_source=source_ids(train); valid_source=source_ids(valid)
    if train_source & valid_source: raise ValueError('Train/valid source result identities overlap.')
    train_keys=boundary_keys(train,args.boundary); valid_keys=boundary_keys(valid,args.boundary)
    if set(train_keys) & set(valid_keys):
        raise ValueError('Train/valid registered group overlap.')
    boundary_audit={'train_ids':sorted(train.stable_record_id.astype(str).tolist()),
                    'valid_ids':sorted(valid.stable_record_id.astype(str).tolist()),
                    'development_group_keys':sorted(set(train_keys+valid_keys)),
                    'train_source_result_ids':sorted(train_source),'valid_source_result_ids':sorted(valid_source)}
    encoder=encoder_for(legacy); pre=fit_preprocessing(train,contract,legacy,encoder)
    eligible_heads=set(pre['task_heads']); excluded={}
    for label,frame in [('train',train),('valid',valid)]:
        excluded[label]=int((~frame.model_head.isin(eligible_heads)).sum())
    train=train.loc[train.model_head.isin(eligible_heads)].reset_index(drop=True)
    valid=valid.loc[valid.model_head.isin(eligible_heads) & ~valid.is_censored].reset_index(drop=True)
    if valid.empty:
        raise ValueError('No common uncensored valid records for selection.')
    if args.variant=='uncensored_point':
        train=train.loc[~train.is_censored].reset_index(drop=True)
    samples=make_samples(train,pre,legacy,encoder); vsamples=make_samples(valid,pre,legacy,encoder)
    task_weights=legacy.resolve_task_weights(samples,
        train_indices=[i for i,s in enumerate(samples) if not s['is_censored']] if args.route=='W00' else [],
        train_cfg=contract['task_weighting_config'],task_heads=tuple(pre['task_heads']))
    write_json(out/'preprocessing.json',pre)
    write_json(out/'boundary_audit.json',boundary_audit)
    groups=[('full',pre['task_heads'])] if args.mode=='mtl' else [(canonical_hash(h)[:16],[h]) for h in pre['task_heads']]
    config=Training(epochs=args.epochs,batch_size=contract['batch_size'],learning_rate=contract['learning_rate'],
                    weight_decay=contract['weight_decay'],gradient_clip_norm=contract['gradient_clip_norm'],
                    scheduler=contract['scheduler'],huber_delta=contract['huber_delta'],mse_loss_weight=0,
                    task_weights=task_weights,device=args.device,seed=42)
    if contract['optimizer']!='adamw': raise ValueError('Expected AdamW.')
    history=[]; outputs=[]; model_records=[]
    for group,heads in groups:
        seed_fn(42); np.random.seed(42)
        mc=model_config(pre,contract,heads,Config); model=Network(mc).to(args.device)
        ts=[s for s in samples if s['task_head'] in heads]; vi=[i for i,s in enumerate(vsamples) if s['task_head'] in heads]
        vv=[vsamples[i] for i in vi]
        loader=torch.utils.data.DataLoader(ts,batch_size=config.batch_size,shuffle=True,num_workers=0,
                 generator=torch.Generator().manual_seed(42),collate_fn=make_collate(torch,base_collate))
        vl=torch.utils.data.DataLoader(vv,batch_size=config.batch_size,shuffle=False,num_workers=0,collate_fn=make_collate(torch,base_collate)) if vv else None
        optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
        scheduler=legacy.build_scheduler(optimizer,config,args.epochs); loss_fn=legacy.build_regression_loss(config)
        best=math.inf; state=None; best_epoch=0; stale=0
        for epoch in range(1,args.epochs+1):
            tl=epoch_step(model,loader,config,torch,legacy,loss_fn,args.device,optimizer,1.0 if args.variant=='censor_aware' else 0.0)
            value=epoch_step(model,vl,config,torch,legacy,loss_fn,args.device) if vl is not None else None
            if value is None or value<best-contract['min_delta']:
                if value is not None: best=value
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; best_epoch=epoch; stale=0
            else: stale+=1
            history.append({'model':group,'epoch':epoch,'train_loss':tl,'valid_loss':value,'best_epoch':best_epoch})
            print(f'{args.route} {args.mode} {args.variant} {group} epoch={epoch} train={tl:.6f} valid={value}',flush=True)
            legacy.step_scheduler(scheduler,value if value is not None else tl)
            if vl is not None and stale>=contract['patience']: break
        model.load_state_dict(state,strict=True)
        checkpoint=f'model_{group}.pt'; torch.save(state,out/checkpoint)
        model_records.append({'group':group,'heads':heads,'config':asdict(mc),'checkpoint':checkpoint,
            'checkpoint_sha256':digest(out/checkpoint),'train_n':len(ts),'valid_n':len(vv),'best_epoch':best_epoch,
            'parameter_count':sum(p.numel() for p in model.parameters()),
            'selection_status':'valid_best' if vv else 'no_valid_fixed_final_epoch_not_comparison_eligible'})
        if vv:
            outputs.append(predict_rows(model,valid.iloc[vi].reset_index(drop=True),vv,pre,args.device,torch,pd,base_collate,config.batch_size))
        del model,optimizer; torch.cuda.empty_cache()
    prediction=pd.concat(outputs,ignore_index=True).sort_values('stable_record_id').reset_index(drop=True)
    prediction.to_parquet(out/'valid_predictions.parquet',index=False)
    pd.DataFrame(history).to_csv(out/'history.csv',index=False)
    manifest={'schema':'revision_train_v1','status':'smoke_complete' if args.epochs==1 else 'formal_complete',
        'route':args.route,'boundary':args.boundary,'seed':42,'variant':args.variant,'mode':args.mode,
        'target_scale':'ptox_mol_l' if args.route=='W00' else 'neg_log10_mol_kg',
        'comparison_eligible_heads':sorted(valid.model_head.unique().tolist()),
        'max_epochs':args.epochs,'contract':contract,'models':model_records,
        'task_weights':task_weights,'task_weight_fit':'common_point_train_W00; original_Stage3_unit_weights_M00',
        'preprocessing_sha256':digest(out/'preprocessing.json'),'valid_predictions_sha256':digest(out/'valid_predictions.parquet'),
        'boundary_audit_sha256':digest(out/'boundary_audit.json'),
        'valid_metrics':score(prediction),'data_sha256':digest(args.data),'valid_sha256':digest(args.valid),
        'test_loaded_rows':0,'test_labels_used_for_selection':False,'censor_alpha':1.0 if args.variant=='censor_aware' else 0.0,
        'censor_margin':0.0,'censor_method':'p_scale_squared_interval_hinge_not_Tobit',
        'excluded_no_point_train_head':excluded,'code_identity':code_identity(),
        'environment':{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'rdkit':rdkit.__version__},
        'scientific_limitations':['single_seed42','legacy_response_QC_retained','legacy_medium_and_aggregation_retained'],
        'preprocessing_fit':'common_uncensored_train_only','stl_unit':'model_head','initialization':'independent_seed42_no_MTL_weights'}
    write_json(out/'manifest.json',manifest)


def lock_winner(args):
    import pandas as pd
    runs=[Path(p) for p in args.runs]; manifests=[read_json(p/'manifest.json') for p in runs]
    if len(runs)!=2 or {m['variant'] for m in manifests}!={'uncensored_point','censor_aware'}:
        raise ValueError('Exactly the two preregistered variants are required.')
    for key in ('route','boundary','seed','mode','data_sha256','valid_sha256','preprocessing_sha256','code_identity'):
        if manifests[0][key]!=manifests[1][key]: raise ValueError(f'Unpaired candidate contract: {key}')
    frames=[]
    for p,m in zip(runs,manifests):
        if m['status']!='formal_complete' or m['max_epochs']!=30 or m['test_loaded_rows']!=0:
            raise ValueError('Only complete 30-epoch-budget, test-unseen runs can be selected.')
        if digest(p/'valid_predictions.parquet')!=m['valid_predictions_sha256']: raise ValueError('Prediction hash mismatch.')
        f=pd.read_parquet(p/'valid_predictions.parquet').sort_values('stable_record_id').reset_index(drop=True)
        if set(f.assigned_split)!={'valid'} or f.is_censored.any(): raise ValueError('Selection requires point valid only.')
        frames.append(f)
    if not frames[0][['stable_record_id','model_head','target']].equals(frames[1][['stable_record_id','model_head','target']]):
        raise ValueError('Validation identities, tasks or targets differ.')
    scores=[score(f)['task_macro_mae'] for f in frames]
    idx=min(range(2),key=lambda i:(scores[i],manifests[i]['variant']!='uncensored_point'))
    chosen=runs[idx]; m=manifests[idx]
    lock={'schema':'revision_valid_winner_v1','route':m['route'],'boundary':m['boundary'],'seed':42,'mode':m['mode'],
          'criterion':'uncensored_valid_task_macro_MAE','tie_break':'uncensored_point','winner_variant':m['variant'],
          'winner_manifest_sha256':digest(chosen/'manifest.json'),'winner_run':str(chosen.resolve()),
          'candidates':[{'manifest_sha256':digest(p/'manifest.json'),'variant':x['variant'],'valid_task_macro_mae':s} for p,x,s in zip(runs,manifests,scores)],
          'test_used':False}
    if Path(args.out).exists(): raise FileExistsError(args.out)
    write_json(args.out,lock)


def predict(args):
    np,pd,torch,rdkit,legacy,Training,base_collate,seed_fn,Config,Network=runtime()
    run=Path(args.run); m=read_json(run/'manifest.json'); lock=read_json(args.winner_lock)
    manifest_hash=digest(run/'manifest.json')
    registered={lock['winner_manifest_sha256']}
    if args.report_registered_candidate:
        registered.update(c['manifest_sha256'] for c in lock.get('candidates',[]))
    if manifest_hash not in registered or lock['test_used'] is not False:
        raise ValueError('Run is neither the locked winner nor explicitly registered for secondary reporting.')
    if m['code_identity']!=code_identity(): raise ValueError('Runtime source differs from training source.')
    if m['environment']['rdkit']!=rdkit.__version__: raise ValueError('RDKit version differs from training.')
    if digest(run/'preprocessing.json')!=m['preprocessing_sha256']: raise ValueError('Preprocessor changed.')
    pre=read_json(run/'preprocessing.json'); frame=load_frame(args.data,m['route'],args.part)
    if digest(run/'boundary_audit.json')!=m['boundary_audit_sha256']: raise ValueError('Boundary audit changed.')
    audit=read_json(run/'boundary_audit.json')
    if args.part=='test':
        if source_ids(frame) & set(audit['train_source_result_ids']+audit['valid_source_result_ids']):
            raise ValueError('Test source result identities overlap development.')
        if set(frame.stable_record_id.astype(str)) & set(audit['train_ids']+audit['valid_ids']):
            raise ValueError('Test identity overlaps development.')
        if set(boundary_keys(frame,m['boundary'])) & set(audit['development_group_keys']):
            raise ValueError('Test group overlaps development in the registered boundary.')
    elif digest(args.data)!=m['data_sha256' if args.part=='train' else 'valid_sha256']:
        raise ValueError('Development export is not the exact physical training/validation file.')
    unknown=frame.loc[~frame.model_head.isin(pre['task_heads'])].copy()
    frame=frame.loc[frame.model_head.isin(pre['task_heads'])].reset_index(drop=True)
    encoder=encoder_for(legacy); samples=make_samples(frame,pre,legacy,encoder); outputs=[]
    for record in m['models']:
        if digest(run/record['checkpoint'])!=record['checkpoint_sha256']: raise ValueError('Checkpoint hash mismatch.')
        indices=[i for i,s in enumerate(samples) if s['task_head'] in record['heads']]
        if not indices: continue
        model=Network(Config(**record['config'])).to(args.device)
        model.load_state_dict(torch.load(run/record['checkpoint'],map_location=args.device,weights_only=True),strict=True)
        outputs.append(predict_rows(model,frame.iloc[indices].reset_index(drop=True),[samples[i] for i in indices],pre,
            args.device,torch,pd,base_collate,m['contract']['batch_size'],embeddings=args.command=='export-embeddings'))
        del model
    if not outputs: raise ValueError('No trained heads available for this partition.')
    out=Path(args.out)
    if out.exists(): raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    prediction=pd.concat(outputs,ignore_index=True).sort_values('stable_record_id').reset_index(drop=True)
    prediction['comparison_eligible']=prediction.model_head.isin(m['comparison_eligible_heads'])
    prediction.to_parquet(out,index=False)
    unknown[['stable_record_id','model_head']].to_csv(out.with_suffix('.untrained_heads.csv'),index=False)
    write_json(out.with_suffix('.manifest.json'),{'schema':'revision_prediction_v1','route':m['route'],'boundary':m['boundary'],
        'partition':args.part,'winner_lock_sha256':digest(args.winner_lock),'training_manifest_sha256':digest(run/'manifest.json'),
        'report_role':'primary_winner' if manifest_hash==lock['winner_manifest_sha256'] else 'registered_secondary_variant',
        'preprocessing_refit':False,'input_sha256':digest(args.data),'output_sha256':digest(out),'n':len(prediction),
        'untrained_head_rows':len(unknown),'metrics':score(prediction),
        'metrics_scope':'point rows with uncensored training and validation support; all rows retained in parquet',
        'fused_Z_dimension':128 if args.command=='export-embeddings' else None})


def lock_comparison(args):
    """Register one independent STL run using the MTL validation-winning variant."""
    lock=read_json(args.winner_lock); stl=Path(args.run); sm=read_json(stl/'manifest.json')
    winner=Path(args.winner_run or lock['winner_run']); wm=read_json(winner/'manifest.json')
    if digest(winner/'manifest.json')!=lock['winner_manifest_sha256'] or wm['mode']!='mtl' or sm['mode']!='stl':
        raise ValueError('Comparison requires the original MTL winner and an independent STL run.')
    if sm['status']!='formal_complete' or sm['max_epochs']!=30 or sm['test_loaded_rows']!=0:
        raise ValueError('STL comparison must be complete and test-unseen.')
    for key in ('route','boundary','seed','variant','data_sha256','valid_sha256','preprocessing_sha256','code_identity','task_weights'):
        if sm[key]!=wm[key]: raise ValueError(f'STL/MTL comparison mismatch: {key}')
    result={'schema':'revision_stl_comparison_lock_v1','route':sm['route'],'boundary':sm['boundary'],'seed':42,'mode':'stl',
            'winner_manifest_sha256':digest(stl/'manifest.json'),'winner_run':str(stl.resolve()),
            'parent_mtl_winner_manifest_sha256':digest(winner/'manifest.json'),
            'parent_mtl_winner_lock_sha256':digest(args.winner_lock),'winner_variant':sm['variant'],
            'criterion':'MTL_valid_winner_variant_fixed_for_independent_STL','test_used':False}
    if Path(args.out).exists(): raise FileExistsError(args.out)
    write_json(args.out,result)


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('fit'); f.add_argument('--data',required=True); f.add_argument('--valid',required=True)
    f.add_argument('--reference-manifest',required=True); f.add_argument('--out',required=True)
    f.add_argument('--route',choices=['W00','M00'],required=True); f.add_argument('--boundary',required=True)
    f.add_argument('--variant',choices=['uncensored_point','censor_aware'],required=True)
    f.add_argument('--mode',choices=['mtl','stl'],default='mtl'); f.add_argument('--seed',type=int,default=42)
    f.add_argument('--epochs',type=int,default=30); f.add_argument('--device',default='cuda:0')
    l=sub.add_parser('lock-winner'); l.add_argument('--runs',nargs=2,required=True); l.add_argument('--out',required=True)
    c=sub.add_parser('lock-comparison'); c.add_argument('--run',required=True); c.add_argument('--winner-lock',required=True)
    c.add_argument('--winner-run'); c.add_argument('--out',required=True)
    for name in ('predict','export-embeddings'):
        e=sub.add_parser(name); e.add_argument('--data',required=True); e.add_argument('--part',choices=['train','valid','test'],required=True)
        e.add_argument('--run',required=True); e.add_argument('--winner-lock',required=True); e.add_argument('--out',required=True)
        e.add_argument('--device',default='cuda:0')
        e.add_argument('--report-registered-candidate',action='store_true')
    args=p.parse_args()
    {'fit':fit,'lock-winner':lock_winner,'lock-comparison':lock_comparison,'predict':predict,'export-embeddings':predict}[args.command](args)


if __name__=='__main__': main()

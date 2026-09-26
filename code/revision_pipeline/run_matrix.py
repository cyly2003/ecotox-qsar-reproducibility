"""Remote matrix controller with separate smoke, fitting, locking and reporting.

No shell interpolation. A failed cell stops advancement; no old run overwritten.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from revision_pipeline.train_contract import digest, read_json, write_json

BOUNDARIES = ('parent_disjoint', 'record_random', 'combination_holdout')
ROUTES = ('M00', 'W00')
VARIANTS = ('uncensored_point', 'censor_aware')


def run(command, logfile):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    started=time.time()
    print('START ' + logfile.stem, flush=True)
    with logfile.open('w', encoding='utf-8') as log:
        log.write(json.dumps({'argv':command,'started_unix':started},ensure_ascii=False)+'\n')
        log.flush()
        result=subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
    if result.returncode:
        print(logfile.read_text(encoding='utf-8',errors='replace')[-9000:],flush=True)
        raise RuntimeError(f'{logfile.stem} failed ({result.returncode})')
    print(f'DONE {logfile.stem} seconds={time.time()-started:.1f}',flush=True)


def training_command(data, reference, target, route, boundary, variant, epochs, mode='mtl'):
    return [sys.executable,'-m','revision_pipeline.train','fit',
            '--data',str(data/boundary/route/'train.parquet'),
            '--valid',str(data/boundary/route/'valid.parquet'),
            '--reference-manifest',str(reference/route/'manifest.json'),
            '--out',str(target),'--route',route,'--boundary',boundary,
            '--variant',variant,'--mode',mode,'--seed','42','--epochs',str(epochs)]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--data-version',default='revision_inputs_v2')
    parser.add_argument('--phase',choices=['smoke','full','followup'],required=True)
    parser.add_argument('--parallel',type=int,default=2)
    parser.add_argument('--tree-jobs',type=int,default=4)
    parser.add_argument('--run-tag',default='pilot01')
    args=parser.parse_args()
    if os.name=='nt': raise RuntimeError('All fits must run on remote Linux')
    if args.parallel not in (1,2): raise ValueError('Initial GPU concurrency limited to 1 or 2')
    root=args.root.resolve(strict=True)
    code=Path(__file__).resolve().parents[1]
    os.chdir(code)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(key,'4')
    os.environ['PYTHONIOENCODING']='utf-8'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    data_root=root/'data'/args.data_version
    data=data_root/'splits'
    audit=read_json(data_root/'split_audit.json')
    for entry in audit['physical_files']:
        if digest(data_root/entry['relative_path'])!=entry['sha256']:
            raise ValueError('Data physical file hash mismatch: '+entry['relative_path'])
    accepted=read_json(data_root/'v2_data_acceptance.json')
    if not accepted.get('accepted_for_registered_pilot',False):
        raise ValueError('Qualified data acceptance gate is not open')
    reference=root/'reference_models'
    if not args.run_tag.replace('_','').isalnum():raise ValueError('Invalid run tag')
    outputs=root/'runs'/args.data_version/args.run_tag
    logs=root/'logs'/args.data_version/args.run_tag
    outputs.mkdir(parents=True,exist_ok=True)
    logs.mkdir(parents=True,exist_ok=True)
    status_path=outputs/f'{args.phase}_controller_status.json'
    code_hash={str(p.relative_to(code)):digest(p) for p in sorted((code/'revision_pipeline').glob('*.py'))}
    state={'phase':args.phase,'status':'running','code_root':str(code),'started_unix':time.time(),
           'data_version':args.data_version,'code_hash':code_hash,'cells':[],'test_selection':False}
    write_json(status_path,state)

    def cell(route,boundary,variant,phase,epochs,mode='mtl'):
        target=outputs/phase/boundary/route/(mode+'_'+variant)
        logfile=logs/f'{phase}_{boundary}_{route}_{mode}_{variant}.log'
        if (target/'manifest.json').exists():
            m=read_json(target/'manifest.json')
            if m['seed']==42 and m['max_epochs']==epochs and m['variant']==variant and m['mode']==mode and m['test_loaded_rows']==0:
                return {'route':route,'boundary':boundary,'variant':variant,'mode':mode,'run':str(target),'status':'existing_complete'}
            raise ValueError('Incompatible existing manifest')
        if target.exists():
            raise ValueError('Incomplete prior run retained; choose a new run workspace rather than overwrite '+str(target))
        run(training_command(data,reference,target,route,boundary,variant,epochs,mode),logfile)
        return {'route':route,'boundary':boundary,'variant':variant,'mode':mode,'run':str(target),'status':'complete'}

    try:
        if args.phase=='smoke':
            # Both routes and both objectives on the strict data boundary.
            for route in ROUTES:
                for variant in VARIANTS:
                    state['cells'].append(cell(route,'parent_disjoint',variant,'smoke',1))
                    write_json(status_path,state)
            state['status']='smoke_complete'
        elif args.phase=='full':
            smoke=read_json(outputs/'smoke_controller_status.json')
            if smoke['status']!='smoke_complete' or smoke['code_hash']!=code_hash:
                raise ValueError('The same code package must pass all four smoke cells first')
            jobs=[(r,b,v) for b in BOUNDARIES for v in VARIANTS for r in ROUTES]
            with ThreadPoolExecutor(max_workers=args.parallel) as pool:
                futures=[pool.submit(cell,r,b,v,'full',30) for r,b,v in jobs]
                try:
                    for future in as_completed(futures):
                        state['cells'].append(future.result())
                        write_json(status_path,state)
                except Exception:
                    for future in futures:
                        future.cancel()
                    raise
            # Every data/model choice is made before held-out inference.
            for boundary in BOUNDARIES:
                for route in ROUTES:
                    group=outputs/'full'/boundary/route
                    lock=group/'winner_lock.json'
                    if not lock.exists():
                        run([sys.executable,'-m','revision_pipeline.train','lock-winner','--runs',
                             str(group/'mtl_uncensored_point'),str(group/'mtl_censor_aware'),'--out',str(lock)],
                            logs/f'lock_{boundary}_{route}.log')
            state['status']='full_candidates_and_locks_complete'
        else:
            full=read_json(outputs/'full_controller_status.json')
            if full['status']!='full_candidates_and_locks_complete' or full['code_hash']!=code_hash:
                raise ValueError('Complete unchanged Full matrix and validation locks required')
            for boundary in BOUNDARIES:
                ba=next(x for x in audit['boundaries'] if x['boundary_id']==boundary)
                for route in ROUTES:
                    group=outputs/'full'/boundary/route
                    lock_path=group/'winner_lock.json'; lock=read_json(lock_path)
                    winner=Path(lock['winner_run'])
                    exports=outputs/'exports'/boundary/route
                    exports.mkdir(parents=True,exist_ok=True)
                    for part in ('train','valid','test'):
                        file=exports/f'{part}_Z.parquet'
                        if not file.exists():
                            run([sys.executable,'-m','revision_pipeline.train','export-embeddings',
                                 '--data',str(data/boundary/route/f'{part}.parquet'),'--part',part,
                                 '--run',str(winner),'--winner-lock',str(lock_path),'--out',str(file)],
                                logs/f'Z_{boundary}_{route}_{part}.log')
                    # Report both registered point/censor candidates after the same winner lock.
                    for variant in VARIANTS:
                        file=exports/f'test_{variant}.parquet'
                        if not file.exists():
                            run([sys.executable,'-m','revision_pipeline.train','predict',
                                 '--data',str(data/boundary/route/'test.parquet'),'--part','test',
                                 '--run',str(group/('mtl_'+variant)),'--winner-lock',str(lock_path),
                                 '--report-registered-candidate','--out',str(file)],
                                logs/f'test_{boundary}_{route}_{variant}.log')
                    stl=cell(route,boundary,lock['winner_variant'],'stl',30,'stl')
                    state['cells'].append(stl); write_json(status_path,state)
                    stl_run=Path(stl['run']); stl_lock=stl_run.parent/'comparison_lock.json'
                    if not stl_lock.exists():
                        run([sys.executable,'-m','revision_pipeline.train','lock-comparison','--run',str(stl_run),
                             '--winner-lock',str(lock_path),'--out',str(stl_lock)],logs/f'stl_lock_{boundary}_{route}.log')
                    file=exports/'test_stl.parquet'
                    if not file.exists():
                        run([sys.executable,'-m','revision_pipeline.train','predict',
                             '--data',str(data/boundary/route/'test.parquet'),'--part','test',
                             '--run',str(stl_run),'--winner-lock',str(stl_lock),'--out',str(file)],
                            logs/f'test_stl_{boundary}_{route}.log')
                    heads=[s['model_head'] for s in ba['support'] if s['route']==route and s['common_exact_support_eligible']]
                    cfg={'route':route,'boundary_id':boundary,'seed':42,'tree_training_contract':'uncensored_point',
                         'target_column':'target','encoder_lock':str(lock_path),'split_audit':str(data_root/'split_audit.json'),
                         'eligible_heads':heads,'minimum_exact_counts':{'train':35,'valid':5,'test':10},
                         'parts':{part:{'data':str(data/boundary/route/f'{part}.parquet'),'z':str(exports/f'{part}_Z.parquet')} for part in ('train','valid','test')}}
                    config=exports/'trees_config.json'
                    if not config.exists():write_json(config,cfg)
                    trees=outputs/'trees'/boundary/route
                    for phase,finished in [('selection','selection_lock.json'),('inference','inference_manifest.json')]:
                        if not (trees/finished).exists():
                            run([sys.executable,'-m','revision_pipeline.trees','--config',str(config),'--output',str(trees),
                                 '--phase',phase,'--n-jobs',str(args.tree_jobs),'--execute-remote'],
                                logs/f'trees_{phase}_{boundary}_{route}.log')
                    state['cells'].append({'route':route,'boundary':boundary,'stage':'trees_complete','common_heads':len(heads)})
                    write_json(status_path,state)
            state['status']='followup_complete'
    except Exception as exc:
        state['status']='failed';state['error']=str(exc);write_json(status_path,state)
        raise
    state['finished_unix']=time.time();write_json(status_path,state)
    print(json.dumps({'status':state['status'],'cells':len(state['cells'])}),flush=True)


if __name__=='__main__':main()

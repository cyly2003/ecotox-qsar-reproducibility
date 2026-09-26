"""Triggered S_SOURCE: original test experiment closure plus S1, no reference closure."""
import json,sqlite3
import numpy as np
import pandas as pd
from audit_sources import ROOT,REV,OUT,sha,stable
from blind_builder import build,FIELDS

def main():
    meta=pd.concat([pd.read_parquet(OUT/f'D_SUBMITTED_{r}_prepared_metadata.parquet') for r in ['W00','M00']],ignore_index=True)
    meta=meta[meta.canonical_parent.ne('')]
    a=build(meta[FIELDS+['test_ids']],'S_SOURCE')
    a['condition_key_complete']=~(a.condition_time_key.str.contains('UNKNOWN')|a.condition_effect_key.str.contains('UNKNOWN'))
    a.to_parquet(OUT/'S_SOURCE_assignments.parquet',index=False)
    report=[]; support=[]
    for route,d in a.groupby('route'):
        closure=d[['test_ids','assigned_split']].copy();closure.test_ids=closure.test_ids.map(json.loads);closure=closure.explode('test_ids')
        assert closure.groupby('test_ids').assigned_split.nunique().max()==1
        n=d.groupby('split_group_id').size()
        head=d.groupby(['model_head','assigned_split']).size().unstack(fill_value=0).reindex(columns=['train','valid','test'],fill_value=0)
        absent=head[head.train.eq(0)]
        for name,row in head.iterrows(): support.append({'route':route,'model_head':name,**row.to_dict(),'prediction_status':'NOT_ESTIMABLE_NO_TRAINING_LABEL' if row.train==0 else ('NO_VALIDATION_LABEL' if row.valid==0 else 'TRAIN_VALID_AVAILABLE')})
        report.append({'route':route,'rows':len(d),'groups':len(n),'largest_group_n':int(n.max()),'largest_group_fraction':float(n.max()/len(d)),'split_counts':d.assigned_split.value_counts().to_dict(),'no_train_heads':absent.index.tolist(),'no_train_head_test_rows':int(absent.test.sum()),'test_group_overlap':0})
    db=REV/'01_冻结来源/data/submitted_v1_2_57.sqlite';c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
    payload=pd.read_sql_query('select * from aggregated_task_records_ptox_soil_mass_molar_qc_no_metal_inorganic',c)
    payload['stable_record_id']=[stable(row) for row in payload.to_dict('records')]
    payload=payload[payload.stable_record_id.isin(meta.stable_record_id)]
    manifest=[]
    for (route,part),d in a.groupby(['route','assigned_split']):
        overlap=(set(payload.columns)&set(d.columns))-{'stable_record_id'}
        d=d.merge(payload.drop(columns=list(overlap)),on='stable_record_id',validate='one_to_one')
        d['target']=d.target_value_median;d['is_censored']=False;d['bound_lower']=np.nan;d['bound_upper']=np.nan
        d['data_version']='D_SUBMITTED_PARENT_RESOLVED';d['boundary_id']='S_SOURCE';d['observation_kind']='legacy_point'
        d['target_scale']='neg_log10_mol_l' if route=='W00' else 'neg_log10_mol_kg'
        dest=OUT/'physical_splits/S_SOURCE'/route/f'{part}.parquet';dest.parent.mkdir(parents=True,exist_ok=True);d.to_parquet(dest,index=False)
        manifest.append({'route':route,'part':part,'n':len(d),'path':str(dest.relative_to(ROOT)),'sha256':sha(dest)})
    pd.DataFrame(support).to_csv(OUT/'S_SOURCE_head_support.csv',index=False)
    r={'status':'GENERATED_PENDING_INDEPENDENT_ACCEPTANCE','definition':'reliable original test_id plus S1 complete-key equivalence and same result derivation; NO reference union','split_seed':20260914,'assignment_sha256':sha(OUT/'S_SOURCE_assignments.parquet'),'routes':report,'physical_files':manifest,'test_labels_used_for_assignment':False,'authorized_models':['MTL_FULL','STL_FULL'],'training_seeds':[42,2042,3407,8417]}
    (OUT/'S_SOURCE_prelock_report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

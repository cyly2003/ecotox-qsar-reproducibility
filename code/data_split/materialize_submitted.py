"""Authorized D_SUBMITTED resolved-parent splits; targets only copied AFTER locking metadata assignments."""
import json,sqlite3
import numpy as np
import pandas as pd
from audit_sources import ROOT,REV,OUT,sha,stable
from blind_builder import build,FIELDS,digest

def main():
    allmeta=pd.concat([pd.read_parquet(OUT/f'D_SUBMITTED_{r}_prepared_metadata.parquet') for r in ['W00','M00']],ignore_index=True)
    rejected=allmeta[allmeta.canonical_parent.eq('')].copy()
    rejected['eligibility_status']='NOT_ELIGIBLE_PARENT_ID'
    rejected.to_parquet(OUT/'D_SUBMITTED_unresolved_parent_identity_metadata.parquet',index=False)
    meta=allmeta[allmeta.canonical_parent.ne('')].copy()
    parts=[]; reports=[]
    # All assignments materialized before numerical targets are opened.
    for sid in ['S1_CONDITION','S2_COMBINATION','S3_PARENT']:
        a=build(meta[FIELDS],sid)
        a['condition_key_complete']=~(a.condition_time_key.str.contains('UNKNOWN')|a.condition_effect_key.str.contains('UNKNOWN'))
        for route,d in a.groupby('route'):
            dest=OUT/'assignments'/route/f'{sid}.parquet'; dest.parent.mkdir(parents=True,exist_ok=True)
            d.to_parquet(dest,index=False)
            reports.append({'route':route,'split_id':sid,'rows':len(d),'groups':int(d.split_group_id.nunique()),'parts':d.assigned_split.value_counts().to_dict(),'complete_condition_rows':int(d.condition_key_complete.sum()),'unknown_condition_rows':int((~d.condition_key_complete).sum()),'sha256':sha(dest),'path':str(dest.relative_to(ROOT))})
        parts.append(a)
    assignments=pd.concat(parts,ignore_index=True)
    assignments.to_parquet(OUT/'split_assignments.parquet',index=False)
    split_hash=sha(OUT/'split_assignments.parquet')
    # Copy complete original columns without inspecting/optimizing target values.
    db=REV/'01_冻结来源/data/submitted_v1_2_57.sqlite'
    c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
    payload=pd.read_sql_query('select * from aggregated_task_records_ptox_soil_mass_molar_qc_no_metal_inorganic',c)
    payload['stable_record_id']=[stable(row) for row in payload.to_dict('records')]
    payload=payload[payload.stable_record_id.isin(meta.stable_record_id)]
    if payload.stable_record_id.duplicated().any(): raise ValueError('Nonunique payload identity')
    manifest=[]
    for (route,sid,part),a in assignments.groupby(['route','split_id','assigned_split']):
        overlap=(set(payload.columns)&set(a.columns))-{'stable_record_id'}
        d=a.merge(payload.drop(columns=list(overlap)),on='stable_record_id',validate='one_to_one')
        assert len(d)==len(a)
        d['target']=d.target_value_median
        d['is_censored']=False; d['bound_lower']=np.nan; d['bound_upper']=np.nan
        d['observation_kind']='legacy_point'; d['data_version']='D_SUBMITTED_PARENT_RESOLVED'
        d['target_scale']='neg_log10_mol_l' if route=='W00' else 'neg_log10_mol_kg'
        d['boundary_id']=sid
        dest=OUT/'physical_splits'/sid/route/f'{part}.parquet';dest.parent.mkdir(parents=True,exist_ok=True)
        d.to_parquet(dest,index=False)
        manifest.append({'route':route,'split_id':sid,'part':part,'n':len(d),'path':str(dest.relative_to(ROOT)),'sha256':sha(dest),'columns':list(d.columns)})
    assert sha(OUT/'split_assignments.parquet')==split_hash
    result={'status':'GENERATED_PENDING_INDEPENDENT_LOCK_ACCEPTANCE','split_seed':20260914,'training_seeds':[42,2042,3407,8417],'fraction_targets':[.64,.16,.20],'source':'D_SUBMITTED resolved-parent subset; metadata-only eligibility, no target quality filtering added','assignment_sha256':split_hash,'source_db_sha256':sha(db),'responses_used_in_assignment':False,'test_target_values_inspected':False,'routes_independent':True,'source_closure':'within route source_result_id plus requested boundary key','unknown_condition_policy':'conservative shared UNKNOWN token, not claimed equal measured time','NOEC_LOEC':'NA_NOT_APPLICABLE retained separately','assignments':reports,'physical_files':manifest}
    (OUT/'split_prelock_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':result['status'],'assignment_sha256':split_hash,'assignments':reports,'physical_files':len(manifest)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()

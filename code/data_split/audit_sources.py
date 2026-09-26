"""Read-only source audit. Never reads numerical toxicity targets/predictions."""
import hashlib,json,sqlite3
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
REV=ROOT.parent
OUT=ROOT/'01_数据版本与划分核查'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()
def stable(row):
    x=[str(row[c]) for c in ('aggregate_id','medium_domain','target_name','target_family')]
    return 'stage_sample_v1:'+hashlib.sha256(json.dumps(x,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    db=REV/'01_冻结来源/data/submitted_v1_2_57.sqlite'
    parquet=REV/'02_清洗重建/revision_inputs_v2/all_observations.parquet'
    c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
    cols='aggregate_id,medium_domain,target_name,target_family,cas_number,smiles,species_number,latin_name,task_head,effect_level_x,duration_bin_h,duration_bin_rule,result_ids,test_ids,reference_numbers,value_quality'
    p=pd.read_sql_query('select '+cols+' from aggregated_task_records_ptox_soil_mass_molar_qc_no_metal_inorganic',c)
    p['stable_record_id']=[stable(row) for row in p.to_dict('records')]
    splits=pd.read_sql_query('select split_name,record_id,split_part,seed,group_key from split_assignments',c)
    source=pd.read_sql_query('select result_id,test_id,reference_number,exposure_duration_mean_h,obs_duration_mean_h,conc1_unit from target_records',c)
    # Source IDs are metadata; all repeated IDs must have consistent timing.
    counts=source.groupby('result_id').exposure_duration_mean_h.nunique(dropna=False)
    source=source.drop_duplicates('result_id').set_index('result_id')
    records=[]; outputs=[]
    for route in ('W00','M00'):
        manifest=REV/f'01_冻结来源/reference_models/{route}/manifest.json'
        m=json.loads(manifest.read_text(encoding='utf-8'))
        s=splits[splits.split_name==m['split_name']].copy()
        q=s.merge(p,left_on='record_id',right_on='stable_record_id',how='left',validate='one_to_one')
        assert q.aggregate_id.notna().all()
        q['route']=route; q['data_version']='D_SUBMITTED'; q['split_id']='S0_ORIGINAL'
        q.to_parquet(OUT/f'D_SUBMITTED_{route}_S0_identity_metadata.parquet',index=False)
        records.append({'route':route,'rows':len(q),'heads':int(q.task_head.nunique()),'species':int(q.latin_name.nunique()),'cas':int(q.cas_number.nunique()),'split_counts':q.split_part.value_counts().to_dict(),'manifest_seed':m['seed'],'validation_source':m.get('validation_source'),'validation_rows':m.get('validation_rows'),'validation_seed':m.get('validation_seed'),'identity_sha256':sha(OUT/f'D_SUBMITTED_{route}_S0_identity_metadata.parquet'),'reference_manifest':str(manifest),'reference_manifest_sha256':sha(manifest)})
        outputs.append(q)
    vcols=['route','stable_record_id','canonical_parent','latin_name','model_head','source_result_ids_json','effect_level_x','duration_bin_h','exposure_duration_mean_h','is_censored','value_quality']
    v=pd.read_parquet(parquet,columns=vcols)
    time_sets=[]; missing_sources=0; multitime=0
    for raw in v.source_result_ids_json:
        ids=[int(x) for x in json.loads(raw)]
        vals=[]
        for x in ids:
            if x not in source.index: missing_sources+=1; vals.append('UNKNOWN_SOURCE')
            else:
                t=source.at[x,'exposure_duration_mean_h']
                vals.append('UNKNOWN_TIME' if pd.isna(t) else float(t).hex())
        vals=sorted(set(vals)); multitime+=len(vals)>1
        time_sets.append(json.dumps(vals,separators=(',',':')))
    v['condition_time_key']=time_sets
    v['condition_effect_key']=[('NA_NOT_APPLICABLE' if h.startswith(('NOEC','LOEC')) else 'UNKNOWN_EFFECT') if pd.isna(x) else float(x).hex() for h,x in zip(v.model_head,v.effect_level_x)]
    v.to_parquet(OUT/'revision_v2_identity_metadata_time_recovered.parquet',index=False)
    contract={'schema':'unified_revision_data_discovery_v1','status':'PREPARATION_ONLY_NOT_COMMON_SPLIT_LOCK','D_SUBMITTED':records,
      'D_REVISION_V2':{'source':str(parquet),'sha256':sha(parquet),'rows':len(v),'counts':v.groupby(['route','is_censored']).size().reset_index(name='n').to_dict('records'),'point_exposure_field_missing':int(v.loc[~v.is_censored,'exposure_duration_mean_h'].isna().sum()),'recovered_multi_time_rows':multitime,'missing_source_ids':missing_sources,'source_id_conflicting_time_count':int((counts>1).sum())},
      'source_database':{'path':str(db),'sha256':sha(db)},
      'target':{'W00':'-log10(mol/L)','M00':'-log10((mg/kg)/(1000*MW[g/mol]))','column':'target_value_median (reference manifests); revision target is renamed column'},
      'identity':{'stable_record_id':'SHA256 JSON [aggregate_id,medium_domain,target_name,target_family] prefixed stage_sample_v1:','parent':'legacy normalize_structure: largest fragment, uncharge, non-isomeric canonical SMILES; verify exact source implementation','species':'latin_name; species_number retained for audit','head':'task_head in source; model_head in revision','sources':'result_ids -> target_records.result_id -> test_id/reference_number'},
      'S1':{'time':'raw exposure_duration_mean_h recovered through source result IDs; hexadecimal float preserves exact stored values; observation time NOT substituted','multiple_time_policy':'requires decision: union closure across each represented complete condition, rather than treating a list as a single condition','missing_policy':'explicit UNKNOWN_TIME/UNKNOWN_EFFECT; conservatively group, report separately','effect_not_applicable':'NOEC/LOEC missing effect -> NA_NOT_APPLICABLE'},
      'limitations':['D_SUBMITTED old full-response QC remains; new splits do not undo selection','revision_inputs_v2 point includes approx and midpoint, not D_EXACT_QUALITY','D_EXACT_QUALITY recovery not implemented in this audit','S0 W00 internal validation is manifest-seed-dependent; database train is development pool, do not invent valid IDs','no test numerical targets or predictions read by audit','candidate metadata only, public assignments require main-agent integration'],
      'runtime':{'model_requested':'gpt-6-astra','model_effective':'unknown','effort_requested':'medium','effort_effective':'unknown','client_version':'unknown','evidence':'task requested configuration; no runtime telemetry exposed to this child'}}
    (OUT/'data_contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'submitted':records,'revision':contract['D_REVISION_V2']},ensure_ascii=False,indent=2))
if __name__=='__main__': main()

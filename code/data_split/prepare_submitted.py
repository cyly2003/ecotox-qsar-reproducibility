"""Prepare label-free D_SUBMITTED identity fields; no split locking."""
import importlib.util,json,sqlite3,sys
from pathlib import Path
import pandas as pd
from rdkit import RDLogger
from audit_sources import ROOT,REV,OUT,sha

def main():
    code=REV/'01_冻结来源/code_current'
    sys.path.insert(0,str(code))
    module=code/'scripts/build_scaffold_cluster_splits.py'
    spec=importlib.util.spec_from_file_location('original_parent_builder',module)
    legacy=importlib.util.module_from_spec(spec); sys.modules[spec.name]=legacy; spec.loader.exec_module(legacy)
    RDLogger.DisableLog('rdApp.*')
    c=sqlite3.connect((REV/'01_冻结来源/data/submitted_v1_2_57.sqlite').resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
    src=pd.read_sql_query('select result_id,exposure_duration_mean_h from target_records',c).drop_duplicates('result_id').set_index('result_id')
    audit=[]
    for route in ['W00','M00']:
        d=pd.read_parquet(OUT/f'D_SUBMITTED_{route}_S0_identity_metadata.parquet')
        lookup={s:legacy.normalize_structure(s) for s in d.smiles.fillna('').unique()}
        d['canonical_parent']=[lookup[s]['canonical_smiles'] for s in d.smiles.fillna('')]
        d['structure_status']=[lookup[s]['structure_status'] for s in d.smiles.fillna('')]
        d['source_result_ids_json']=[json.dumps(sorted(set(str(x) for x in json.loads(v))),separators=(',',':')) for v in d.result_ids]
        times=[]; ns=[]
        for raw in d.result_ids:
            vals=[]
            for rid in json.loads(raw):
                t=src.at[int(rid),'exposure_duration_mean_h']
                vals.append('UNKNOWN_TIME' if pd.isna(t) else float(t).hex())
            vals=sorted(set(vals)); ns.append(len(vals)); times.append(vals[0] if len(vals)==1 else json.dumps(vals,separators=(',',':')))
        d['condition_time_key']=times
        d['condition_effect_key']=[('NA_NOT_APPLICABLE' if h.startswith(('NOEC','LOEC')) else 'UNKNOWN_EFFECT') if pd.isna(x) else float(x).hex() for h,x in zip(d.task_head,d.effect_level_x)]
        d['model_head']=d.task_head
        dest=OUT/f'D_SUBMITTED_{route}_prepared_metadata.parquet'; d.to_parquet(dest,index=False)
        audit.append({'route':route,'rows':len(d),'valid_parent_n':int(d.canonical_parent.ne('').sum()),'parent_n':int(d.loc[d.canonical_parent.ne(''),'canonical_parent'].nunique()),'unresolved_parent_n':int(d.canonical_parent.eq('').sum()),'multiple_source_times_n':sum(n>1 for n in ns),'unknown_time_n':int(d.condition_time_key.eq('UNKNOWN_TIME').sum()),'unknown_effect_n':int(d.condition_effect_key.eq('UNKNOWN_EFFECT').sum()),'effect_not_applicable_n':int(d.condition_effect_key.eq('NA_NOT_APPLICABLE').sum()),'metadata_sha256':sha(dest)})
    report={'status':'METADATA_READY_COMMON_SPLIT_NOT_LOCKED','routes':audit,'normalization_source':str(module),'normalization_source_sha256':sha(module),'unresolved_parent_policy':'cannot call CAS surrogate a canonical parent; block affected chemical rows or use explicitly separate unresolved-stratum policy approved by main agent','runtime_evidence':'Main agent reports verified session metadata in 00_总控与审稿意见索引/agent_runtime.json; child did not independently verify telemetry.'}
    (OUT/'submitted_metadata_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

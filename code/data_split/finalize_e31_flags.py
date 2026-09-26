"""Correct ancillary conflict flag against actual frozen manifest; no target or partition change."""
import json,sqlite3
import pandas as pd
from audit_sources import ROOT,REV,OUT,sha

def main():
    c=sqlite3.connect((REV/'01_冻结来源/data/modeling_parent_historical.sqlite').resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
    source=dict(c.execute('select key,value from qc_task_build_manifest').fetchall())
    threshold=float(source['conflict_threshold_log_unit'])
    path=OUT/'E31/E31_data_contract.json';contract=json.loads(path.read_text(encoding='utf-8'));changed=[]
    for f in contract['files']:
        p=ROOT/f['path'];d=pd.read_parquet(p)
        desired=(d.source_reference_count.gt(1)&d.cross_reference_range.gt(threshold)).astype(int)
        changed.append({'path':f['path'],'flags_changed':int((desired!=d.cross_reference_conflict_flag).sum())})
        d['cross_reference_conflict_flag']=desired;d.to_parquet(p,index=False);f['sha256']=sha(p)
    contract['conflict_threshold_log_unit']=threshold
    contract['qc_manifest_actual_parameters']={k:source[k] for k in ['min_outlier_group_n','robust_z_threshold','conflict_threshold_log_unit']}
    contract['ancillary_flag_correction']={'reason':'initial implementation assumed 1.0; frozen qc_task_build_manifest proves 0.3; corrected before independent lock','target_and_partition_changed':False,'files':changed}
    path.write_text(json.dumps(contract,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'E31/REBUILD_STATUS.json').write_text(json.dumps({'status':'REBUILD_COMPLETE_PENDING_INDEPENDENT_ACCEPTANCE','medium_expansion_restored':True,'conflict_threshold_from_frozen_manifest':threshold,'contract_sha256':sha(path)}),encoding='utf-8')
    print(json.dumps(changed,ensure_ascii=False))
if __name__=='__main__':main()

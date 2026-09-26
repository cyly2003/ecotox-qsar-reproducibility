"""E30 strict mean-point / paired mean-censor data, immutable existing S3 parent map."""
import hashlib,importlib.util,json,math,sqlite3,sys
from collections import Counter
import numpy as np
import pandas as pd
from rdkit import RDLogger
from audit_sources import ROOT,REV,OUT,sha
from audit_exact_qualifiers import operator

def main():
    code=REV/'04_执行代码';sys.path.insert(0,str(code))
    spec=importlib.util.spec_from_file_location('existing_revision_data',code/'revision_pipeline/build_data.py')
    legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)
    RDLogger.DisableLog('rdApp.*')
    destroot=OUT/'E30';destroot.mkdir(parents=True,exist_ok=True)
    db=REV/'01_冻结来源/data/submitted_v1_2_57.sqlite';c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
    source=pd.read_sql_query('select result_id,conc1_mean_op,conc1_unit,molecular_weight_g_mol,unit_family_v2,standard_unit_v2,standard_value_mg_l,standard_value_mol_l,standard_value_mg_kg from target_records',c).drop_duplicates('result_id')
    clean=sqlite3.connect((REV/'01_冻结来源/data/ecotox_clean.sqlite').resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
    raw=pd.read_sql_query('select result_id,conc1_mean_op from results',clean).drop_duplicates('result_id')
    check=source[['result_id','conc1_mean_op']].merge(raw,on='result_id',how='left',suffixes=('_source','_clean'),indicator=True,validate='one_to_one')
    mismatch=(check.conc1_mean_op_source.map(operator)!=check.conc1_mean_op_clean.map(operator))|check['_merge'].ne('both')
    if mismatch.any():raise ValueError('Raw-clean qualifier provenance mismatch')
    source=source.set_index('result_id')
    gate=pd.read_parquet(OUT/'strict_exact_source_admission.parquet').set_index(['route','stable_record_id'])
    assignments=pd.read_parquet(OUT/'split_assignments.parquet');assignments=assignments[assignments.split_id=='S3_PARENT']
    parent_map={(r,p):part for r,p,part in assignments[['route','canonical_parent','assigned_split']].drop_duplicates().itertuples(index=False,name=None)}
    assert len(parent_map)==len(assignments[['route','canonical_parent','assigned_split']].drop_duplicates())
    original_parent_map=dict(parent_map)
    frames=[];point_logs=[];struct={}
    for route in ['W00','M00']:
        for part in ['train','valid','test']:
            d=pd.read_parquet(OUT/f'physical_splits/S3_PARENT/{route}/{part}.parquet')
            keep=[]
            for row in d.to_dict('records'):
                reason='admitted'
                if not gate.at[(route,row['stable_record_id']),'all_sources_strict_mean_eligible']: reason='source_strict_mean_gate_failed'
                s=row['smiles'] or ''
                if s not in struct:struct[s]=legacy.structure(s)
                if reason=='admitted' and struct[s]['structure_admission']!='retain':reason=struct[s]['structure_reason']
                if reason=='admitted':
                    for rid in json.loads(row['source_result_ids_json']):
                        src=source.loc[int(rid)]
                        if legacy.normalize_unit_text(src.conc1_unit) in {'ul/l','nl/l','ai ul/l','ai nl/l'}:reason='volume_missing_density';break
                        fam=src.unit_family_v2
                        legal={'water_mg_l','water_mol_l'} if route=='W00' else {'soil_mg_kg'}
                        if fam not in legal:reason='route_unit_not_legal';break
                        if fam!='water_mol_l' and (pd.isna(src.molecular_weight_g_mol) or not math.isfinite(float(src.molecular_weight_g_mol)) or src.molecular_weight_g_mol<=0):reason='mass_conversion_invalid_mw';break
                keep.append(reason=='admitted')
                point_logs.append({'route':route,'stable_record_id':row['stable_record_id'],'reason':reason,'assigned_split':part})
            frames.append(d.loc[keep].copy())
    exact=pd.concat(frames,ignore_index=True)
    # Every actual paired mean inequality is considered, including Unicode; no old excluded_reason-only scan.
    candidates=pd.read_sql_query("select * from target_records where trim(conc1_mean_op) in ('<','>','<=','>=','≤','≥')",c)
    legal_heads=set(zip(exact.route,exact.model_head));censors=[];logs=[]
    for row in candidates.to_dict('records'):
        s=row.get('smiles') or ''
        if s not in struct:struct[s]=legacy.structure(s)
        st=struct[s];record=None;reason=st['structure_reason']
        if st['structure_admission']=='retain':
            row.update(st);row['conc1_mean_op']=operator(row['conc1_mean_op']);row['excluded_reason']='censored_toxicity_value'
            record,reason=legacy.build_mean_censor(row)
            if record is not None and (record['route'],record['model_head']) not in legal_heads: record=None;reason='no_exact_route_head'
        logs.append({'result_id':row['result_id'],'reason':reason,'raw_mean_op':row['conc1_mean_op']})
        if record is not None:
            key=(record['route'],record['canonical_parent'])
            if key not in parent_map:
                u=int(hashlib.sha256(json.dumps(['E30_NEW_PARENT',20260914,*key],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()[:16],16)/2**64
                parent_map[key]='train' if u<.64 else ('valid' if u<.8 else 'test')
            record['assigned_split']=parent_map[key];record['split_id']='S3_PARENT';record['boundary_id']='S3_PARENT'
            record['split_group_id']='parent:'+hashlib.sha256(json.dumps(key,ensure_ascii=False).encode()).hexdigest()
            censors.append(record)
    cens=pd.DataFrame(censors)
    assert all(parent_map[k]==v for k,v in original_parent_map.items())
    assert set(exact.stable_record_id).isdisjoint(cens.stable_record_id)
    combined=pd.concat([exact,cens],ignore_index=True)
    for col in combined.columns:
        if combined[col].dtype=='object' and len({type(v) for v in combined[col].dropna()})>1:
            combined[col]=combined[col].map(lambda v:None if v is None or pd.isna(v) else str(v))
    for col in ['target','bound_lower','bound_upper','duration_bin_h','effect_level_x']:
        combined[col]=pd.to_numeric(combined[col],errors='raise').astype(float)
    combined['is_censored']=combined.is_censored.astype(bool)
    combined['data_version']='E30_STRICT_MEAN_QUALITY'
    assert not combined.duplicated(['route','stable_record_id']).any()
    assert combined.groupby(['route','canonical_parent']).assigned_split.nunique().max()==1
    manifest=[]
    for (route,part),d in combined.groupby(['route','assigned_split']):
        # Save common exact view once, never separately resample C0/C1/C2.
        for name,q in [('exact',d[~d.is_censored]),('exact_plus_censored',d),('censored_only',d[d.is_censored])]:
            dest=destroot/name/route/f'{part}.parquet';dest.parent.mkdir(parents=True,exist_ok=True);q.to_parquet(dest,index=False)
            manifest.append({'view':name,'route':route,'part':part,'n':len(q),'path':str(dest.relative_to(ROOT)),'sha256':sha(dest)})
    pd.DataFrame(point_logs).to_parquet(destroot/'point_admission.parquet',index=False)
    pd.DataFrame(logs).to_parquet(destroot/'censor_admission.parquet',index=False)
    pd.DataFrame([{'route':r,'canonical_parent':p,'assigned_split':v,'in_original_S3':(r,p) in original_parent_map} for (r,p),v in sorted(parent_map.items())]).to_parquet(destroot/'parent_assignment_ledger.parquet',index=False)
    r={'status':'MATERIALIZED_PENDING_INDEPENDENT_ACCEPTANCE','raw_qualifier_clean_match_rows':len(check),'raw_qualifier_mismatches':int(mismatch.sum()),'source_sha256':sha(db),'base_assignment_sha256':sha(OUT/'split_assignments.parquet'),'parent_ledger_sha256':sha(destroot/'parent_assignment_ledger.parquet'),'old_parent_partitions_changed':0,'new_parent_count':len(parent_map)-len(original_parent_map),'point_reason_counts':dict(Counter(x['reason'] for x in point_logs)),'censor_reason_counts':dict(Counter(x['reason'] for x in logs)),'files':manifest,'training_contract':{'C0':'exact Huber new run; cannot reuse broader non-exact main model','C1':'exact Gaussian NLL','C2':'exact plus legal paired mean-censor Gaussian NLL','preprocessing_and_scaler':'fit common exact train only','validation_and_point_test':'common exact view identical IDs and values','censored_validation_test':'separate likelihood or violation diagnostics; no point R2 against bound'},'limitations':['exact points retain submitted global response QC','strict structural/volume/MW quality selection differs from main D_SUBMITTED','range-only censors remain pending; no invented interval semantics','old aggregate target preserved only if all source rows pass strict source gate','E31 QC-before-recovery not yet completed']}
    (destroot/'E30_data_contract.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':r['status'],'new_parents':r['new_parent_count'],'point_reasons':r['point_reason_counts'],'censor_reasons':r['censor_reason_counts'],'files':[{k:v for k,v in f.items() if k in ['view','route','part','n']} for f in manifest]},ensure_ascii=False,indent=2))
if __name__=='__main__':main()

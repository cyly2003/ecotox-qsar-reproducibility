"""Recover quality-exact raw records, split before QC, preserve original aggregation and mass->molar order."""
import hashlib,importlib.util,json,math,sqlite3,sys
from collections import Counter,defaultdict
from statistics import median,stdev
import numpy as np
import pandas as pd
from rdkit import RDLogger
from audit_sources import ROOT,REV,OUT,sha
from audit_exact_qualifiers import classify

def main():
    code=REV/'04_执行代码';sys.path.insert(0,str(code))
    from qsar_tl.data.qc_aggregation import QcRow,assign_reference_weights,aggregate_qc_rows,robust_z_scores
    spec=importlib.util.spec_from_file_location('e31_revision_data',code/'revision_pipeline/build_data.py');legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)
    RDLogger.DisableLog('rdApp.*')
    destroot=OUT/'E31';destroot.mkdir(parents=True,exist_ok=True)
    db=REV/'01_冻结来源/data/modeling_parent_historical.sqlite';c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro&immutable=1',uri=True);c.row_factory=sqlite3.Row
    original_qc_manifest=dict(c.execute('SELECT key,value FROM qc_task_build_manifest').fetchall())
    conflict_threshold=float(original_qc_manifest['conflict_threshold_log_unit'])
    ledger=pd.read_parquet(OUT/'E30/parent_assignment_ledger.parquet');parent_map={(r,p):part for r,p,part in ledger[['route','canonical_parent','assigned_split']].itertuples(index=False,name=None)};old_map=dict(parent_map)
    submitted={r:pd.read_parquet(OUT/f'D_SUBMITTED_{r}_prepared_metadata.parquet') for r in ['W00','M00']}
    heads={r:set(d.model_head) for r,d in submitted.items()}
    old_sources={r:set(str(x) for raw in d.source_result_ids_json for x in json.loads(raw)) for r,d in submitted.items()}
    structure={};pools=defaultdict(list);admission=Counter();records=[];payload_by_id={};raw_n=0
    sql="SELECT * FROM task_records WHERE task_status='included' AND target_value IS NOT NULL AND ((target_name='ptox_mol_l' AND EXISTS (SELECT 1 FROM json_each(task_records.medium_domains) WHERE value='aquatic')) OR (target_name='neg_log10_mg_kg' AND EXISTS (SELECT 1 FROM json_each(task_records.medium_domains) WHERE value='soil'))) ORDER BY result_id"
    for item in c.execute(sql):
        row=dict(item);raw_n+=1;route='W00' if row['target_name']=='ptox_mol_l' else 'M00'
        reason=classify(row)
        if reason!='strict_source_mean_eligible':admission[(route,reason)]+=1;continue
        if row['task_head'] not in heads[route]:admission[(route,'outside_frozen_submitted_head_set')]+=1;continue
        smi=row.get('smiles') or ''
        if smi not in structure:structure[smi]=legacy.structure(smi)
        st=structure[smi]
        if st['structure_admission']!='retain':admission[(route,st['structure_reason'])]+=1;continue
        if legacy.normalize_unit_text(row['conc1_unit']) in {'ul/l','nl/l','ai ul/l','ai nl/l'}:admission[(route,'volume_missing_density')]+=1;continue
        legal={'water_mg_l','water_mol_l'} if route=='W00' else {'soil_mg_kg'}
        if row['unit_family_v2'] not in legal:admission[(route,'route_unit_not_legal')]+=1;continue
        mw=row.get('molecular_weight_g_mol')
        if row['unit_family_v2']!='water_mol_l' and (mw is None or not math.isfinite(mw) or mw<=0):admission[(route,'invalid_mass_conversion_mw')]+=1;continue
        key=(route,st['canonical_parent'])
        if key not in parent_map:
            u=int(hashlib.sha256(json.dumps(['E30_NEW_PARENT',20260914,*key],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()[:16],16)/2**64
            parent_map[key]='train' if u<.64 else ('valid' if u<.8 else 'test')
        part=parent_map[key]
        identity=(route,str(row['result_id']))
        if identity in payload_by_id:raise ValueError('Duplicate route/source record; resolve before aggregation')
        payload_by_id[identity]=row
        q=QcRow(payload=row,target_value=float(row['target_value']),medium_domain=row['medium_domain'],qc_group=(row['target_name'],row['medium_domain'],row['task_head']))
        pools[(route,part)].append(q);admission[(route,'admitted')]+=1
        records.append({'route':route,'result_id':str(row['result_id']),'test_id':str(row['test_id']),'reference_number':str(row['reference_number']),'canonical_parent':st['canonical_parent'],'assigned_split':part,'model_head':row['task_head'],'original_target_name':row['target_name'],'in_submitted_source_set':str(row['result_id']) in old_sources[route]})
    print('Quality source records prepared: '+json.dumps(dict(Counter(k[0] for k in payload_by_id))),flush=True)
    pd.DataFrame(records).to_parquet(destroot/'quality_source_identity_ledger.parquet',index=False)
    pd.DataFrame([{'route':r,'canonical_parent':p,'assigned_split':v,'in_inherited_parent_map':(r,p) in old_map} for (r,p),v in sorted(parent_map.items())]).to_parquet(destroot/'parent_assignment_ledger.parquet',index=False)
    assert all(parent_map[k]==v for k,v in old_map.items())
    manifest=[];qc_log=[];qc_stats=[];recovery=[]
    def materialize(rows,route,part,arm):
        # Exact original reference/test median -> recency-weighted mean implementation.
        # Legacy load_and_qc_rows computes recency weights on all raw rows,
        # including rows subsequently excluded by QC. A initializes weights;
        # B preserves these same train-only weights instead of refitting them.
        if arm!='B_TRAIN_ONLY_QC':assign_reference_weights(rows)
        aggregates=aggregate_qc_rows(rows,conflict_threshold_log_unit=conflict_threshold)
        result=[]
        for a in aggregates:
            ids=[str(x) for x in json.loads(a['result_ids'])]
            mw_values=[payload_by_id[(route,x)]['molecular_weight_g_mol'] for x in ids]
            first=payload_by_id[(route,ids[0])];s=structure[first.get('smiles') or '']
            a['canonical_parent']=s['canonical_parent'];a['route']=route;a['assigned_split']=part;a['split_id']='S3_PARENT';a['boundary_id']='S3_PARENT'
            # Reproduce post-QC aggregated medium expansion. QC above uses the
            # original primary domain; the final modeling route is one member
            # of medium_domains, not necessarily that primary domain.
            domains=json.loads(a['medium_domains']);domain='aquatic' if route=='W00' else 'soil'
            if domain not in domains:raise ValueError('Requested route absent from declared medium domains')
            a['qc_primary_medium_domain']=a['medium_domain'];a['medium_domain']=domain
            a['medium_assignment_weight']=1.0/len(domains)
            a['source_result_ids_json']=json.dumps(sorted(ids),separators=(',',':'))
            a['model_head']=a['task_head'];a['original_target_name']=a['target_name'];a['original_target_value_weighted_mean']=a['target_value_median']
            a['parent_target_name']=a['target_name'];a['parent_target_family']=a['target_family'];a['parent_target_basis']=a['target_basis']
            if route=='M00':
                # Match original precision-level MW agreement rule; fail closed on conflicts.
                if len(set(f'{x:.8f}' for x in mw_values))!=1:raise ValueError('Conflicting source MW within aggregate')
                mw=max(mw_values);offset=3+math.log10(mw)
                for col in ['target_value_median','target_value_mean','target_value_weighted_mean','target_value_unweighted_median','target_value_min','target_value_max']:
                    if a.get(col) is not None:a[col]+=offset
                a.update(target_name='neg_log10_mol_kg',target_family='solid_neglog_mol_kg',target_basis='mol/kg_from_mg/kg:soil',unit_family_v2='soil_mol_kg',standard_unit_v2='mol/kg',molecular_weight_g_mol_used=mw,target_transform='mgkg_to_molkg_after_original_aggregation',standard_value_mol_kg=None)
            else:a['target_transform']='identity_ptox'
            # New identity keys source membership and original grouping; not reused submitted aggregate numbers.
            basis=[route,a['canonical_parent'],a['source_result_ids_json'],a['task_head'],a['duration_bin_h'],a['effect_level_x'],a['target_name']]
            a['stable_record_id']='e31_exact_v1:'+hashlib.sha256(json.dumps(basis,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
            a['aggregate_id']=a['stable_record_id'];a['split_group_id']='parent:'+hashlib.sha256(json.dumps([route,a['canonical_parent']],ensure_ascii=False).encode()).hexdigest()
            a['target']=a['target_value_median'];a['is_censored']=False;a['bound_lower']=np.nan;a['bound_upper']=np.nan
            a['target_scale']='neg_log10_mol_l' if route=='W00' else 'neg_log10_mol_kg';a['observation_kind']='strict_quality_exact';a['data_version']='D_EXACT_QUALITY_PRE_GLOBAL_QC'
            a['all_sources_in_submitted']=all(x in old_sources[route] for x in ids)
            # Expose source time only if one precise value; future S1 recovery must refuse multi-time groups.
            times=set(payload_by_id[(route,x)]['exposure_duration_mean_h'] for x in ids)
            a['source_exposure_time_count']=len(times)
            result.append(a)
        d=pd.DataFrame(result)
        if d.stable_record_id.duplicated().any():raise ValueError('New stable aggregate IDs collide')
        dest=destroot/arm/route/f'{part}.parquet';dest.parent.mkdir(parents=True,exist_ok=True);d.to_parquet(dest,index=False)
        manifest.append({'arm':arm,'route':route,'part':part,'source_n':len(rows),'aggregate_n':len(d),'outside_submitted_source_aggregate_n':int((~d.all_sources_in_submitted).sum()),'path':str(dest.relative_to(ROOT)),'sha256':sha(dest)})
    for (route,part),rows in sorted(pools.items()):
        if part!='train':
            materialize(rows,route,part,'common_eval');continue
        materialize(rows,route,part,'A_KEEP')
        grouped=defaultdict(list)
        for q in rows:grouped[q.qc_group].append(q)
        keep=[]
        for group,qs in grouped.items():
            vals=[q.target_value for q in qs];zs=robust_z_scores(vals);center=median(vals);mad=median(abs(v-center) for v in vals);std=stdev(vals) if len(vals)>1 else 0
            qc_stats.append({'route':route,'target_name':group[0],'medium_domain':group[1],'task_head':group[2],'train_source_n':len(vals),'median':center,'MAD':mad,'sample_std':std,'fallback':'MAD' if mad>1e-12 else ('sample_std' if std>1e-12 else 'zero'),'threshold':4.0,'minimum_group_n':50})
            for q,z in zip(qs,zs):
                exclude=len(vals)>=50 and z is not None and abs(z)>4
                qc_log.append({'route':route,'result_id':str(q.payload['result_id']),'original_target_name':group[0],'task_head':group[2],'robust_z':z,'train_group_n':len(vals),'excluded':exclude})
                if not exclude:keep.append(q)
        materialize(keep,route,part,'B_TRAIN_ONLY_QC')
    pd.DataFrame(qc_log).to_parquet(destroot/'train_only_qc_decisions.parquet',index=False)
    pd.DataFrame(qc_stats).to_json(destroot/'train_only_qc_fitted_stats.json',orient='records',indent=2,force_ascii=False)
    r={'status':'MATERIALIZED_PENDING_INDEPENDENT_ACCEPTANCE','source_database':str(db),'source_database_sha256':sha(db),'source_table':'task_records BEFORE global QC','candidate_source_n':raw_n,'quality_admission_counts':[{'route':r,'reason':why,'n':n} for (r,why),n in sorted(admission.items())],'frozen_head_scope':'original D_SUBMITTED route task-head set; no whole-universe count cutoff','inherited_parent_changes':0,'new_parent_n':len(parent_map)-len(old_map),'parent_ledger_sha256':sha(destroot/'parent_assignment_ledger.parquet'),'qc_group':['target_name','original primary medium_domain','task_head'],'qc_fit_split':'train raw source records only','qc_scale':'original W00 ptox_mol_l / M00 neg_log10_mg_kg before aggregation and molar conversion','aggregation':'frozen reference/test median -> reference-recency weighted mean, stored under target_value_median','medium_expansion':'after original QC aggregation, retain target route in medium_domains and weight=1/len(domains)','eval':'single common_eval per route; never screened for target extremes','files':manifest,'qc_excluded_source_n':sum(x['excluded'] for x in qc_log),'limitations':['legacy medium/task mapping retained','reference-recency weighting retained; not causal source-isolation claim','source_exposure_time_count>1 must fail S1 recovery or use atomic condition closure','dataset differs from D_SUBMITTED; metrics cannot be paired across different test IDs']}
    (destroot/'E31_data_contract.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    (destroot/'REBUILD_STATUS.json').write_text(json.dumps({'status':'REBUILD_COMPLETE_PENDING_INDEPENDENT_ACCEPTANCE','reason':'Post-aggregation medium_domains expansion restored','contract_sha256':sha(destroot/'E31_data_contract.json')}),encoding='utf-8')
    print(json.dumps({'status':r['status'],'raw_candidate_n':raw_n,'qc_excluded_source_n':r['qc_excluded_source_n'],'files':manifest},ensure_ascii=False,indent=2))
if __name__=='__main__':main()

"""Strict source-mean qualifier whitelist audit; no threshold tuning or target inspection."""
import json,sqlite3,math
from collections import Counter
import pandas as pd
from audit_sources import REV,OUT,sha

def operator(v):
    if v is None or (isinstance(v,float) and math.isnan(v)):return ''
    return str(v).strip().replace('≤','<=').replace('≥','>=')

def classify(row):
    op=operator(row['conc1_mean_op'])
    # Only absent qualifier is admitted; '=' waits for source dictionary verification.
    if op!='': return 'censored_mean_operator' if op in {'<','<=','>','>='} else 'unapproved_mean_operator'
    if row['tox_value_source']!='mean':return 'not_reported_mean'
    if row['value_quality']!='exact':return 'legacy_quality_not_exact'
    if row['tox_value_imputed'] not in (0,False):return 'imputed_or_unknown'
    values=[row.get('standard_value_mol_l'),row.get('standard_value_mg_l'),row.get('standard_value_mg_kg')]
    if not any(isinstance(x,(int,float)) and math.isfinite(x) and x>0 for x in values):return 'no_positive_standard_mean'
    return 'strict_source_mean_eligible'

def main():
    db=REV/'01_冻结来源/data/submitted_v1_2_57.sqlite';c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
    cols='result_id,conc1_mean_op,conc1_min_op,conc1_max_op,tox_value_source,tox_value_imputed,value_quality,standard_value_mol_l,standard_value_mg_l,standard_value_mg_kg,conc1_unit,target_name,medium_domain'
    source=pd.read_sql_query('select '+cols+' from target_records',c)
    source['normalized_mean_op']=source.conc1_mean_op.map(operator)
    source.groupby(['normalized_mean_op','value_quality','tox_value_source'],dropna=False).size().reset_index(name='n').to_csv(OUT/'source_mean_operator_enumeration.csv',index=False)
    subset_cols=['result_id']+[x for x in cols.split(',') if x!='result_id']
    if source[subset_cols].drop_duplicates().result_id.duplicated().any():raise ValueError('Conflicting duplicate source IDs')
    source=source.drop_duplicates('result_id')
    source['source_exact_status']=[classify(r) for r in source.to_dict('records')]
    statuses=source.set_index('result_id').source_exact_status.to_dict()
    outputs=[];summary=[]
    for route in ['W00','M00']:
        d=pd.read_parquet(OUT/f'D_SUBMITTED_{route}_prepared_metadata.parquet')
        for row in d.to_dict('records'):
            keys=json.loads(row['result_ids']);reasons=sorted(set(statuses.get(int(k),'missing_source_id') for k in keys))
            good=reasons==['strict_source_mean_eligible']
            outputs.append({'route':route,'stable_record_id':row['stable_record_id'],'source_result_ids_json':row['source_result_ids_json'],'all_sources_strict_mean_eligible':good,'source_reasons':json.dumps(reasons),'parent_resolved':bool(row['canonical_parent'])})
    ledger=pd.DataFrame(outputs);ledger.to_parquet(OUT/'strict_exact_source_admission.parquet',index=False)
    for route,d in ledger.groupby('route'):
        summary.append({'route':route,'submitted_rows':len(d),'all_source_strict_mean_n':int(d.all_sources_strict_mean_eligible.sum()),'strict_and_resolved_n':int((d.all_sources_strict_mean_eligible & d.parent_resolved).sum())})
    r={'status':'SOURCE_QUALIFIER_GATE_AUDITED_NOT_COMPLETE_QUALITY_ADMISSION','source_database_sha256':sha(db),'raw_operator_histogram':source.normalized_mean_op.value_counts(dropna=False).to_dict(),'strict_whitelist':['blank_or_SQL_NULL_no_qualifier'],'equals_policy':'not admitted absent verified source dictionary; not inferred exact','source_predicate':'unqualified source mean, no imputation, legacy exact, positive standardized concentration','aggregation_policy':'all original source rows must pass; never remove internal source then reuse old aggregate target','remaining_gates':['legal unit and route-specific MW check','full-structure independent quality policy','source mean-op null semantics against raw clean database'],'routes':summary}
    (OUT/'strict_exact_source_audit.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

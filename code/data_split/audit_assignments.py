"""Independent aggregate identity overlap audit; no target/prediction columns."""
import json
from itertools import combinations
import pandas as pd
from audit_sources import OUT,sha

def main():
    a=pd.read_parquet(OUT/'split_assignments.parquet')
    m=pd.concat([pd.read_parquet(OUT/f'D_SUBMITTED_{r}_prepared_metadata.parquet',columns=['route','stable_record_id','test_ids','reference_numbers']) for r in ['W00','M00']])
    a=a.merge(m,on=['route','stable_record_id'],validate='many_to_one')
    rows=[]; summaries=[]
    for (route,sid),d in a.groupby(['route','split_id']):
        groups={'parent':['canonical_parent'],'combination':['canonical_parent','latin_name','model_head'],'condition':['canonical_parent','latin_name','model_head','condition_effect_key','condition_time_key']}
        overlap={}
        for name,cols in groups.items():
            overlap[name]=int((d.groupby(cols,dropna=False).assigned_split.nunique()>1).sum())
        for name,col in [('result','source_result_ids_json'),('test','test_ids'),('reference','reference_numbers')]:
            exploded=d[['assigned_split',col]].copy();exploded[col]=exploded[col].map(json.loads);exploded=exploded.explode(col)
            overlap[name]=int((exploded.groupby(col).assigned_split.nunique()>1).sum())
        assert overlap['result']==0 and overlap['condition']==0
        if sid in ['S2_COMBINATION','S3_PARENT']: assert overlap['combination']==0
        if sid=='S3_PARENT': assert overlap['parent']==0
        train=d[d.assigned_split=='train'];test=d[d.assigned_split=='test']
        train_combo=set(map(tuple,train[['canonical_parent','latin_name','model_head']].to_numpy()))
        test_combo=list(map(tuple,test[['canonical_parent','latin_name','model_head']].to_numpy()))
        summaries.append({'route':route,'split_id':sid,'cross_partition_group_counts':overlap,'test_n':len(test),'test_parent_seen_train_n':int(test.canonical_parent.isin(train.canonical_parent).sum()),'test_species_seen_train_n':int(test.latin_name.isin(train.latin_name).sum()),'test_head_seen_train_n':int(test.model_head.isin(train.model_head).sum()),'test_combination_seen_train_n':sum(x in train_combo for x in test_combo),'test_complete_condition_n':int(test.condition_key_complete.sum())})
        for head,h in d.groupby('model_head'):
            row={'route':route,'split_id':sid,'model_head':head}
            for part in ['train','valid','test']:row[part+'_n']=int(h.assigned_split.eq(part).sum())
            row['train_valid_eligible_35_5']=row['train_n']>=35 and row['valid_n']>=5
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'head_support_metadata.csv',index=False)
    report={'status':'PASS_IDENTITY_ASSERTIONS_NOT_TRAINING_ACCEPTANCE','assignment_sha256':sha(OUT/'split_assignments.parquet'),'test_and_reference_overlap':'Descriptive: these boundaries enforce source-result closure, not study/test isolation','task_eligibility':'training/validation only; test n descriptive, negative R2 never used','audits':summaries}
    (OUT/'split_identity_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

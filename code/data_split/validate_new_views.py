"""Validate shared cohorts, all-row denominators and native-scale censor bounds."""
import json
from collections import defaultdict
import numpy as np
import pandas as pd
from audit_sources import OUT,sha

def main():
    checks=[];heads=[]
    old=pd.read_parquet(OUT/'split_assignments.parquet');old=old[old.split_id=='S3_PARENT']
    old_map={(r,p):part for r,p,part in old[['route','canonical_parent','assigned_split']].drop_duplicates().itertuples(index=False,name=None)}
    for route in ['W00','M00']:
        accum=[]
        for part in ['train','valid','test']:
            exact=pd.read_parquet(OUT/f'E30/exact/{route}/{part}.parquet')
            full=pd.read_parquet(OUT/f'E30/exact_plus_censored/{route}/{part}.parquet')
            chosen=full[~full.is_censored]
            assert exact.stable_record_id.tolist()==chosen.stable_record_id.tolist()
            assert np.array_equal(exact.target.to_numpy(),chosen.target.to_numpy())
            censor=full[full.is_censored]
            assert censor.target.isna().all()
            assert (censor.bound_lower.notna() ^ censor.bound_upper.notna()).all()
            assert np.allclose(-np.log10(censor.censor_bound_molar_concentration.to_numpy()),censor.bound_lower.fillna(censor.bound_upper).to_numpy(),atol=1e-12,rtol=0)
            lower=censor.censor_operator_concentration.str.startswith('<')
            assert lower.equals(censor.bound_lower.notna())
            for p in full.canonical_parent.unique():
                if (route,p) in old_map:assert old_map[(route,p)]==part
            checks.append({'experiment':'E30','route':route,'part':part,'exact_n':len(exact),'censored_n':len(censor),'point_identity_value_equality':True,'bound_conversion_and_direction':True})
            accum.append(full[['model_head','assigned_split','is_censored']])
        e=pd.concat(accum);n=e[~e.is_censored].groupby(['model_head','assigned_split']).size().unstack(fill_value=0).reindex(columns=['train','valid','test'],fill_value=0)
        for head,row in n.iterrows():heads.append({'experiment':'E30','route':route,'model_head':head,**row.to_dict(),'status':'NO_EXACT_TRAIN_LABEL' if row.train==0 else ('NO_EXACT_VALID_LABEL' if row.valid==0 else 'TRAIN_VALID_AVAILABLE')})
    e31=OUT/'E31/E31_data_contract.json'
    if e31.exists():
        contract=json.loads(e31.read_text(encoding='utf-8'))
        for route in ['W00','M00']:
            data={}
            for name,part in [('A_KEEP','train'),('B_TRAIN_ONLY_QC','train'),('common_eval','valid'),('common_eval','test')]:
                d=pd.read_parquet(OUT/f'E31/{name}/{route}/{part}.parquet');data[(name,part)]=d
                assert d.target.notna().all() and np.isfinite(d.target).all()
            sets=[set(data[(name,part)].canonical_parent) for name,part in [('A_KEEP','train'),('common_eval','valid'),('common_eval','test')]]
            assert all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i+1,3))
            for key,d in data.items():
                for p in d.canonical_parent.unique():
                    if (route,p) in old_map:assert old_map[(route,p)]==key[1]
            checks.append({'experiment':'E31','route':route,'parent_overlap':0,'inherited_partition_changes':0,'A_train_n':len(data[('A_KEEP','train')]),'B_train_n':len(data[('B_TRAIN_ONLY_QC','train')]),'common_valid_n':len(data[('common_eval','valid')]),'common_test_n':len(data[('common_eval','test')])})
            for arm in ['A_KEEP','B_TRAIN_ONLY_QC']:
                d=pd.concat([data[(arm,'train')],data[('common_eval','valid')],data[('common_eval','test')]])
                n=d.groupby(['model_head','assigned_split']).size().unstack(fill_value=0).reindex(columns=['train','valid','test'],fill_value=0)
                for head,row in n.iterrows():heads.append({'experiment':'E31_'+arm,'route':route,'model_head':head,**row.to_dict(),'status':'NO_EXACT_TRAIN_LABEL' if row.train==0 else ('NO_EXACT_VALID_LABEL' if row.valid==0 else 'TRAIN_VALID_AVAILABLE')})
    pd.DataFrame(heads).to_csv(OUT/'E30_E31_head_denominators.csv',index=False)
    result={'status':'PASS_DATA_VIEW_ASSERTIONS_PENDING_INDEPENDENT_TRAINING_GATE','checks':checks,'performance_metrics_read':False,'source_bound_numeric_checks':'concentration->p limit transformation verification only; no test model error calculated'}
    (OUT/'E30_E31_view_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

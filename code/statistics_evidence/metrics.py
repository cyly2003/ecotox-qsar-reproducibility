"""Auditable native-scale exact-point metrics with explicit complete cohort denominators."""
from dataclasses import dataclass,asdict
import hashlib,json
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Context:
    route:str
    split_id:str
    comparison_cohort_id:str
    target_scale:str
    def __post_init__(self):
        if self.route not in ('W00','M00') or not all([self.split_id,self.comparison_cohort_id,self.target_scale]):
            raise ValueError('Explicit route, split, cohort and target scale required')
        if self.target_scale!=('neg_log10_mol_l' if self.route=='W00' else 'neg_log10_mol_kg'):
            raise ValueError('Route/native target scale mismatch')

def identity_hash(ids):
    return hashlib.sha256(json.dumps(sorted(str(x) for x in ids),ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def align(predictions,cohort,context):
    """Missing identity rows fail. Explicit NULL predictions are legitimate non-estimates."""
    required={'stable_record_id','target','model_head','canonical_parent'}
    if not required<=set(cohort):raise ValueError('Incomplete cohort schema')
    if not {'stable_record_id','prediction'}<=set(predictions):raise ValueError('Incomplete prediction schema')
    for name,d in [('cohort',cohort),('predictions',predictions)]:
        if d.stable_record_id.isna().any() or d.stable_record_id.astype(str).duplicated().any():raise ValueError(name+' identities nonunique/missing')
        for col,value in asdict(context).items():
            if col in d and not d[col].astype(str).eq(value).all():raise ValueError(name+' '+col+' context mismatch')
    if set(predictions.stable_record_id.astype(str))!=set(cohort.stable_record_id.astype(str)):
        raise ValueError('Prediction rows must match full expected cohort; preserve non-estimates as NULL')
    c=cohort.copy();p=predictions.copy();c['stable_record_id']=c.stable_record_id.astype(str);p['stable_record_id']=p.stable_record_id.astype(str)
    c=c.set_index('stable_record_id').sort_index();p=p.set_index('stable_record_id').reindex(c.index)
    for col in ['target','model_head','canonical_parent']:
        if col in p:
            if col=='target':
                if not np.array_equal(pd.to_numeric(p[col]).to_numpy(),pd.to_numeric(c[col]).to_numpy(),equal_nan=True):raise ValueError('Prediction target differs from cohort')
            elif not p[col].fillna('__NULL__').astype(str).equals(c[col].fillna('__NULL__').astype(str)):raise ValueError('Prediction '+col+' differs from cohort')
    if c.model_head.isna().any() or c.canonical_parent.isna().any():raise ValueError('Missing cohort head/parent identity')
    c['target']=pd.to_numeric(c.target,errors='raise');c['prediction']=pd.to_numeric(p.prediction,errors='raise')
    if np.isinf(c.prediction.to_numpy()).any() or np.isinf(c.target.to_numpy()).any():raise ValueError('Infinite targets/predictions invalid; absent estimates must be NULL')
    if 'is_censored' not in c:c['is_censored']=False
    if c.is_censored.isna().any() or not c.is_censored.isin([True,False,0,1]).all():raise ValueError('Invalid censor flag')
    c['is_censored']=c.is_censored.astype(bool)
    c['exact_expected']=~c.is_censored & c.target.notna()
    c['scored']=c.exact_expected & c.prediction.notna()
    return c.reset_index()

def point_metrics(y,p):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    if len(y)==0:return {'mae':None,'rmse':None,'r2':None}
    err=p-y;ss=float(np.square(y-y.mean()).sum())
    return {'mae':float(np.abs(err).mean()),'rmse':float(np.sqrt(np.square(err).mean())),
            'r2':float(1-np.square(err).sum()/ss) if len(y)>=2 and ss>0 else None}

def evaluate(predictions,cohort,context):
    d=align(predictions,cohort,context);rows=[]
    for head,h in d.groupby('model_head',sort=True):
        q=h[h.scored];m=point_metrics(q.target,q.prediction)
        rows.append({'model_head':head,'n_cohort':len(h),'n_exact_expected':int(h.exact_expected.sum()),'n_predicted_exact':len(q),'n_unestimated_exact':int(h.exact_expected.sum())-len(q),'coverage_exact':float(len(q)/h.exact_expected.sum()) if h.exact_expected.sum() else None,**m,'r2_status':'DEFINED' if m['r2'] is not None else ('N_LT_2' if len(q)<2 else 'CONSTANT_TARGET')})
    heads=pd.DataFrame(rows);q=d[d.scored];pooled=point_metrics(q.target,q.prediction)
    summary={**asdict(context),'cohort_identity_sha256':identity_hash(d.stable_record_id),'n_cohort':len(d),'n_exact_expected':int(d.exact_expected.sum()),'n_censored':int(d.is_censored.sum()),'n_missing_uncensored_target':int((~d.is_censored & d.target.isna()).sum()),'n_predicted_exact':len(q),'n_unestimated_exact':int(d.exact_expected.sum())-len(q),'coverage_exact':float(len(q)/d.exact_expected.sum()) if d.exact_expected.sum() else None,'n_heads_expected':len(heads),'n_heads_with_prediction':int(heads.n_predicted_exact.gt(0).sum()),'n_negative_r2_heads':int(heads.r2.lt(0).sum()),'pooled_scored_subset_only':True}
    for key,value in pooled.items():summary['pooled_'+key]=value
    for metric in ['mae','rmse','r2']:
        vals=pd.to_numeric(heads[metric],errors='coerce').dropna()
        summary['macro_'+metric]=float(vals.mean()) if len(vals) else None
        summary['median_'+metric]=float(vals.median()) if len(vals) else None
        summary['n_defined_'+metric+'_heads']=len(vals)
    return summary,heads,d

def cluster_replicate_delta(cell_parent,cell_head,cell_n,cell_delta_sum,parent_multiplicity,n_heads):
    """Multiply every cluster contribution by its draw count, never reduce draws to isin."""
    w=np.asarray(parent_multiplicity)[cell_parent]
    n=np.bincount(cell_head,weights=w*cell_n,minlength=n_heads)
    sums=np.bincount(cell_head,weights=w*cell_delta_sum,minlength=n_heads)
    pooled=float(sums.sum()/n.sum()) if n.sum() else np.nan
    macro=float(np.mean(sums/n)) if np.all(n>0) else np.nan
    return pooled,macro

def paired_compare(baseline,variant,cohort,baseline_context,variant_context,*,replicates=2000,bootstrap_seed=20260914):
    if baseline_context!=variant_context:raise ValueError('Cross-route/split/cohort/scale comparison forbidden')
    if replicates<1:raise ValueError('Positive bootstrap replicate budget required')
    b=align(baseline,cohort,baseline_context);v=align(variant,cohort,variant_context)
    paired=b.scored & v.scored
    d=b.loc[paired,['stable_record_id','canonical_parent','model_head','target','prediction']].copy()
    if d.empty:raise ValueError('No paired finite exact predictions')
    d['baseline_ae']=np.abs(d.prediction-d.target);d['variant_ae']=np.abs(v.loc[paired,'prediction'].to_numpy()-d.target.to_numpy());d['delta_ae']=d.baseline_ae-d.variant_ae
    fixed_heads=sorted(d.model_head.unique());parents=sorted(d.canonical_parent.unique())
    cells=d.groupby(['canonical_parent','model_head']).delta_ae.agg(['sum','size']).reset_index()
    pi={x:i for i,x in enumerate(parents)};hi={x:i for i,x in enumerate(fixed_heads)}
    cp=cells.canonical_parent.map(pi).to_numpy();ch=cells.model_head.map(hi).to_numpy();cn=cells['size'].to_numpy();cd=cells['sum'].to_numpy()
    observed_pooled,observed_macro=cluster_replicate_delta(cp,ch,cn,cd,np.ones(len(parents)),len(fixed_heads))
    rng=np.random.default_rng(bootstrap_seed);samples=[]
    for i in range(replicates):
        multiplicity=rng.multinomial(len(parents),np.full(len(parents),1/len(parents)))
        pooled,macro=cluster_replicate_delta(cp,ch,cn,cd,multiplicity,len(fixed_heads))
        samples.append({'replicate':i,'delta_mae_pooled':pooled,'delta_mae_macro':macro,'macro_valid':bool(np.isfinite(macro))})
    draws=pd.DataFrame(samples)
    summary={**asdict(baseline_context),'delta_definition':'baseline_MAE_minus_variant_MAE_positive_favors_variant','n_expected_exact':int(b.exact_expected.sum()),'n_paired_predicted_exact':len(d),'n_not_paired_exact':int(b.exact_expected.sum())-len(d),'paired_identity_sha256':identity_hash(d.stable_record_id),'n_parent_clusters':len(parents),'fixed_macro_heads':fixed_heads,'bootstrap_replicates':replicates,'bootstrap_seed':bootstrap_seed,'bootstrap_unit':'canonical_parent','cluster_multiplicity_retained':True,'missing_head_policy':'replicate macro is NA when any fixed paired head absent','delta_mae_pooled':observed_pooled,'delta_mae_macro':observed_macro,'small_cluster_warning':len(parents)<30}
    for metric in ['pooled','macro']:
        valid=draws['delta_mae_'+metric].dropna();summary['bootstrap_valid_'+metric]=len(valid);summary['bootstrap_invalid_'+metric]=replicates-len(valid)
        summary['delta_mae_'+metric+'_ci95']=[float(x) for x in np.quantile(valid,[.025,.975])] if len(valid) else [None,None]
    return summary,draws,d

def four_seed_summary(predictions_by_seed,cohort,context,*,expected_seeds=(42,2042,3407,8417)):
    if set(predictions_by_seed)!=set(expected_seeds):raise ValueError('Exactly all expected training seeds required; no partial ensemble')
    seed_rows=[];aligned=[]
    for seed in expected_seeds:
        summary,_,d=evaluate(predictions_by_seed[seed],cohort,context);summary['seed']=seed;seed_rows.append(summary);aligned.append(d)
    prediction_matrix=np.column_stack([d.prediction.to_numpy() for d in aligned])
    full=np.isfinite(prediction_matrix).all(axis=1);ensemble=np.full(len(cohort),np.nan)
    ensemble[full]=prediction_matrix[full].mean(axis=1)
    ef=aligned[0][['stable_record_id']].copy();ef['prediction']=ensemble
    es,eh,ed=evaluate(ef,cohort,context);es['aggregation']='prediction_level_ensemble_all_four_seeds_required_per_row'
    ss=pd.DataFrame(seed_rows);moments=[]
    keys=[c for c in ss if c.startswith(('pooled_','macro_','median_')) and c!='pooled_scored_subset_only']
    for metric in keys:
        values=pd.to_numeric(ss[metric],errors='coerce').dropna()
        moments.append({'metric':metric,'seed_n_defined':len(values),'mean':float(values.mean()) if len(values) else None,'sd_ddof1':float(values.std(ddof=1)) if len(values)>1 else None})
    return {'seed_metrics':ss,'seed_mean_sd':pd.DataFrame(moments),'ensemble_summary':es,'ensemble_heads':eh,'ensemble_predictions':ed}

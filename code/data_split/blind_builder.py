"""Prediction-blind, route-independent S1/S2/S3 proposal builder. No training.

Only accepts identity metadata. Caller resolves raw exposure-time sets first.
Missing condition values are conservatively co-grouped and explicitly flagged.
Assign the UNION of all sensitivity universes once, then subset the ledger.
"""
import hashlib
import json
from collections import defaultdict, Counter
import pandas as pd

FIELDS = ['route','stable_record_id','canonical_parent','latin_name','model_head',
          'source_result_ids_json','condition_effect_key','condition_time_key']
PARTS = ('train','valid','test')

def digest(x):
    return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def build(frame, split_id, seed=20260914, fractions=(.64,.16,.20)):
    if split_id not in ('S1_CONDITION','S2_COMBINATION','S3_PARENT','S_SOURCE'):
        raise ValueError('S0 must preserve original assignments, never regenerate here')
    allowed=FIELDS+(['test_ids'] if split_id=='S_SOURCE' else [])
    if set(frame.columns) != set(allowed):
        raise ValueError('Pass exact identity allowlist; labels/predictions forbidden')
    if len(fractions)!=3 or any(x<=0 for x in fractions) or abs(sum(fractions)-1)>1e-12:
        raise ValueError('Invalid fractions')
    d=frame.sort_values(['route','stable_record_id']).reset_index(drop=True).copy()
    if split_id in ('S1_CONDITION','S_SOURCE'):
        for key in d.condition_time_key:
            if isinstance(key,str) and key.startswith('[') and len(json.loads(key))>1:
                raise ValueError('Multi-time aggregate requires atomic condition closure; refusing S1/S_SOURCE')
    if d.empty or d.duplicated(['route','stable_record_id']).any():
        raise ValueError('Empty or duplicate universe')
    for col in FIELDS:
        if d[col].isna().any() or d[col].astype(str).str.strip().eq('').any():
            raise ValueError('Unresolved metadata: '+col)
    out=[]
    for route, r in d.groupby('route',sort=True):
        r=r.reset_index(drop=True); parents=list(range(len(r)))
        def find(i):
            while parents[i]!=i:
                parents[i]=parents[parents[i]]; i=parents[i]
            return i
        def union(a,b):
            a,b=find(a),find(b)
            parents[max(a,b)]=min(a,b)
        source_seen={}; key_seen={}; keys=[]; sources=[]
        for i,row in enumerate(r.to_dict('records')):
            ss=json.loads(row['source_result_ids_json'])
            if not isinstance(ss,list) or not ss or any(x is None or isinstance(x,(bool,list,dict)) for x in ss):
                raise ValueError('Invalid source identities')
            ss=sorted(set(str(x).strip() for x in ss))
            if '' in ss: raise ValueError('Blank source identity')
            if split_id=='S_SOURCE':
                tests=json.loads(row['test_ids'])
                if not isinstance(tests,list) or not tests or any(x is None or isinstance(x,(bool,list,dict)) or not str(x).strip() for x in tests):
                    raise ValueError('Reliable test IDs required for S_SOURCE')
                ss=['result:'+x for x in ss]+['test:'+str(x).strip() for x in tests]
            sources.append(ss)
            key=[row['canonical_parent']]
            if split_id!='S3_PARENT': key += [row['latin_name'],row['model_head']]
            if split_id in ('S1_CONDITION','S_SOURCE'): key += [row['condition_effect_key'],row['condition_time_key']]
            key=digest(key); keys.append(key)
            for value, seen in [(key,key_seen)]+[(x,source_seen) for x in ss]:
                if value in seen: union(i,seen[value])
                else: seen[value]=i
        components=defaultdict(list)
        for i in range(len(r)): components[find(i)].append(i)
        identities=r.stable_record_id.tolist(); heads=r.model_head.tolist()
        groups={digest([route,split_id,sorted(identities[i] for i in idx)]):idx for idx in components.values()}
        totals=Counter(r.model_head); counts={h:[0,0,0] for h in totals}
        # One deterministic order; no test labels, no candidate performance selection.
        assignments={}; gids={}
        for g,idx in sorted(groups.items(),key=lambda x:(-len(x[1]),digest([seed,x[0]]))):
            profile=Counter(heads[i] for i in idx)
            def cost(p):
                delta=sum(((counts[h][p]+n-totals[h]*fractions[p])**2-(counts[h][p]-totals[h]*fractions[p])**2)/max(totals[h]*fractions[p],1) for h,n in profile.items())
                return delta,digest([seed,g,p])
            p=min(range(3),key=cost)
            for h,n in profile.items(): counts[h][p]+=n
            for i in idx: assignments[i]=PARTS[p]; gids[i]=g
        r['split_id']=split_id; r['assigned_split']=[assignments[i] for i in range(len(r))]
        r['split_group_id']=[gids[i] for i in range(len(r))]
        for values in (keys,):
            check=defaultdict(set)
            for i,k in enumerate(values): check[k].add(assignments[i])
            assert all(len(v)==1 for v in check.values())
        check=defaultdict(set)
        for i,ss in enumerate(sources):
            for s in ss: check[s].add(assignments[i])
        assert all(len(v)==1 for v in check.values())
        out.append(r)
    return pd.concat(out,ignore_index=True)

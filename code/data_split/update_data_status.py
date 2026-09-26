"""Build compact data delivery status from actual contracts, without training claims."""
import json
from audit_sources import ROOT,OUT,sha

def main():
    paths=['data_contract.json','submitted_metadata_audit.json','split_prelock_report.json','split_identity_audit.json','S_SOURCE_prelock_report.json','strict_exact_source_audit.json','E30/E30_data_contract.json','E31/E31_data_contract.json']
    rows=[]
    for path in paths:
        p=OUT/path
        if p.exists():
            d=json.loads(p.read_text(encoding='utf-8'));rows.append({'path':path,'status':d.get('status','DISCOVERY'),'sha256':sha(p)})
        else:rows.append({'path':path,'status':'NOT_YET_WRITTEN'})
    runtime=json.loads((ROOT/'00_总控与审稿意见索引/agent_runtime.json').read_text(encoding='utf-8'))
    actual=next(x for x in runtime['agents'] if x['role']=='/root/data_split')
    d={'status':'DATA_PREPARATION_AND_MATERIALIZATION_NOT_TRAINING_COMPLETION','contracts':rows,'runtime':actual,'runtime_evidence_scope':'child read parent-maintained runtime record pointing to session turn_context; did not directly inspect original session','current_boundary_rules':['D_SUBMITTED raw S0 remains original','S1/S2/S3 share 202381 W00 and 10878 M00 resolved-parent rows','S1 result+complete condition; S_SOURCE additionally original test IDs; neither unions references','E30 inherits old S3 parents and adds hash-assigned parents','E31 source-level train-only QC before original aggregation and soil molar conversion'],'main_agent_must_accept_before_training':True}
    (OUT/'DATA_STATUS.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(rows,ensure_ascii=False))
if __name__=='__main__':main()

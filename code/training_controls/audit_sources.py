"""Read only source audit; writes only the task-owned audit report directory."""
import hashlib
import json
from pathlib import Path

def main():
    new = Path(__file__).resolve().parents[2]
    revision = new.parent
    names = [
        '01_冻结来源/code_current/scripts/run_v1_2_56_framework_confirmation_remote.sh',
        '01_冻结来源/code_current/scripts/resume_v1_2_57_confirmation_after_validator_fix.sh',
        '01_冻结来源/code_current/scripts/run_v1_2_56_framework_hpo_remote.sh',
        '01_冻结来源/code_current/scripts/summarize_v1_2_56_framework_hpo.py',
        '01_冻结来源/code_current/qsar_tl/training/traditional_domain_comparison.py',
        '01_冻结来源/code_current/scripts/run_v1_2_58_no_metal_traditional_ml.py',
        '01_冻结来源/code_current/scripts/run_v1_2_62_frozen_encoder_tree_injection.py',
        '01_冻结来源/code_current/scripts/run_v1_2_68_traditional_qsar_frozen_probes.py',
        '01_冻结来源/reference_models/W00/manifest.json',
        '01_冻结来源/reference_models/M00/manifest.json',
        '04_执行代码/revision_pipeline/train.py',
        '04_执行代码/revision_pipeline/train_contract.py',
        '04_执行代码/qsar_tl/training/deep_experiment.py',
        '04_执行代码/qsar_tl/training/deep_train.py',
        '04_执行代码/qsar_tl/training/toxicity_binning.py',
        '04_执行代码/qsar_tl/modeling/network.py',
    ]
    audit = {'schema':'training_controls_source_audit_v1', 'source_files':{},
             'original_seeds':[42,2042,3407,8417],
             'actual_model':'gpt-6-astra','actual_reasoning_effort':'medium',
             'runtime_evidence':'Parent read current session turn_context, reported via collaboration; not independently inspected by this role',
             'local_tests':'5 passed CPU scope/seed/boundary/input-hash/bundle-resume tests; compileall passed; Windows temporary-directory ACL required approved unsandboxed test execution',
             'real_legacy_preprocessing_test':'Parent reports previous source snapshot remote --legacy 4 tests passed; new boundary/resume tests local passed, new full remote suite pending',
             'training_performed_by_this_agent':False}
    for name in names:
        p = revision / name
        audit['source_files'][name] = {'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
    for route in ['W00','M00']:
        d=json.loads((revision/f'01_冻结来源/reference_models/{route}/manifest.json').read_text(encoding='utf-8'))
        phase=d if route=='W00' else d['finetune_mgkg']
        audit[route] = {'reference_seed':d['seed'],'reference_split':d['split_name'],
                        'learning_rate':phase['learning_rate'],'batch_size':phase['batch_size'],
                        'dropout':d['dropout'],'weight_decay':d['weight_decay'],
                        'toxicity_binning':d['toxicity_binning'],
                        'regression_loss':phase['regression_loss']}
    out=new/'03_同输入共享对照'; out.mkdir(parents=True,exist_ok=True)
    (out/'training_source_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__': main()

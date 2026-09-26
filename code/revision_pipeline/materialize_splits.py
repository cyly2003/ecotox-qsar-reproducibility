"""Materialize immutable physical partitions from the qualified observation table."""
from pathlib import Path
import json
import argparse
import pandas as pd
from revision_pipeline.splits import build_revision_splits
from revision_pipeline.train_contract import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='revision_inputs_v2')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if Path(args.data_dir).name != args.data_dir:
        raise ValueError('Input directory name must have one component')
    out = root / '02_清洗重建' / args.data_dir
    source = out / 'all_observations.parquet'
    combined = pd.read_parquet(source)
    print(f'Assigning 3 boundaries on {len(combined)} observations', flush=True)
    assignments, audit = build_revision_splits(combined, seed=42)
    audit['observation_file_sha256'] = digest(source)
    assignments.to_parquet(out / 'split_assignments.parquet', index=False)
    files = []
    for (boundary, route, part), group in assignments.groupby(['boundary_id', 'route', 'assigned_split']):
        joined = group[['route', 'stable_record_id', 'assigned_split', 'split_group_id']].merge(combined, on=['route', 'stable_record_id'], validate='one_to_one')
        joined['boundary_id'] = boundary
        directory = out / 'splits' / boundary / route
        directory.mkdir(parents=True, exist_ok=True)
        file = directory / f'{part}.parquet'
        joined.to_parquet(file, index=False)
        files.append({'relative_path': str(file.relative_to(out)).replace('\\','/'), 'sha256': digest(file), 'n': len(joined), 'point_n': int((~joined.is_censored).sum())})
    audit['physical_files'] = files
    (out / 'split_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame([{'boundary': b['boundary_id'], **row} for b in audit['boundaries'] for row in b['support']]).to_csv(out / 'head_split_support.csv', index=False, encoding='utf-8-sig')
    print(json.dumps({'physical_files': files, 'status': 'split_materialization_complete'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

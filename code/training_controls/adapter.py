"""Reuse legacy numerical contracts, fitting every STL preprocessor per head.

This module is a preparation/config adapter, not a training launcher. Pass loaded
physical train/valid frames from revision_pipeline.train.load_frame. Prediction
must call transform with the corresponding group's own persisted preprocessor.
"""
from __future__ import annotations
from dataclasses import replace
import hashlib
import json

SEEDS = (42, 2042, 3407, 8417)
VARIANTS = ('MTL_FULL', 'STL_FULL', 'MTL_MOL', 'STL_MOL')
CANDIDATES = {
    'C0': dict(learning_rate=.0005, dropout=.10, weight_decay=.00001),
    'C1': dict(learning_rate=.0003, dropout=.10, weight_decay=.00001),
    'C2': dict(learning_rate=.0008, dropout=.10, weight_decay=.00001),
    'C3': dict(learning_rate=.0005, dropout=.05, weight_decay=.00001),
    'C4': dict(learning_rate=.0005, dropout=.15, weight_decay=.00001),
    'C5': dict(learning_rate=.0005, dropout=.10, weight_decay=.00010),
}

def specification(legacy, variant):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant {variant}')
    full = replace(legacy.ABLATION_SPECS['full'], use_medium_adapter=False)
    if variant.endswith('_FULL'):
        return full
    return replace(full, name='molecule_only_registered', use_context_numeric=False,
                   use_effect_level_features=False, use_duration_features=False,
                   use_species_lifestage=False, use_other_categorical_context=False)

def _check_frame(frame, partition):
    if frame.empty or set(frame.assigned_split.astype(str)) != {partition}:
        raise ValueError(f'Expected nonempty physical {partition} partition')
    if frame.stable_record_id.isna().any() or frame.stable_record_id.duplicated().any():
        raise ValueError('Duplicate or null stable identities')
    if not frame.is_censored.isin([True, False, 0, 1]).all():
        raise ValueError('Censor flags must be boolean')

def scope_frames(train, valid, variant, head=None):
    """Scope BEFORE fitting; no evaluation/test handle accepted."""
    if variant not in VARIANTS:
        raise ValueError('Unknown variant')
    _check_frame(train, 'train')
    _check_frame(valid, 'valid')
    if set(train.stable_record_id.astype(str)) & set(valid.stable_record_id.astype(str)):
        raise ValueError('Train/valid stable identity overlap')
    if variant.startswith('STL_'):
        if head is None:
            raise ValueError('STL requires exactly one model_head before preprocessing')
        train = train.loc[train.model_head.astype(str).eq(str(head))].copy()
        valid = valid.loc[valid.model_head.astype(str).eq(str(head))].copy()
    elif head is not None:
        raise ValueError('MTL retains the full legal auxiliary pool; head filter forbidden')
    if train.empty:
        raise ValueError('No training records for requested scope')
    return train.reset_index(drop=True), valid.reset_index(drop=True)

def fit_preprocessor(train, contract, legacy, encoder, variant):
    _check_frame(train, 'train')
    if variant.startswith('STL_') and train.model_head.nunique() != 1:
        raise ValueError('STL preprocessing must be task isolated')
    point = train.loc[~train.is_censored.astype(bool)].copy()
    if point.empty:
        raise ValueError('No point training rows for preprocessing')
    point['target_value'] = point.target.astype(float)
    spec = specification(legacy, variant)
    names = tuple(contract['descriptor_names'])
    category = legacy.fit_categorical_maps(point, ablation=spec, min_count=contract['categorical_min_count'])
    stats = legacy.fit_numeric_stats(point, encoder, descriptor_names=names, ablation=spec)
    zscore = legacy.fit_zscore_correction(point, encoder, numeric_stats=stats,
        feature_names=legacy.build_numeric_feature_names(len(names), descriptor_names=names),
        descriptor_names=names, config=legacy.ZScoreCorrectionConfig(
            enabled=contract['zscore_enabled'], threshold=contract['zscore_threshold']), ablation=spec)
    scaler = legacy.fit_target_scaler(point, target_column='target_value',
        mode='per_task_target', fit_indices=list(range(len(point))))
    return {'variant': variant, 'categorical_maps': category, 'numeric_stats': stats,
            'target_scaler': scaler.to_manifest(), 'zscore_correction': zscore.to_manifest(),
            'descriptor_names': list(names), 'task_heads': sorted(point.model_head.astype(str).unique()),
            'fit_ids': sorted(point.stable_record_id.astype(str)), 'fit_partition': 'train_point_only',
            'supervised_encoder_reused': False,
            'toxicity_binning_config': contract.get('toxicity_binning_config'),
            'toxicity_bin_scheme': contract.get('toxicity_bin_scheme'),
            'numeric_dim': len(names) + len(legacy.CONTEXT_NUMERIC_COLUMNS),
            'mol_context_policy': 'zero_numeric_context_no_categorical_embeddings' if variant.endswith('_MOL') else 'full'}

def transform(frame, pre, legacy, encoder):
    """Transform only; target scaling remains task routed, never refitted."""
    if frame.is_censored.astype(bool).any():
        raise ValueError('Core E10 uses point targets; censored loss adapter required for other arms')
    if not set(frame.model_head.astype(str)) <= set(pre['task_heads']):
        raise ValueError('Unknown model head in this preprocessor')
    work = frame.copy()
    work['target_value'] = work.target.astype(float)
    zscore = legacy.ZScoreCorrection(**{k: v for k, v in pre['zscore_correction'].items() if k != 'method'})
    samples = legacy.build_deep_samples(work, encoder=encoder, descriptor_names=tuple(pre['descriptor_names']),
        categorical_maps=pre['categorical_maps'], adapter_map={}, numeric_stats=pre['numeric_stats'],
        target_column='target_value', target_scaler=legacy.TargetScaler(**pre['target_scaler']),
        zscore_correction=zscore, ablation=specification(legacy, pre['variant']),
        toxicity_binning_config=legacy.ToxicityBinningConfig(**pre['toxicity_binning_config']) if pre.get('toxicity_binning_config') else None,
        toxicity_bin_scheme=pre.get('toxicity_bin_scheme'))
    for sample, row in zip(samples, work.to_dict('records')):
        sample.update(stable_record_id=str(row['stable_record_id']), target_value_raw=float(row['target']),
                      is_censored=False, censored_direction_id=0)
    return samples

def prepare_group(train, valid, variant, contract, legacy, encoder, head=None):
    train, valid = scope_frames(train, valid, variant, head)
    train = train.loc[~train.is_censored.astype(bool)].reset_index(drop=True)
    valid = valid.loc[~valid.is_censored.astype(bool)].reset_index(drop=True)
    pre = fit_preprocessor(train, contract, legacy, encoder, variant)
    valid = valid.loc[valid.model_head.astype(str).isin(pre['task_heads'])].reset_index(drop=True)
    return {'preprocessing': pre, 'train_frame': train, 'valid_frame': valid,
            'train_samples': transform(train, pre, legacy, encoder),
            'valid_samples': transform(valid, pre, legacy, encoder) if len(valid) else [],
            'selection_status': 'validation_available' if len(valid) else 'fixed_final_epoch_not_comparison_eligible'}

def model_config(pre, contract, Config):
    return Config(numeric_dim=pre['numeric_dim'], fingerprint_dim=512,
        task_heads=tuple(pre['task_heads']), descriptor_count=len(pre['descriptor_names']),
        categorical_cardinalities={k: max(v.values()) + 1 for k,v in pre['categorical_maps'].items()},
        hidden_dims=tuple(contract['hidden_dims']), dropout=contract['dropout'],
        effect_level_numeric_indices=(8,9,10,11), use_molecular_residual=True,
        use_adapters=False, adapter_count=0, fusion_mode='concat',
        toxicity_bin_count=contract.get('toxicity_bin_count', 0),
        toxicity_binning_mode='aux_classification' if contract.get('toxicity_bin_count', 0) else 'none')

def initialization_contract(seed):
    if seed not in SEEDS:
        raise ValueError('Unregistered original seed')
    return {'torch_seed': seed, 'numpy_seed': seed, 'dataloader_seed': seed,
            'load_state_dict': False, 'encoder_initialization': 'independent_from_scratch'}

def comparison_contract(mtl, stl):
    """Check schema, not equality of fitted maps/statistics across MTL/STL."""
    if not mtl['variant'].startswith('MTL_') or not stl['variant'].startswith('STL_'):
        raise ValueError('Expected MTL then STL')
    if mtl['variant'][4:] != stl['variant'][4:] or mtl['descriptor_names'] != stl['descriptor_names']:
        raise ValueError('Input schema mismatch')
    if set(mtl['categorical_maps']) != set(stl['categorical_maps']):
        raise ValueError('Categorical raw field mismatch')
    if len(stl['task_heads']) != 1 or not set(stl['fit_ids']) <= set(mtl['fit_ids']):
        raise ValueError('STL is not one head within legal MTL train pool')
    if stl['supervised_encoder_reused']:
        raise ValueError('STL cannot reuse supervised MTL encoding')
    return True

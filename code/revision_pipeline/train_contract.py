"""Torch-free contracts for the revision-only training runner."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    indent=2, allow_nan=False), encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def bounds_contract(censored, target, lower, upper):
    """Bounds are already on p scale. Never infer direction from endpoint names."""
    if not censored:
        if not finite(target):
            raise ValueError('Uncensored point requires a finite target.')
        return float(target), float(target)
    lo = float(lower) if finite(lower) else -math.inf
    hi = float(upper) if finite(upper) else math.inf
    if lo == -math.inf and hi == math.inf:
        raise ValueError('Censored observation has no finite bound.')
    if lo >= hi:
        raise ValueError('Censored interval requires lower < upper; equal bounds are point data.')
    if finite(target):
        raise ValueError('Censored target must be null: bound is not an exact response.')
    return lo, hi


def full_contract(path, route):
    """Read the actual Full manifest; Stage3 fields override M00 top-level defaults."""
    d = read_json(path)
    phase = d if route == 'W00' else d['finetune_mgkg']
    loss = phase['regression_loss']
    if d['ablation'] != 'full' or d['fingerprint_size'] != 512 or d['numeric_dim'] != 22:
        raise ValueError('Reference is not the frozen Full RDKit8/Morgan512 contract.')
    if d['descriptor_encoder']['mode'] != 'raw' or d['adapter_cardinality'] != 0:
        raise ValueError('Reference architecture is outside this thin runner.')
    if len(d['molecular_descriptor_names']) != 8 or len(d['categorical_cardinalities']) != 18:
        raise ValueError('Reference input groups must be RDKit8 + 14 numeric + 18 categorical.')
    if loss != {'kind': 'huber', 'huber_delta': 1.0, 'mse_weight': 0.0}:
        raise ValueError('Unexpected reference regression loss.')
    if d['task_weighting'] != 'balanced':
        raise ValueError('Expected balanced task weighting.')
    if route=='M00' and any(float(v) != 1 for v in d['task_weights'].values()):
        raise ValueError('M00 reference Stage3-only task weights are expected to be 1.')
    if not d['ablation_features']['use_molecular_residual']:
        raise ValueError('Reference must retain the molecular residual.')
    return {'batch_size': phase['batch_size'], 'learning_rate': phase['learning_rate'],
            'dropout': d['dropout'], 'optimizer': d['optimizer'],
            'weight_decay': d['weight_decay'], 'gradient_clip_norm': d['gradient_clip_norm'],
            'scheduler': d['scheduler'], 'huber_delta': loss['huber_delta'],
            'mse_loss_weight': loss['mse_weight'],
            'patience': d['early_stopping']['patience'] if route == 'W00' else phase['early_stopping_patience'],
            'min_delta': d['early_stopping']['min_delta'] if route == 'W00' else phase['early_stopping_min_delta'],
            'categorical_min_count': d['categorical_min_count'],
            'zscore_enabled': d['feature_zscore_correction']['enabled'],
            'zscore_threshold': d['feature_zscore_correction']['threshold'],
            'descriptor_names': d['molecular_descriptor_names'], 'hidden_dims': [256, 128],
            'task_weighting_config': {'task_weighting':'balanced','main_task_weight':1.0,
                'toxicity_aux_task_weight':0.35,'bioaccumulation_aux_task_weight':0.2,
                'task_weight_exponent':0.5,'task_weight_min':0.25,'task_weight_max':4.0},
            'task_weighting_evidence':'frozen experiment.remote.easyai.yaml:178-184; M00 Stage3-only empty Stage1 train yields unit weights',
            'hidden_dims_evidence': 'experiment.remote.easyai.yaml hidden_dim=256; _hidden_dims; validate_v1_2_71_model_contract legacy concat-128',
            'reference_manifest_sha256': digest(path)}

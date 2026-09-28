# EcoTox-QSAR: aquatic and soil toxicity prediction

This package provides the processed observations, fixed evaluation splits,
row-level predictions, model and preprocessing objects, numerical result
tables, and core modeling source needed to inspect the principal results of
the aquatic and soil ecotoxicity study. The [online predictor](https://ecotox-qsarpredictor.cn/)
is an interactive demonstration; it is not the download or training interface
for this research package.

## Citation and archive

Version 1.0.2 adds the upstream cleaning scripts and a required structure/split
helper; it does not change the supplied observations, predictions, or metrics.
The version-specific Zenodo DOI will be added after the new source snapshot is
published. The [v1.0.1 DOI](https://doi.org/10.5281/zenodo.22981474)
archives the prior source and does not contain these added scripts. Zenodo
archives source and numerical tables, but **not** the nine large data,
prediction, and model-object ZIPs. Download those from the
[v1.0.2 GitHub Release](https://github.com/cyly2003/ecotox-qsar-reproducibility/releases/tag/v1.0.2)
and verify them against `ASSETS.json`. Cite the accompanying study separately.

## Evaluation scope

W00 (aquatic) and M00 (soil) are separate prediction routes. Their target
scales are `ptox_mol_l` and `neg_log10_mol_kg`, respectively; values must not
be pooled or compared as if they had one common physical unit. S1/S2 are the
principal missing-condition and missing-combination evaluation boundaries.
S3 (parent-disjoint) and S4 (ring-scaffold-disjoint) are extrapolation pressure
tests. Four frozen seeds were used where the result is labeled a four-seed
ensemble. Unsupported outcomes remain missing (`NA`), not zero.

The `results/` directory contains final numerical tables for the core
evaluation and the E10, E11, E20, E61, E72, and E90 supporting analyses.
These tables preserve their original denominators and statistical scope;
descriptive comparisons must not be interpreted as new confirmatory tests.
No draft plots, candidate figure exports, internal queue records, or server
deployment materials are included.

## Contents

| Location | Contents |
| --- | --- |
| `code/` | Model, upstream cleaning, preprocessing, split, and training implementations |
| `code/cleaning/` | Four upstream ECOTOX cleaning scripts |
| `code/scripts/build_scaffold_cluster_splits.py` | Structure normalization and scaffold-split helper required by `build_data.py` |
| `results/` | Numerical result tables, without figure exports |
| `FILES_SHA256.csv` | Checksums for the 116 code and result files |
| `ASSETS.json` and `ASSET_FILES_SHA256.csv` | Nine downloadable asset ZIPs and their 793 member checksums |
| `verify_release.py` | Integrity and core pooled-metric replay |

The ZIPs are distributed as release assets, not committed to Git. Download
all nine into a local `assets/` directory before verification. They provide
processed/model-ready data and fixed physical splits, core single-seed and
ensemble row predictions, traditional-model and representation-probe row
predictions, input-ablation predictions and draws, and the primary MTL model
weights with their preprocessing objects and manifests. The earlier website
model snapshot is not part of this package.

## Data preparation scope

The four `code/cleaning/` files expose the upstream ECOTOX cleaning logic.
Only two machine-specific default input paths in
`build_clean_ecotox_sqlite.py` were replaced with relative `inputs/` paths;
the cleaning transformations were not changed. The structure normalization
and scaffold helper is supplied under `code/scripts/` so the public
`revision_pipeline.build_data` import resolves. These scripts do not turn the
release into a standalone raw-data rebuild: the original ECOTOX SQLite source
and intermediate local SQLite snapshots are not distributed here. The
processed observations and fixed splits are instead supplied in release
assets. Inspect the manuscript and Supporting Information for source version,
filters, split definitions, and interpretation of the resulting tables.

To avoid repeating model-input tables in every prediction file, 608 prediction
Parquet files contain an exact-value column projection. The retained fields
cover row identity, route, task, fixed split, target scale, observed target,
prediction, prediction status, and any available seed, cohort, censoring, or
scoring flags. The full model-ready inputs remain in the processed-data and
physical-split assets. `ASSET_FILES_SHA256.csv` records both the original
source SHA-256 and the public projected-file SHA-256. Projection was checked
by an exact typed Parquet round trip; numerical values were not rounded.

## Quick verification

Use Python 3.12 with `numpy`, `pandas`, and `pyarrow`, then run from this
directory:

```bash
python verify_release.py --assets-dir assets --deep
```

The script checks every staged code/result SHA-256, each ZIP and member hash,
the 32 supplied model/preprocessing objects against their source identities,
all 28 core ensembles against their four frozen seed predictions, and pooled
MAE, RMSE, and R2 for all 28 full-cohort core arms from saved fixed-test rows.
It does not retrain a model or create new data splits. The result is a
prediction-level four-seed ensemble metric, not a mean of four seed-level
scores.

## Reproducibility boundary

The source and frozen data permit inspection and attempted retraining, while
the saved predictions provide the fastest numerical check. A fresh-machine
raw-ECOTOX-to-final rebuild and portable checkpoint inference for every arm
have **not** been verified. The verification command above directly checks
the core pooled metrics; the other result families are supplied as source-
locked tables and row artifacts but are not all replayed by that command.
Do not describe this package as a one-command reconstruction of every figure
or analysis. Consult the manuscript and Supporting Information for the
definition and interpretation of each experiment.

The observation source is the [U.S. EPA ECOTOX Knowledgebase](https://www.epa.gov/ecotox).
The MIT license
applies to author-created code only; it does not relicense ECOTOX records,
third-party dependencies, or external data. Cite both the study and the
ECOTOX source when reusing the processed observations.

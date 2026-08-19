# Analysis

Exploratory analysis, one-off investigations, and the tools that produced them.
Everything here reads from the warehouse or from the built training table;
nothing here writes to production.

## The two notebooks

| notebook | what it answers |
|---|---|
| `eda.ipynb` | What predicts viewership, and how much? Variance budget, category, timing, duration, cadence, segmentation, data mining. |
| `thumbnail_eda.ipynb` | Do thumbnail properties add anything, on 30,554 measured images? |

Both are **generated**, not hand-edited — run `build_eda_notebook.py` or
`build_thumbnail_notebook.py` to rebuild against a newer dataset. Edit the
generator, never the `.ipynb`, or your changes are lost on the next rebuild.

Open them with the **ViewCastLK (venv)** kernel. The default `python3` kernel
points at the system Python, which has none of the libraries and fails at the
first `read_parquet` with a misleading "no engine" error. Register it once:

```bash
"../../Project Code/venv/Scripts/python.exe" -m ipykernel install --user --name viewcastlk --display-name "ViewCastLK (venv)"
```

### One methodological rule

Channel identity explains ~64% of the variance in log day-7 views. Any question
about a *video* attribute is therefore asked on **within-channel residuals** —
`log1p(views)` minus that channel's own mean. Comparing raw group means measures
which channels post what, not the effect of the attribute. The category section
shows the ranking inverting once this is done.

Model comparisons are **paired across eight splits**. A single grouped split on
this data swings by ±0.1 R² and will answer whichever way the seed falls; an
earlier draft reported a single-split figure and was wrong because of it.

## Modelling

`MODEL_FINDINGS.md` is the consolidated result: what the model achieves, in
which regime, against which baseline, with the feature set and the questions
that were tested and settled. Read that before re-running anything.

| script | purpose |
|---|---|
| `train_baseline.py` | R^2/MAE per horizon, cold and warm, with the channel-history trap measured |
| `evaluate_srs_metrics.py` | MAPE and the metrics SRS v1.1 names, against the naive category-average baseline |
| `test_target_binning.py` | Does predicting view bands beat regressing? |
| `compare_encodings.py` | Full correlation table; ordinal vs native vs one-hot |

Every figure is a mean over five splits. A single grouped split swings by
±0.1 R^2 on this data, so single-split numbers are not reportable.

## Tools

| script | purpose |
|---|---|
| `paths.py` | Finds the training table. Use it rather than a relative path — see below. |
| `dataset_stats.py` | Full profile of a build: coverage, targets, segments, quality flags. |
| `compare_datasets.py` | Diff two builds — what grew, what moved, what a teammate must re-run. |
| `fetch_thumbnails.py` | Downloads thumbnails and reduces each to measurements. Resumable, no API quota. |
| `verify_archives.py` | Checks archived Parquet against the manifests and against Postgres. |
| `quota_model.py`, `quota_breakdown.py` | Where the daily API budget goes. |
| `test_rss_discovery.py`, `test_rss_at_scale.py` | Evidence behind RSS-first discovery. |
| `sweep_last_upload.py`, `find_new_channels.py`, `check_candidates_live.py` | Roster expansion and dead-channel detection. |
| `label_availability.py`, `prune_plan.py` | Label coverage and retention planning. |
| `build_sad_*.py`, `build_team_brief.py`, `edit_srs_intervals.py`, `sad_extract.py` | Document generation. |

## Where the dataset lives

**Outside this repository**, at `DSEP/Dataset/`. The CSV is 70 MB and the
Parquet is a build artefact of `scripts/build_training_table.py`, regenerated as
collection continues — neither belongs in git.

Always resolve it with `paths.dataset_path()`. A bare `../Dataset` from here
points at the repository's *own* `Dataset/` folder, which holds title scores and
not the training table: a real directory, wrong contents, confusing error.
Override with `VIEWCASTLK_DATASET` if you keep it elsewhere.

## What is deliberately not committed

See `.gitignore`. Excluded: dataset copies, restore-test archives (already on
Drive), the pre-migration safety snapshot, SAD figure extractions, and
intermediate channel-discovery scratch. Everything ignored is either regenerable
by a script here or tracked elsewhere — a clone can reproduce it.

`thumbnail_features.parquet` **is** committed at 2.9 MB, because regenerating it
costs 26 minutes and a 450 MB download, and modelling depends on it.

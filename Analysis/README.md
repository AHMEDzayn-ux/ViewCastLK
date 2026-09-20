# Analysis

Everything we learned from the data, and the tools that produced it. Nothing
here writes to production; it all reads the warehouse or the built training
table.

## Start here: `viewership_eda/`

**[`viewership_eda/viewership_eda.html`](viewership_eda/viewership_eda.html)**
is the main exploratory analysis, readable in any browser without running
anything. The same notebook with its code is `viewership_eda.ipynb`.

It takes every input the forecast could use, one at a time, and shows how
day-7 views change with it, using only counts, medians and percentages. Every
chart has a plain explanation under it. It covers:

1. the data, its quality, and the target at 7, 14, 21 and 30 days
2. every input: channel, video format, language, timing, title, tags and description
3. each input compared both with all videos and with the video's own channel
4. a category-by-category section, including a check of how much News & Politics
   pulls the overall results
5. growth after the first week, thin data, change over time, viral videos
6. what the forecast API actually receives, and a keep or drop verdict for every input

| file | what it is |
|---|---|
| `viewership_eda.ipynb` / `.html` | the notebook, with and without code |
| `build_notebook.py` | generates and runs the notebook: `python Analysis/viewership_eda/build_notebook.py` |
| `helpers.py` | every chart and number in the notebook comes from here |
| `notes.py` | the explanation text, written against the 18 September 2026 build |
| `figures/` | every chart as a PNG, for slides and reports |

## Deeper analyses: `deep_dives/`

More technical work that the main notebook builds on. Each folder holds its
script, its outputs and, where there is one, its notebook.

| folder | question |
|---|---|
| `within_channel_eda/` | The original, more statistical EDA: variance budget, within-channel residuals, segment reversals, cold and warm start. |
| `feature_study/` | Which inputs the model should keep, drop, fix or add, measured with the team's own model. Start with `FEATURE_RECOMMENDATIONS.md`. |
| `descriptive_stats/` | Plain views-by-category, size and format charts, plus summary numbers. |
| `view_growth/` | How views accumulate from day 1 to day 30. |
| `label_quality/` | How close each label is to its exact day, and which channel statistics were recorded late. |
| `data_drift/` | Whether the data changes over the collection period. |
| `channel_guidelines/` | Which creator advice holds for individual channels, not only on average. |
| `thumbnails/` | Whether thumbnail properties add anything, on 30,554 measured images. |

## Tools: `tools/`

| folder | what is in it |
|---|---|
| `dataset/` | `dataset_stats.py` (profile a build), `compare_datasets.py` (diff two builds), `label_availability.py`, `verify_archives.py` (check archived Parquet against Postgres) |
| `collection/` | API quota models, RSS discovery tests, roster expansion and dead-channel checks, and the channel lists they produced |
| `documents/` | Generators for the SAD, SRS and team brief documents |

## Generated notebooks

The notebooks are **generated**. Edit the builder next to them, never the
`.ipynb`, or your changes are lost on the next rebuild.

Open them with the **ViewCastLK (venv)** kernel. The default `python3` kernel
points at the system Python, which has none of the libraries and fails at the
first `read_parquet` with a misleading "no engine" error. Register it once:

```bash
"../../Project Code/venv/Scripts/python.exe" -m ipykernel install --user --name viewcastlk --display-name "ViewCastLK (venv)"
```

## Where the dataset lives

**Outside this repository**, at `DSEP/Dataset/`. The CSV is over 100 MB and
the Parquet is a build artefact of `scripts/build_training_table.py`,
regenerated as collection continues, so neither belongs in git.

Always resolve it with `paths.dataset_path()`. A bare `../Dataset` points at
the repository's *own* `Dataset/` folder, which holds title scores and not the
training table. Override with `VIEWCASTLK_DATASET` if you keep it elsewhere.

`paths.py` also names the other fixed locations (`REPO`, `SCRIPTS`, `DSEP`,
`DELIVERABLES`), so a script in any subfolder finds what it needs without
counting `..`. Scripts find `paths.py` by walking up from their own folder.

## What is deliberately not committed

See `.gitignore`. Excluded: dataset copies, restore-test archives (already on
Drive), the pre-migration safety snapshot, SAD figure extractions, and
intermediate channel-discovery scratch. Everything ignored is either
regenerable by a script here or tracked elsewhere.

`deep_dives/thumbnails/thumbnail_features.parquet` **is** committed at 2.9 MB,
because regenerating it costs 26 minutes and a 450 MB download.

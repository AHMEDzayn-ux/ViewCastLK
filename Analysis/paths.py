"""Locate the training dataset from anywhere in this folder.

The dataset deliberately lives OUTSIDE the repository. The CSV is 70 MB, the
Parquet is a build artefact of scripts/build_training_table.py, and both are
regenerated as collection continues -- none of that belongs in git.

There is a trap this exists to avoid. The repository has its own Dataset/
folder (title scores), so once Analysis/ moved inside the repo a bare
"../Dataset" resolves to that one: a real directory, without the training
table in it. The failure would be a confusing "file not found" pointing at a
path that plainly exists. Resolution order below tries the external location
first and names every path it tried when it gives up.

Override with the VIEWCASTLK_DATASET environment variable.
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_NAME = "viewcastlk_training_table.parquet"


def dataset_path(name: str = DEFAULT_NAME) -> Path:
    env = os.environ.get("VIEWCASTLK_DATASET")
    candidates = []
    if env:
        p = Path(env)
        candidates.append(p if p.suffix else p / name)
    candidates += [
        HERE.parent.parent / "Dataset" / name,   # DSEP/Dataset -- the usual home
        HERE.parent / "Dataset" / name,          # in-repo, if ever placed there
        HERE / name,                             # alongside this folder
    ]
    for c in candidates:
        if c.is_file():
            return c
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"could not find {name}. Tried:\n  {tried}\n"
        f"Rebuild it with scripts/build_training_table.py, or set "
        f"VIEWCASTLK_DATASET to its location.")

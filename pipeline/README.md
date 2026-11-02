# agerating pipeline

Python code that builds the agerating databases from scratch:

1. collect Wikidata facts for every film and TV show,
2. add Wikipedia text,
3. fine-tune the classifier on titles with an official FSK rating,
4. score everything else,
5. write the full SQLite database,
6. derive the small ID → age database from it.

Afterwards, `update.py` adds the films and TV shows that appear on Wikidata, and `weekly.sh` runs it every week and
publishes both databases as a GitHub release.

## Setup

- Python 3.12 (with headers: the LFM3 base model compiles GPU kernels on first use)
- a GPU for training and batch scoring. Releases are trained and scored on an RTX 4090 (24 GB) with a Ryzen 5 7600X
  and 24 GB RAM. `train.py` also runs on Apple silicon (MPS): one fold took 95 min on an M5 Max, against about 35 min
  on the 4090. `update.py` also runs on CPU, see [Weekly updates](#weekly-updates).

```sh
cd pipeline
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
export AGERATING_USER_AGENT="agerating-pipeline/1.0 (https://example.org/you; you@example.org)"   # Wikimedia asks for contact details
```

| Variable | Default | Meaning |
|---|---|---|
| `AGERATING_USER_AGENT` | `agerating-pipeline/1.0 (python-urllib)` | User-Agent for Wikidata and Wikipedia requests. Put your contact in it. |
| `AGERATING_DATA` | `pipeline/data` | Where collected data, predictions and the databases go. |

## Steps

Timings are from the 1 November 2026 rebuild: steps 1-8 on a Mac Studio (M5 Max), steps 9-12 on the RTX 4090. The
download steps depend mostly on Wikimedia rate limits; the first build in October needed about 40 min for each of
steps 6 and 2 hours for step 8. Every collection step is resumable: rerun the same command after an interruption.

| # | Command | Output | Time |
|---|---|---|---|
| 1 | `python wikidata.py items` | `data/wd/items.parquet`: 598k universe items and the 19.8k FSK-rated items | minutes |
| 2 | `python wikidata.py stmts` | `data/wd/stmts_*.parquet`, `meta_*.parquet`: 14.7M truthy statements, 250 items per query | ~1 h |
| 3 | `python wikidata.py labels` | `data/wd/labels.parquet`: labels of linked items (people, genres, studios, ...) | ~10 min |
| 4 | `python wikidata.py ids` | `data/wd/ids.parquet`, `id_properties.parquet`: external IDs of the universe items | < 1 min |
| 5 | `python wikidata.py titles` | `data/wd/item_labels.parquet`: any-language titles for items without an English one | ~1 min |
| 6 | `python wikipedia.py api en --training`<br>`python wikipedia.py api de --training` | API text for the FSK-rated items (training input) | 5 + 5 min |
| 7 | `python wikipedia.py dump en`<br>`python wikipedia.py dump de` | dump text (20231101) for the remaining items; downloads 11 + 5 GB | 3 + 2 min |
| 8 | `python wikipedia.py api en`<br>`python wikipedia.py api de` | API text for articles missing from the dump | 11 + 2 min |
| 9 | `python train.py --labels FILE` | `models/agerating-lfm3-1.7b-v2/` (`config.json`, `model.safetensors`) | 41 min |
| 10 | `python score.py` | `data/serial_univ.parquet` (model inputs), `data/pred/agerating-lfm3-1.7b-v2.parquet` | 79 min |
| 11 | `python build_db.py` | `data/agerating.db` (588 MB) | 1 min |
| 12 | `python build_ids_db.py` | `data/agerating-ids.db` (10 MB) | seconds |
| 13 | `python pack_ids.py --check` | `data/agerating-ids.bin` (2 MB), the small database's packed list | seconds |

`python wikidata.py all` runs steps 1-5. The Wikipedia API steps are slow because Wikipedia rate-limits bulk
fetching; the code honours `Retry-After`.

To evaluate instead of training the final model, run `python train.py --fold 0` (5-fold split, about 35 min per fold).
`--split group` keeps titles of the same series or director in one fold. This writes `results/eval.csv` and
out-of-fold probabilities to `results/oof/`, for the full input and with the Wikipedia line removed.

`--labels FILE` adds training labels for titles without an official FSK rating: a parquet or CSV file with the
columns `qid` and `age` (0, 6, 12, 16 or 18). Official ratings win where both exist. With `--fold`, these titles are
split into their own five folds and scored separately (`extra_*` columns in `results/eval.csv`). The released v2
model was trained with a file of 1,381 TV ratings that is not published, see the
[model card](../model/README.md#training).

To use the released model instead of training (step 9), put its files (`config.json`, `model.safetensors`) in
`models/agerating-lfm3-1.7b-v2/`, or pass `--model DIR` to `score.py` and `update.py`.

## Modules

| File | Role |
|---|---|
| `common.py` | paths, Wikidata property and class lists, the age buckets, the decision threshold, metrics |
| `wikidata.py` | batch collection through the Wikidata Query Service; `universe()` lists the film and TV items |
| `wikipedia.py` | lead and plot text from the MediaWiki API and the Hugging Face dump, `load_texts()` |
| `data.py` | loaders for the collected data; official ratings, title kind, CV groups |
| `serialize.py` | facts + text → model input. One function for training, batch scoring and new titles |
| `model.py` | the classifier (LFM3-1.7B-Base + LoRA + mean-pool head), `predict`, `save`, `load` |
| `train.py` | fine-tuning and cross-validation |
| `score.py` | scores every universe title without an official rating |
| `build_db.py` | the full database, see the schema in its docstring |
| `build_ids_db.py` | the small database: one `(id, age)` table per ID type |
| `pack_ids.py` | the small database's lookups as a packed list (layout in its docstring), with a reference reader |
| `wikidata_live.py` | per-item Wikidata API access for `update.py`, in the same format as the batch collection |
| `update.py` | adds titles that appeared on Wikidata since the last build or update |
| `weekly.sh` | the weekly job: `update.py`, `build_ids_db.py`, `pack_ids.py`, GitHub release |

## Weekly updates

```sh
python update.py --db data/agerating.db     # --device cpu|mps|cuda, --minutes M (time budget), --limit N
```

It loads the model named in the database's `info` table from `models/<name>/`, or from `--model DIR`.

`update.py`:

1. lists the film and TV items on the Wikidata Query Service (16 queries, about 30 s) and keeps the ones missing from
   the database,
2. per chunk of 100 new items, fetches the entities from the Wikidata API, labels of linked items the database does
   not have yet, and the English and German Wikipedia articles,
3. scores the items without an official FSK rating with the same serializer and model as the batch, and inserts them
   with all their external IDs (`source` = `update`).

The per-item input matches the batch: on 200 random titles fetched from the Wikidata API, 199 inputs were identical to
the batch input (the other differed by an invisible character in a title) and all 200 estimates matched. Chunks that
fail (rate limits, network errors) are left for the next run, and the script exits with status 1. Existing titles are
not refreshed: a new official FSK rating for an estimated title, new IDs on existing titles, and deletions or class
changes on Wikidata only arrive with a full rebuild.

Wikidata gained 306 new films and TV shows in the 2.6 days after the first snapshot, so a week is about 800.
Measured on those 306 titles:

| Machine | Time |
|---|---|
| Mac Studio, M5 Max (MPS) | 0.9 min, including the 17 s listing |
| MacBook Pro, M5 Pro (MPS) | 1.1 min |
| MacBook Pro, M5 Pro (CPU, 4 threads) | 13.3 min |

`weekly.sh` runs the update and, if the database changed since the last published release, rebuilds the small
database and its packed list (checked against the database), compresses both databases with `zstd -19` and the packed
list with `gzip -9 -n`, writes `SHA256SUMS` and creates the release `vYYYY.MM.DD` with `gh`. The
database's version is `updated_at` from its `info` table (set when `update.py` adds titles), or `built_at` for a fresh
build. The script records the published version in `data/published` only after the release exists, so a failed
upload is retried on the next run. When `data/agerating.db` is missing it starts from the latest release.

| Variable | Default | Meaning |
|---|---|---|
| `AGERATING_REPO` | `edde746/agerating` | repository for the releases (`gh` must be logged in with write access) |
| `AGERATING_MINUTES` | `240` | time budget for `update.py` |
| `PYTHON` | `.venv/bin/python` | |

### On studio.local

The job runs on studio.local from `~/agerating/pipeline` every Monday at 04:00 local time, through the launchd agent
`~/Library/LaunchAgents/app.agerating.weekly.plist`. The agent sets `PATH` (Homebrew first, for `gh` and `zstd`),
`LANG`, `AGERATING_REPO` and `AGERATING_USER_AGENT`, and logs to `~/Library/Logs/agerating.log`.

```sh
launchctl kickstart gui/$(id -u)/app.agerating.weekly    # run now
launchctl print gui/$(id -u)/app.agerating.weekly        # state, last exit code
tail -f ~/Library/Logs/agerating.log
```

The working database is `~/agerating/pipeline/data/agerating.db`, the model `models/agerating-lfm3-1.7b-v2/`.

## Rebuilding everything

Wikidata changes daily. A full rebuild picks up what the weekly update skips (see above): delete `data/wd/` and
`data/text/`, rerun steps 1-8 and 10-12, and replace `data/agerating.db`. Retraining (step 9) is only needed to pick
up new official ratings as training data. The next `weekly.sh` run publishes the rebuilt databases.

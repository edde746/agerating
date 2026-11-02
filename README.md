# agerating

Suggested minimum ages for about 600,000 films and TV shows on Wikidata, published every week as SQLite databases.
About 20,000 of the ages are official FSK ratings from Wikidata; the rest are estimates from a fine-tuned language
model.

Ages are on the German FSK scale: 0, 6, 12, 16, 18. **Estimates are machine predictions, not ratings.** Never present
them as FSK or other official ratings; the [license](LICENSE) requires this.

## Downloads

Each [release](https://github.com/edde746/agerating/releases) has two databases, compressed with zstd, the small one's
lookups as a gzipped packed list and a gzip copy of it for platforms without a zstd decoder, and their checksums in
`SHA256SUMS`:

| File | Contents | Size |
|---|---|---|
| `agerating-ids.db.zst` | small: IMDb, TMDB and TheTVDB ID → age | 2.3 MB (10 MB unpacked) |
| `agerating-ids.bin.gz` | the small database's lookups as a [packed list](#the-packed-list), gzip | 0.8 MB (2 MB unpacked) |
| `agerating-ids.db.gz` | small, gzip | 5.3 MB (10 MB unpacked) |
| `agerating.db.zst` | full: every title with its external IDs, official rating, estimate and model probabilities | 164 MB (567 MB unpacked) |

```sh
gh release download --repo edde746/agerating --pattern agerating-ids.db.zst
zstd -d agerating-ids.db.zst
sqlite3 agerating-ids.db "SELECT age FROM imdb WHERE id = 133093"   # The Matrix (tt0133093) -> 16
```

### The small database

One table per ID type, each `(id INTEGER PRIMARY KEY, age INTEGER NOT NULL)`:

| Table | ID | Example |
|---|---|---|
| `imdb` | IMDb title ID as a number, without `tt` and leading zeros | `tt0133093` → `133093` |
| `tmdb_movie` | TMDB movie ID | `603` |
| `tmdb_tv` | TMDB TV series ID | `1399` |
| `tvdb_series` | TheTVDB series ID | `121361` |
| `tvdb_movie` | TheTVDB movie ID | |

TMDB and TheTVDB number movies and series separately, so look IDs up in the matching table. `age` is the official
FSK rating where Wikidata has one, otherwise the estimate. When an ID belongs to several Wikidata items, an official
rating wins, otherwise the highest estimate. IDs that are not plain numbers on Wikidata (URLs, slugs) are left out.

### The packed list

`agerating-ids.bin.gz` holds the same IDs and ages as the small database in a form an app can read with nothing but
a gzip decoder. Integers are little-endian; a varint is LEB128 (7 bits per byte, low bits first, the high bit set on
every byte but the last).

| Bytes | Contents |
|---|---|
| 4 | `AGID` |
| 1 | format version, `1` |
| 1 | number of tables, then per table: |
| 1 + name | the name's length and the table's name in ASCII, as in the small database (`imdb`, `tmdb_movie`, ...) |
| 4 | the number of rows |
| one varint per row | the IDs in ascending order, each as its gap from the previous ID (the first from 0) |
| one byte per row | the ages, in the same order (0, 6, 12, 16 or 18) |

`pipeline/pack_ids.py` writes it and has a reference reader (`read`).

### The full database

| Table | Contents |
|---|---|
| `title` | one row per Wikidata item: `qid`, `label`, `year`, `kind` (film, tv, other), `fsk_official`, `fsk_estimate`, probabilities `p0` ... `p18`, `confidence`, `had_article`, `source` (official, batch, update), `model`, `scored_at` |
| `external_id` | `pid`, `value`, `qid`: every external ID Wikidata has for the title (IMDb is `P345`), 6.8 million in total |
| `id_property` | names of the external-ID properties |
| `label` | labels of linked items (people, genres, studios), used to build the model input for new titles |
| `info` | model name, decision threshold, build and update times |

```sql
SELECT t.label, t.year, coalesce(t.fsk_official, t.fsk_estimate) AS age, t.confidence
FROM external_id e JOIN title t ON t.qid = e.qid
WHERE e.pid = 'P345' AND e.value = 'tt15239678';   -- Dune: Part Two, 2024, 12, NULL (official)
```

## Weekly updates

Every Monday a scheduled job lists the films and TV shows on Wikidata, scores the ones that are not in the database
yet with the same model, and publishes a new release if it added any. About 800 titles appear on Wikidata each
week. The job does not refresh titles already in the database: an estimate stays even after Wikidata gets an
official FSK rating for the title, and titles that are deleted or reclassified on Wikidata stay. A full rebuild
picks those changes up. Details are in [pipeline/README.md](pipeline/README.md#weekly-updates).

## The current data

The databases were rebuilt from a Wikidata snapshot of 1 November 2026 with the v2 model: every item whose "instance
of" is one of 16 film and TV classes (film, short film, animated film, television series, miniseries, anime, web
series, ...).

| | Films | TV | Other | Total |
|---|---|---|---|---|
| Titles | 431,225 | 165,695 | 592 | 597,512 |
| Official FSK rating | 19,590 | 59 | 3 | 19,652 |
| Estimate | 411,635 | 165,636 | 589 | 577,860 |
| With an IMDb ID | 334,752 | 74,136 | 239 | 409,127 |
| With a TMDB ID (movie or TV) | 280,742 | 65,432 | 210 | 346,384 |
| With a TheTVDB ID (series or movie) | 200,938 | 52,453 | 172 | 253,563 |

| Age | 0 | 6 | 12 | 16 | 18 |
|---|---|---|---|---|---|
| Official | 1,741 | 3,134 | 7,935 | 5,516 | 1,326 |
| Estimated | 41,788 | 142,173 | 251,221 | 132,281 | 10,397 |

Titles without an English label fall back to the English or German Wikipedia title, then to a label in another
language.

## How the estimates are made

1. **Facts:** each title's Wikidata facts are written out as text, one line per fact: genres, country, director,
   cast, studio, network, series, awards, duration and so on. Ratings from other countries' boards are added
   (MPA PG-13, Kijkwijzer 12, ...), but never the FSK rating itself.
2. **Wikipedia:** when the title has an English or German Wikipedia article, its plot section or lead is appended.
   36.8% of the estimated titles have one.
3. **Model:** [LFM3-1.7B-Base](https://huggingface.co/LiquidAI/LFM3-1.7B-Base) with LoRA adapters and a classification
   head, fine-tuned on the 19,713 Wikidata titles that have an official FSK rating and on 1,381 TV shows whose FSK
   rating was looked up on German retail and streaming sites.
4. **Decision:** the estimate is the smallest age whose cumulative probability reaches 60%, so the model rounds up
   when it is unsure between two ages.

Titles with an official FSK rating keep it, and have no estimate.

## How good the estimates are

On held-out titles (fold 0 of a 5-fold split), with the database's decision rule:

| Held-out titles | Number | Exact | Within one step | Two or more steps too low |
|---|---|---|---|---|
| Films with an official FSK rating, with Wikipedia text | 3,938 | 63.5% | 97.3% | 0.51% |
| the same films, Wikipedia text removed | 3,938 | 61.4% | 96.4% | 0.71% |
| TV shows with an official FSK rating | 59 | 57.6% | 91.5% | 1.69% |
| TV shows with a looked-up FSK rating | 277 | 63.9% | 98.6% | 1.08% |

- **One step** is one position on the scale 0 → 6 → 12 → 16 → 18.
- **v2 against v1:** the looked-up TV ratings raised TV accuracy from 51% to 64% exact on held-out shows and halved
  the suggestions that are too low (Game of Thrones, Shōgun and Succession now get 16 instead of 12). Films are
  unchanged on the test set. Retraining on the new snapshot moved about a quarter of the estimates for films
  without a Wikipedia article one step younger, whether or not the TV ratings were used; on held-out films with
  similarly sparse data, the models' suggestions were older than the official ratings on average.
- **Label disagreement:** rating boards often disagree by a step. FSK and the MPA give the same age to only 49.8% of
  the films they both rated.
- **Obscure titles:** titles with few Wikidata facts and no article are harder than the test set. New titles often
  start on Wikidata with only a few facts.

Details are in the [model card](model/README.md).

## What is here

| Path | Contents |
|---|---|
| [`pipeline/`](pipeline/) | Python: Wikidata and Wikipedia collection, training, scoring, both database builds, the weekly update and release script |
| [`model/`](model/) | model card, `config.json` and license files for the Hugging Face model repository [`edde746/agerating-lfm3-1.7b-v2`](https://huggingface.co/edde746/agerating-lfm3-1.7b-v2), which also holds `model.safetensors` (66 MB) |

[pipeline/README.md](pipeline/README.md) lists every step of a full rebuild, from the Wikidata collection to the
databases. The 1 November 2026 rebuild took about 3.5 hours: 1.5 hours of Wikidata and Wikipedia downloads on a Mac
Studio, then 41 minutes of training, 79 minutes of scoring and a minute for the database on one RTX 4090.

## License

The code, model weights and databases are under the [agerating License 1.1](LICENSE):

- **Free use:** free and open-source projects may use them without asking, as long as the use is not commercial.
- **Needs written approval:** every commercial use, including commercial use by or in a free and open-source
  project, and every use in software that is not free and open source. Contact edde746 via
  [GitHub](https://github.com/edde746).

The model weights are also subject to the LFM Open License v1.0 of the base model. The Wikidata facts are CC0 and
can be taken from Wikidata directly. See [NOTICE](NOTICE) for all third-party terms.

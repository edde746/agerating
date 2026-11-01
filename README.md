# agerating

Suggested minimum ages for about 600,000 films and TV shows on Wikidata, published every week as SQLite databases.
About 20,000 of the ages are official FSK ratings from Wikidata; the rest are estimates from a fine-tuned language
model.

Ages are on the German FSK scale: 0, 6, 12, 16, 18. **Estimates are machine predictions, not ratings.** Show them as
suggestions ("Suggested 12+"), never as an FSK or other official rating; the [license](LICENSE) requires this.

## Downloads

Each [release](https://github.com/edde746/agerating/releases) has two databases, compressed with zstd, and their
checksums in `SHA256SUMS`:

| File | Contents | Size |
|---|---|---|
| `agerating-ids.db.zst` | small: IMDb, TMDB and TheTVDB ID → age | 2.3 MB (10 MB unpacked) |
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

## The first release

The first release is a Wikidata snapshot from 29-30 October 2026: every item whose "instance of" is one of 16 film
and TV classes (film, short film, animated film, television series, miniseries, anime, web series, ...).

| | Films | TV | Other | Total |
|---|---|---|---|---|
| Titles | 431,087 | 165,657 | 593 | 597,337 |
| Official FSK rating | 19,578 | 59 | 3 | 19,640 |
| Estimate | 411,509 | 165,598 | 590 | 577,697 |
| With an IMDb ID | 334,648 | 74,064 | 240 | 408,952 |
| With a TMDB ID (movie or TV) | 280,622 | 65,332 | 211 | 346,165 |
| With a TheTVDB ID (series or movie) | 200,926 | 52,432 | 172 | 253,530 |

| Age | 0 | 6 | 12 | 16 | 18 |
|---|---|---|---|---|---|
| Official | 1,741 | 3,133 | 7,932 | 5,509 | 1,325 |
| Estimated | 26,579 | 176,821 | 227,647 | 133,975 | 12,675 |

Titles without an English label fall back to the English or German Wikipedia title, then to a label in another
language.

## How the estimates are made

1. **Facts:** each title's Wikidata facts are written out as text, one line per fact: genres, country, director,
   cast, studio, network, series, awards, duration and so on. Ratings from other countries' boards are added
   (MPA PG-13, Kijkwijzer 12, ...), but never the FSK rating itself.
2. **Wikipedia:** when the title has an English or German Wikipedia article, its plot section or lead is appended.
   37% of the estimated titles in the first release had one.
3. **Model:** [LFM3-1.7B-Base](https://huggingface.co/LiquidAI/LFM3-1.7B-Base) with LoRA adapters and a classification
   head, fine-tuned on the 19,701 Wikidata titles that have an official FSK rating.
4. **Decision:** the estimate is the smallest age whose cumulative probability reaches 60%, so the model rounds up
   when it is unsure between two ages.

Titles with an official FSK rating keep it, and have no estimate.

## How good the estimates are

On held-out titles with an official FSK rating (fold 0 of a 5-fold split, 3,941 titles):

| | Exact | Within one step | Two or more steps too low |
|---|---|---|---|
| With Wikipedia text | 63.4% | 97.0% | 0.61% |
| Without Wikipedia text | 61.1% | 96.4% | 0.63% |

- **One step** is one position on the scale 0 → 6 → 12 → 16 → 18.
- **Label disagreement:** rating boards often disagree by a step. FSK and the MPA give the same age to only 49.8% of
  the films they both rated.
- **TV:** estimates are the weakest part, because only 93 TV shows have an FSK rating to learn from. Some adult
  dramas get 12, for example Game of Thrones, Shōgun and Succession.
- **Obscure titles:** titles with few Wikidata facts and no article are harder than the test set. New titles often
  start on Wikidata with only a few facts.

Details are in the [model card](model/README.md).

## What is here

| Path | Contents |
|---|---|
| [`pipeline/`](pipeline/) | Python: Wikidata and Wikipedia collection, training, scoring, both database builds, the weekly update and release script |
| [`model/`](model/) | model card, `config.json` and license files for the Hugging Face model repository `edde746/agerating-lfm3-1.7b-v1`, which also holds `model.safetensors` (66 MB) |

[pipeline/README.md](pipeline/README.md) lists every step of a full rebuild, from the Wikidata collection to the
databases. The steps add up to about 7 hours on one RTX 4090, most of it rate-limited Wikidata and Wikipedia
downloads and scoring.

## License

The code, model weights and databases are under the [agerating License 1.0](LICENSE):

- **Free use:** free and open-source projects may use them without asking, as long as the use is not commercial.
- **Needs written approval:** every commercial use, including commercial use by or in a free and open-source
  project, and every use in software that is not free and open source. Contact edde746 via
  [GitHub](https://github.com/edde746).

The model weights are also subject to the LFM Open License v1.0 of the base model. The Wikidata facts are CC0 and
can be taken from Wikidata directly. See [NOTICE](NOTICE) for all third-party terms.

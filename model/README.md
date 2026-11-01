---
license: other
license_name: agerating-license-1.0
license_link: LICENSE
base_model: LiquidAI/LFM3-1.7B-Base
language:
  - en
  - de
pipeline_tag: text-classification
tags:
  - age-rating
  - movies
  - tv-shows
  - wikidata
  - lora
---

# agerating-lfm3-1.7b-v2

Suggests a minimum viewing age (0, 6, 12, 16 or 18, on the German FSK scale) for a film or TV show. The input is the
title's Wikidata facts and, when there is one, a passage from its Wikipedia article. The output is an **estimate**,
not an official rating.

This model produces the estimates in the agerating databases from the 1 November 2026 rebuild on, and scores the
titles added every week. The pipeline that collects the data, builds the input text, trains, scores and builds the
databases is in the [agerating repository](https://github.com/edde746/agerating).

v2 is v1 ([agerating-lfm3-1.7b-v1](https://huggingface.co/edde746/agerating-lfm3-1.7b-v1)) retrained on a newer
Wikidata snapshot with 1,381 extra TV ratings (see [Training](#training)). It is as good as v1's recipe on films and
better on TV (see [Evaluation](#evaluation)).

## Model

| | |
|---|---|
| Base model | [LiquidAI/LFM3-1.7B-Base](https://huggingface.co/LiquidAI/LFM3-1.7B-Base), revision `545fc8f4c7b5fa9f405105274cd29c4f8f44b526` |
| Adapters | LoRA on all linear layers: r 16, alpha 32, dropout 0.05 |
| Head | mean of the last hidden states over all input tokens, then one linear layer to 5 classes (ages 0, 6, 12, 16, 18) |
| Trainable parameters | 16,478,981 (`model.safetensors`, 66 MB); the base model is downloaded separately |
| Input | up to 768 tokens of serialized facts (format below) |
| Output | softmax probabilities over the five ages |

`config.json` records the base revision, LoRA settings, labels and training settings. The repository's
`pipeline/model.py` loads the files:

```python
import model as M                                   # agerating/pipeline/model.py
clf = M.load("path/to/agerating-lfm3-1.7b-v2")       # downloads the base model on first use
probs = M.predict(clf, [text])                      # array of shape (n, 5): ages 0, 6, 12, 16, 18
```

Build `text` with `pipeline/serialize.py`. The model only works well on that exact format.

## Input format

One line per fact, in a fixed order. Lists are in ascending Wikidata QID order and cut to a few values each. This is
the input for Dune: Part Two:

```
Title: Dune: Part Two (2024)
Description: 2024 film directed by Denis Villeneuve
Type: film
Genres: action film, adventure film, science fiction film, epic film, speculative fiction film
Country: Canada, Hungary, United States
Original language: English
Director: Denis Villeneuve
Screenwriter: Denis Villeneuve, Eric Roth, Jon Spaihts
Cast: Josh Brolin, Dave Bautista, Javier Bardem, Christopher Walken, Zendaya, Stellan Skarsgård, Léa Seydoux, Charlotte Rampling, Austin Butler, Souheila Yacoub
Production company: Legendary Pictures
Distributor: Warner Bros. Entertainment, Universal Pictures, InterCom
Part of series: Dune
Franchise: Dune
Based on: Dune
Main subject: messiah, rebellion, destiny, feud
Set in: Arrakis
Awards: Hugo Award for Best Dramatic Presentation, Long Form, Bradbury Award
Composer: Hans Zimmer
Duration: 165 min
Ratings elsewhere: LSF 13+; MPA PG-13; NMHH Category IV; Kijkwijzer 12; EIRIN G; CNC_FR no age restriction; CNC_FR warning; ICAA not recommended for children under 12; CNC_RO N - 15; IFCO 12A
Plot: Following the defeat of House Atreides by House Harkonnen, Princess Irulan—daughter of the Padishah Emperor, Shaddam Corrino IV—records in her journal [...]
```

- **Fields:** type, genres, country, original language, director, screenwriter, cast, production company,
  distributor, network, series, franchise, source material, main subject, setting, awards, nominations, composer,
  publication year and duration.
- **Ratings elsewhere:** ratings from other rating systems on Wikidata (34 film/TV systems). The FSK rating itself
  is never included.
- **Text:** the first Wikipedia passage with at least 200 characters, in this order: English plot section, English
  lead, German plot section. It is cut to 2,500 characters and left out when there is none.

## Training

- **Official ratings:** 19,713 Wikidata items with one unambiguous FSK rating (property P1981), collected
  1 November 2026.
  - Labels: 1,756 at 0, 3,143 at 6, 7,962 at 12, 5,524 at 16 and 1,328 at 18.
  - 19,602 films, 93 TV titles and 18 others.
- **Looked-up TV ratings:** 1,381 TV shows without an FSK rating on Wikidata, from the 1,800 most-linked ones (19 or
  more Wikipedia language editions).
  - Claude Haiku 5.5 searched the web for each show's FSK rating and quoted the page it read it on: German disc
    retailers (jpc.de, MediaMarkt, Saturn, videobuster.de, ...), fsk.de, German streaming listings (JustWatch,
    Netflix, Disney+, ...) and German Wikipedia.
  - Every cited page was downloaded and checked. 54 ratings whose page showed a different age, and 365 shows where
    nothing was found, were left out.
  - The same lookup on the 55 TV shows that do have an official FSK rating on Wikidata matched it exactly for 76% and
    within one step for 98%.
  - Labels: 155 at 0, 231 at 6, 456 at 12, 470 at 16 and 69 at 18. They are not published.
- **Wikipedia text:** fetched through the MediaWiki API in November 2026 for the rated items, from the 1 November 2023
  dump otherwise. 98% of the officially rated items and nearly all looked-up TV shows have an article.
  - Each training step drops the text with probability 0.5, so the same model handles titles with and without an
    article. Most unrated titles have no article.
- **Recipe:**
  - One epoch over all 21,094 items, batch size 8, bf16.
  - AdamW with weight decay 0.01. Learning rate 1e-4 for LoRA and 1e-3 for the head.
  - 5% linear warm-up, then linear decay.
- **Hardware:** 41.1 minutes on one RTX 4090, 5.4 GB peak GPU memory.

## Evaluation

Fold 0 of a stratified 5-fold split, trained on the other four folds with the same recipe. The test set also holds
all 59 TV shows with an official FSK rating, left out of training. A second run, the **baseline**, is the same except
without the looked-up TV ratings (v1's recipe on the same data), so the two are compared on the same titles.

Films and other titles with an official FSK rating (3,938):

| Input | Decision | Exact | Within one step | Too low | Too low by 2+ steps | Too high |
|---|---|---|---|---|---|---|
| with article text | most likely age | 64.7% | 97.1% | 17.7% | 0.96% | 17.6% |
| with article text | round up at 60% (database rule) | 63.5% | 97.3% | 13.9% | **0.51%** | 22.6% |
| article text removed | most likely age | 62.1% | 95.8% | 19.6% | 1.47% | 18.3% |
| article text removed | round up at 60% (database rule) | 61.4% | 96.4% | 15.4% | 0.71% | 23.1% |

The baseline gets 63.4% exact and 0.51% two or more steps too low with the database rule: no difference (paired 95%
interval for the exact difference: −0.9 to +1.1 points).

TV, database rule, with article text:

| Test titles | Number | Baseline exact | v2 exact | Baseline too low | v2 too low | Baseline too high | v2 too high |
|---|---|---|---|---|---|---|---|
| official FSK rating | 59 | 49.2% | 57.6% | 22.0% | 13.6% | 28.8% | 28.8% |
| looked-up rating, held out | 277 | 51.3% | **63.9%** | 34.3% | 15.9% | 14.4% | 20.2% |
| looked-up, disc/retail/fsk.de source | 205 | 53.7% | 63.9% | 31.7% | 13.7% | 14.6% | 22.4% |

- **Looked-up ratings:** +12.6 points exact (95% interval +6.1 to +19.1, McNemar p = 0.0003), and v2 halves the
  suggestions that are too low. The gain is similar on ratings from disc retailers and fsk.de (+10.2 points,
  p = 0.01), so it is not only learning streaming services' labels. v1 itself gets 51.3% on these titles.
- **Official TV ratings:** +8.5 points, but 59 titles are too few to be sure (95% interval −1.7 to +18.6 points).
- **Yardstick:** the looked-up ratings agree exactly with the official FSK rating for 76% of shows, so they are a
  noisier yardstick than official ratings.
- **One step** is one position on the scale 0 → 6 → 12 → 16 → 18.
- **Round up at 60%:** the database uses the smallest age whose cumulative probability reaches 0.6. When the model
  is unsure this picks the older age, which cuts suggestions that are two or more steps too low by about half.
- **Ceiling for films:** rating boards often disagree by a step. FSK and the MPA give the same age to only 49.8% of
  the films both rated (90.7% within one step).

## Intended use

Showing a suggested minimum age for films and TV shows that have no official rating, for example in a media library
app. Label it as a suggestion ("Suggested 12+"), never as an FSK or other official rating. The license requires this.

## Limitations

- **TV is less accurate than film.** TV estimates rest on 93 official and 1,381 looked-up ratings, mostly for
  well-known shows. Expect obscure TV shows to be less accurate than the tables above.
- **Obscure titles are harder.** The test items have a German rating, so they are better documented than a typical
  unrated title. 63% of the scored titles had no Wikipedia text, and some have only a handful of Wikidata facts.
  Expect lower accuracy there than in the table above.
- **Sparse titles are unstable across retrains.** Against v1, v2 moves about a quarter of the estimates for films
  without a Wikipedia article, almost always one step younger (on 20,000 sampled films: 24% younger, 4% older). A
  retrain on the same data without the TV ratings moves them the same way (28% younger), so this comes from
  retraining, not from the TV ratings. Held-out films with similarly sparse data had official ratings younger than
  either model suggested, on average.
- **German norms.** Labels follow the FSK. Other countries rate violence, sex and language differently.
- **Wikidata errors** (wrong genre, wrong linked film, vandalism) flow straight into the estimate.
- **Not a safety system.** Do not use it for legal compliance or age verification.

## License

- **agerating License 1.0** (`LICENSE`): free and open-source projects may use the model without asking, as long as
  the use is not commercial. Any commercial use, and any use in software that is not free and open source, needs
  written approval from the author.
- **LFM Open License v1.0** (`LICENSE-LFM`): applies too, because the weights are a derivative of LFM3-1.7B-Base. It
  does not license commercial use by organisations with USD 10 million or more in annual revenue.
- **Third-party notices:** in `NOTICE`, including the Wikidata (CC0) and Wikipedia (CC BY-SA) sources.

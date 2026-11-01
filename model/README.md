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

# agerating-lfm3-1.7b-v1

Suggests a minimum viewing age (0, 6, 12, 16 or 18, on the German FSK scale) for a film or TV show. The input is the
title's Wikidata facts and, when there is one, a passage from its Wikipedia article. The output is an **estimate**,
not an official rating.

This model produced the 577,697 estimates in the first agerating release and scores the titles added every week. The
pipeline that collects the data, builds the input text, trains, scores and builds the databases is in the
[agerating repository](https://github.com/edde746/agerating).

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
clf = M.load("path/to/agerating-lfm3-1.7b-v1")       # downloads the base model on first use
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

- **Data:** 19,701 Wikidata items with one unambiguous FSK rating (property P1981), collected 29 October 2026.
  - Labels: 1,756 at 0, 3,142 at 6, 7,959 at 12, 5,517 at 16 and 1,327 at 18.
  - 19,590 films, 93 TV titles and 18 others.
- **Wikipedia text:** fetched through the MediaWiki API in October 2026. 98% of training items have an article.
  - Each training step drops the text with probability 0.5, so the same model handles titles with and without an
    article. Most unrated titles have no article.
- **Recipe:**
  - One epoch, batch size 8, bf16.
  - AdamW with weight decay 0.01. Learning rate 1e-4 for LoRA and 1e-3 for the head.
  - 5% linear warm-up, then linear decay.
- **Hardware:** 39.8 minutes on one RTX 4090, about 6 GB peak GPU memory.

## Evaluation

Fold 0 of a stratified 5-fold split (3,941 items), trained on the other four folds with the same recipe.

| Input | Decision | Exact | Within one step | Too low | Too low by 2+ steps | Too high |
|---|---|---|---|---|---|---|
| with article text | most likely age | 64.3% | 96.3% | 17.6% | 1.42% | 18.1% |
| with article text | round up at 60% (database rule) | 63.4% | 97.0% | 13.8% | **0.61%** | 22.8% |
| article text removed | most likely age | 61.6% | 95.7% | 18.4% | 1.67% | 20.0% |
| article text removed | round up at 60% (database rule) | 61.1% | 96.4% | 13.4% | 0.63% | 25.5% |

- **One step** is one position on the scale 0 → 6 → 12 → 16 → 18.
- **Round up at 60%:** the database uses the smallest age whose cumulative probability reaches 0.6. When the model
  is unsure this picks the older age, which cuts suggestions that are two or more steps too low by more than half.
- **Ceiling:** rating boards often disagree by a step. FSK and MPA give the same age to only 49.8% of the films both
  rated (90.7% within one step).

## Intended use

Showing a suggested minimum age for films and TV shows that have no official rating, for example in a media library
app. Label it as a suggestion ("Suggested 12+"), never as an FSK or other official rating. The license requires this.

## Limitations

- **TV is barely covered.** Only 93 of the training titles are TV shows, too few to evaluate. TV estimates skew
  young: the average TV estimate is 8.4 against 11.6 for films, and some adult dramas get 12 (Game of Thrones, Shōgun,
  Succession).
- **Obscure titles are harder.** The test items have a German rating, so they are better documented than a typical
  unrated title. 63% of the scored titles had no Wikipedia text, and some have only a handful of Wikidata facts.
  Expect lower accuracy there than in the table above.
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

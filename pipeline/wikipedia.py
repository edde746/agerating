"""Wikipedia lead + plot text (English and German) for the collected items.

Two sources, both parsed into the same shape (lead = text before the first heading, plot = the section under a
Plot/Synopsis/Handlung/... heading, whitespace collapsed):
  api   current wikitext via the MediaWiki API (50 titles per request, honours Retry-After), parsed with mwparserfromhell
  dump  the wikimedia/wikipedia parquet dump on Hugging Face (20231101), plain text with headings on their own lines
The released model was trained on API text for the FSK-rated items (fetched first with --training). Most other items
got dump text; the API filled in the articles missing from the dump.
Usage: python wikipedia.py api LANG [--training]   items with a LANG article and no LANG text yet; --training: FSK-rated only
       python wikipedia.py dump LANG               same, from the dump (downloads ~11 GB for en, ~5 GB for de)
Output: data/text/LANG_{api,dump}_NNNNN.parquet  (qid, LANG_title, LANG_lead, LANG_plot); resumable, re-runs only fetch
        articles not tried by that source yet.
"""
import json, re, sys, time, urllib.error, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
import mwparserfromhell as mw
import pandas as pd
from common import DATA, TEXT, UA

TEXT_COLS = ["en_lead", "en_plot", "de_lead", "de_plot"]
PLOT = {"en": re.compile(r"^(plot|plot summary|synopsis|premise|story|summary|storyline|overview)$", re.I),
        "de": re.compile(r"^(handlung|inhalt|story|handlung und hintergrund)$", re.I)}
DUMP_CONFIG = {"en": "20231101.en", "de": "20231101.de"}
DUMP_DIR = DATA / "wpdump"
SEGMENT = 2000  # articles per output file


def http_json(url, params, timeout=120, tries=8):
    full = url + "?" + urllib.parse.urlencode(params)
    for i in range(tries):
        try:
            req = urllib.request.Request(full, headers={"User-Agent": UA, "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if i == tries - 1:
                raise
            wait = int(e.headers.get("Retry-After") or 0) or (5 * 2 ** i if e.code == 429 else 2 ** i)
            print(f"HTTP {e.code}, sleeping {wait}s", flush=True)
            time.sleep(wait)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def _tidy(s):
    return re.sub(r"\n{2,}", "\n", re.sub(r"[ \t]+", " ", s)).strip()


def _plain(node):
    code = mw.parse(str(node))
    for t in code.filter_tags(matches=lambda n: str(n.tag).lower() in ("ref", "gallery", "table")):
        try:
            code.remove(t)
        except ValueError:
            pass
    skip = ("file", "image", "datei", "bild", "category", "kategorie")
    for link in code.filter_wikilinks(matches=lambda n: str(n.title).split(":")[0].lower() in skip):
        try:
            code.remove(link)
        except ValueError:
            pass
    return _tidy(code.strip_code(normalize=True, collapse=True))


def parse_article(lang, wikitext):
    """(lead, plot) from an article's wikitext; plot is None without a matching level-2 section."""
    code = mw.parse(wikitext)
    lead = _plain(code.get_sections(include_lead=True, flat=True)[0])
    plot = None
    for s in code.get_sections(levels=[2]):
        if PLOT[lang].match(s.filter_headings()[0].title.strip_code().strip()):
            body = mw.parse(str(s))
            body.remove(body.filter_headings()[0])
            plot = _plain(body)
            break
    return lead, plot


HEADING = re.compile(r"\n\n([^\n]{1,80})\n(?!\n)")


def parse_dump_text(lang, text):
    """(lead, plot) from the dump's plain text, where section headings are lines of their own."""
    heads = list(HEADING.finditer(text))
    lead = _tidy(text[:heads[0].start()] if heads else text)
    plot = None
    for i, h in enumerate(heads):
        if PLOT[lang].match(h.group(1).strip()):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
            plot = _tidy(text[h.end():end]) or None
            break
    return lead or None, plot


def fetch_titles(lang, titles, batch=50):
    """article title -> (lead, plot) via batched wikitext requests; titles that fail to resolve are left out."""
    def one(chunk):
        d = http_json(f"https://{lang}.wikipedia.org/w/api.php", {
            "action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main", "titles": "|".join(chunk),
            "redirects": 1, "format": "json", "formatversion": 2, "maxlag": 5})
        q, alias = d["query"], {}
        for k in ("normalized", "redirects"):
            for m in q.get(k, []):
                alias[m["from"]] = m["to"]
        pages = {p["title"]: p for p in q["pages"]}
        out = {}
        for t in chunk:
            f = t
            while f in alias:
                f = alias[f]
            p = pages.get(f)
            if p and "revisions" in p:
                try:
                    out[t] = parse_article(lang, p["revisions"][0]["slots"]["main"]["content"])
                except Exception as e:
                    print("PARSE", lang, t, e, flush=True)
        return out
    res, t0 = {}, time.time()
    chunks = [titles[i:i + batch] for i in range(0, len(titles), batch)]
    with ThreadPoolExecutor(2) as ex:
        for i, out in enumerate(ex.map(one, chunks)):
            res |= out
            if i % 20 == 0:
                print(f"{lang}: {i}/{len(chunks)} batches {time.time() - t0:.0f}s", flush=True)
    print(f"{lang}: {len(res)}/{len(titles)} pages in {time.time() - t0:.0f}s", flush=True)
    return res


def load_texts():
    """qid -> en_lead, en_plot, de_lead, de_plot. Per language, API text wins over dump text."""
    parts = []
    for lang in ("en", "de"):
        paths = sorted(TEXT.glob(f"{lang}_*.parquet"))  # "api" sorts before "dump"
        if paths:
            cols = ["qid", f"{lang}_lead", f"{lang}_plot"]
            d = pd.concat([pd.read_parquet(p, columns=cols) for p in paths])
            parts.append(d[d[f"{lang}_lead"].notna()].drop_duplicates("qid").set_index("qid"))
    t = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=pd.Index([], name="qid"))
    return t.reindex(columns=TEXT_COLS)


def _segments(lang, source):
    return sorted(TEXT.glob(f"{lang}_{source}_*.parquet"))


def todo(lang, source, training=False):
    """qid, LANG_title of items with a LANG article, no LANG text yet and not tried by `source` before."""
    from data import load_items, load_meta
    it = load_items()
    it = it[it.fsk_rated] if training else it
    meta = load_meta().reindex(it.qid)
    have = load_texts()
    have = set(have.index[have[f"{lang}_lead"].notna()])
    tried = set()
    for p in _segments(lang, source):
        tried |= set(pd.read_parquet(p, columns=["qid"]).qid)
    col = f"{lang}_title"
    m = meta[meta[col].notna() & ~meta.index.isin(have) & ~meta.index.isin(tried)]
    return m[[col]].rename_axis("qid").reset_index()


def _write(lang, source, df):
    """Append `df` (qid, title, lead, plot) as new numbered segments."""
    n = len(_segments(lang, source))
    for i in range(0, len(df), SEGMENT):
        df.iloc[i:i + SEGMENT].to_parquet(TEXT / f"{lang}_{source}_{n + i // SEGMENT:05d}.parquet")


def api(lang, training=False):
    need = todo(lang, "api", training)
    print(f"{lang}: {len(need)} articles to fetch from the API", flush=True)
    col = f"{lang}_title"
    for i in range(0, len(need), SEGMENT):  # write as we go so an interrupted run keeps its progress
        part = need.iloc[i:i + SEGMENT].copy()
        res = fetch_titles(lang, part[col].dropna().unique().tolist())
        part[f"{lang}_lead"] = part[col].map(lambda t: res.get(t, (None, None))[0])
        part[f"{lang}_plot"] = part[col].map(lambda t: res.get(t, (None, None))[1])
        _write(lang, "api", part)
    print(f"{lang}: done", flush=True)


def dump(lang):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    t0 = time.time()
    need = todo(lang, "dump")
    col = f"{lang}_title"
    print(f"{lang}: {len(need)} articles to look up in the dump", flush=True)
    snapshot_download("wikimedia/wikipedia", repo_type="dataset", allow_patterns=f"{DUMP_CONFIG[lang]}/*.parquet",
                      local_dir=DUMP_DIR)
    print(f"{lang}: dump downloaded ({time.time() - t0:.0f}s)", flush=True)
    found, want = {}, pa.array(sorted(set(need[col])), pa.string())
    for p in sorted((DUMP_DIR / DUMP_CONFIG[lang]).glob("*.parquet")):
        f = pq.ParquetFile(p)
        titles = f.read(columns=["title"]).column("title")
        idx = pc.indices_nonzero(pc.is_in(titles, value_set=want)).to_pylist()
        if idx:
            texts = f.read(columns=["text"]).column("text").take(idx).to_pylist()
            for i, text in zip(idx, texts):
                found[titles[i].as_py()] = parse_dump_text(lang, text)
    need[f"{lang}_lead"] = need[col].map(lambda t: found.get(t, (None, None))[0])
    need[f"{lang}_plot"] = need[col].map(lambda t: found.get(t, (None, None))[1])
    _write(lang, "dump", need)  # misses are kept (lead None) so they count as tried; the API picks them up
    hit = need[f"{lang}_lead"].notna()
    print(f"{lang}: {int(hit.sum())}/{len(need)} found (plot {int(need[f'{lang}_plot'].notna().sum())}), "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    TEXT.mkdir(parents=True, exist_ok=True)
    cmd, lang = sys.argv[1], sys.argv[2]
    if cmd == "api":
        api(lang, "--training" in sys.argv)
    elif cmd == "dump":
        dump(lang)
    else:
        sys.exit(__doc__)

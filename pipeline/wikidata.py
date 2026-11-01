"""Collect Wikidata facts for every film / TV item through the Wikidata Query Service (WDQS).

Items: every item whose P31 (instance of) is one of UNIVERSE_CLASSES, plus every item with an FSK rating (P1981, the
training labels; a few of those are episodes or other classes).
Steps, each resumable; `all` runs them in order:
  items   data/wd/items.parquet          qid, universe, fsk_rated
  stmts   data/wd/stmts_NNNNN.parquet    qid, pid, value, kind (item|literal|uri|extid) of every truthy statement
          data/wd/meta_NNNNN.parquet     qid, sitelinks, n_statements, n_identifiers, revid, English label and
                                         description, en/de Wikipedia article titles
  labels  data/wd/labels.parquet         qid, label: English (else 'mul') label of every item-valued statement value
  ids     data/wd/ids.parquet            qid, pid, value: every external ID of the universe items (from the statements)
          data/wd/id_properties.parquet  pid, name of every external-ID property
  titles  data/wd/item_labels.parquet    qid, label in any language for universe items without an English label
Usage: python wikidata.py [items|stmts|labels|ids|titles|all]
"""
import json, sys, time, urllib.error, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from common import WD, UA, UNIVERSE_CLASSES, TARGET_PID, LABEL_PREF

WDQS = "https://query.wikidata.org/sparql"
BATCH = 250  # items per statement query


def sparql(q, tries=8, timeout=150):
    data = urllib.parse.urlencode({"query": q, "format": "json"}).encode()
    for i in range(tries):
        try:
            req = urllib.request.Request(WDQS, data=data,
                                         headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read(), strict=False)["results"]["bindings"]  # literals may hold raw control chars
        except urllib.error.HTTPError as e:
            if i == tries - 1:
                raise
            wait = int(e.headers.get("Retry-After") or 0) or (10 * 2 ** i if e.code == 429 else 3 * 2 ** i)
            print(f"HTTP {e.code}, sleeping {wait}s", flush=True)
            time.sleep(wait)
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"{type(e).__name__}: {e}; retry", flush=True)
            time.sleep(3 * 2 ** i)


def short(uri):
    return uri.rsplit("/", 1)[1] if uri.startswith("http://www.wikidata.org/entity/") else uri


def qid_order(qids):
    return sorted(qids, key=lambda q: int(q[1:]))


def universe():
    """QIDs of every item whose P31 is one of UNIVERSE_CLASSES (one WDQS query per class, about 30 s)."""
    u = set()
    for c, name in UNIVERSE_CLASSES.items():  # one query per class; a VALUES union stalls on WDQS
        rows = sparql(f"SELECT ?item WHERE {{ ?item wdt:P31 wd:{c} }}", timeout=300)
        u |= {short(r["item"]["value"]) for r in rows}
        print(f"{c} {name}: {len(rows)} (universe {len(u)})", flush=True)
    return u


def items():
    path = WD / "items.parquet"
    if path.exists():
        return pd.read_parquet(path)
    universe_ = universe()
    rated = {short(r["item"]["value"]) for r in sparql(f"SELECT DISTINCT ?item WHERE {{ ?item wdt:{TARGET_PID} [] }}")}
    qids = qid_order(universe_ | rated)
    df = pd.DataFrame({"qid": qids, "universe": [q in universe_ for q in qids],
                       "fsk_rated": [q in rated for q in qids]})
    df.to_parquet(path)
    print(f"items: {len(df)} ({int(df.universe.sum())} universe, {int(df.fsk_rated.sum())} FSK-rated)", flush=True)
    return df


_TYPES = {}


def prop_types():
    """pid -> datatype name; fetched once (joining types per triple made each batch ~8x slower)."""
    if not _TYPES:
        path = WD / "prop_types.parquet"
        if not path.exists():
            rows = sparql("SELECT ?prop ?t WHERE { ?prop wikibase:propertyType ?t }")
            pd.DataFrame([(short(r["prop"]["value"]), r["t"]["value"].rsplit("#", 1)[1]) for r in rows],
                         columns=["pid", "type"]).to_parquet(path)
        _TYPES.update(pd.read_parquet(path).set_index("pid")["type"].to_dict())
    return _TYPES


def stmt_batch(i, chunk, out=WD):
    sp, mp = out / f"stmts_{i:05d}.parquet", out / f"meta_{i:05d}.parquet"
    if sp.exists() and mp.exists():
        return 0
    types = prop_types()
    vals = " ".join(f"wd:{q}" for q in chunk)
    rows = sparql(f"""SELECT ?item ?p ?v WHERE {{ VALUES ?item {{ {vals} }} ?item ?p ?v .
        FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/")) }}""")
    recs = []
    for r in rows:
        pid = r["p"]["value"].rsplit("/", 1)[1]
        v = r["v"]
        if types.get(pid) == "ExternalId":
            recs.append((short(r["item"]["value"]), pid, v["value"], "extid"))
        elif v["type"] == "uri":
            val = short(v["value"])
            recs.append((short(r["item"]["value"]), pid, val, "item" if val[:1] == "Q" and val[1:].isdigit() else "uri"))
        else:
            recs.append((short(r["item"]["value"]), pid, v["value"], "literal"))
    df = pd.DataFrame(recs, columns=["qid", "pid", "value", "kind"]).drop_duplicates()
    meta = sparql(f"""SELECT ?item ?n ?ns ?ni ?rev ?label ?desc ?en ?de WHERE {{ VALUES ?item {{ {vals} }}
        OPTIONAL {{ ?item wikibase:sitelinks ?n }} OPTIONAL {{ ?item wikibase:statements ?ns }}
        OPTIONAL {{ ?item wikibase:identifiers ?ni }} OPTIONAL {{ ?item schema:version ?rev }}
        OPTIONAL {{ ?item rdfs:label ?label FILTER(LANG(?label) = "en") }}
        OPTIONAL {{ ?item schema:description ?desc FILTER(LANG(?desc) = "en") }}
        OPTIONAL {{ ?ea schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> ; schema:name ?en }}
        OPTIONAL {{ ?da schema:about ?item ; schema:isPartOf <https://de.wikipedia.org/> ; schema:name ?de }} }}""")
    g = lambda r, k: r[k]["value"] if k in r else None
    m = pd.DataFrame([{"qid": short(r["item"]["value"]), "sitelinks": g(r, "n"), "n_statements": g(r, "ns"),
                       "n_identifiers": g(r, "ni"), "revid": g(r, "rev"), "label": g(r, "label"),
                       "description": g(r, "desc"), "en_title": g(r, "en"), "de_title": g(r, "de")}
                      for r in meta]).drop_duplicates("qid")
    df.to_parquet(sp)
    m.to_parquet(mp)
    return len(df)


def stmts(workers=2):
    qids = items().qid.tolist()
    chunks = [qids[i:i + BATCH] for i in range(0, len(qids), BATCH)]
    t0, done = time.time(), 0
    with ThreadPoolExecutor(workers) as ex:
        for i, n in enumerate(ex.map(lambda a: stmt_batch(*a), enumerate(chunks))):
            done += n
            if i % 20 == 0:
                print(f"stmts {i + 1}/{len(chunks)} batches, {done} rows, {time.time() - t0:.0f}s", flush=True)
    print(f"stmts done: {len(chunks)} batches in {time.time() - t0:.0f}s", flush=True)


def labels(batch=1500, workers=2):
    path = WD / "labels.parquet"
    st = pd.concat([pd.read_parquet(p, columns=["value", "kind"]) for p in sorted(WD.glob("stmts_*.parquet"))])
    want = sorted(set(st.loc[st.kind == "item", "value"]))
    have = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["qid", "label"])
    todo = sorted(set(want) - set(have.qid))
    print(f"labels: {len(want)} distinct values, {len(todo)} to fetch", flush=True)

    def one(chunk):
        vals = " ".join(f"wd:{q}" for q in chunk)
        rows = sparql(f"""SELECT ?x ?l WHERE {{ VALUES ?x {{ {vals} }}
            ?x rdfs:label ?l FILTER(LANG(?l) IN ("en", "mul")) }}""")
        out = {}
        for r in rows:  # prefer en over mul
            q, l = short(r["x"]["value"]), r["l"]
            if q not in out or l.get("xml:lang") == "en":
                out[q] = l["value"]
        return out

    chunks = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    got, t0 = {}, time.time()
    save = lambda: pd.concat([have, pd.DataFrame(got.items(), columns=["qid", "label"])]) \
        .drop_duplicates("qid").to_parquet(path)
    with ThreadPoolExecutor(workers) as ex:
        for i, d in enumerate(ex.map(one, chunks)):
            got |= d
            if i % 50 == 0:
                print(f"labels {i + 1}/{len(chunks)} {time.time() - t0:.0f}s", flush=True)
                save()
    save()
    print(f"labels done: {len(got)} fetched in {time.time() - t0:.0f}s", flush=True)


def ids():
    """External IDs of the universe items (taken from the collected statements) and the ID property names."""
    u = set(items().query("universe").qid)
    df = pd.concat([pd.read_parquet(p) for p in sorted(WD.glob("stmts_*.parquet"))], ignore_index=True)
    df = df[(df.kind == "extid") & df.qid.isin(u)][["qid", "pid", "value"]].drop_duplicates()
    df.to_parquet(WD / "ids.parquet")
    rows = sparql("""SELECT ?p ?l WHERE { ?p wikibase:propertyType wikibase:ExternalId ;
        rdfs:label ?l FILTER(LANG(?l) = "en") }""")
    props = pd.DataFrame([(short(r["p"]["value"]), r["l"]["value"]) for r in rows], columns=["pid", "name"])
    props.drop_duplicates("pid").to_parquet(WD / "id_properties.parquet")
    print(f"ids: {len(df)} values in {df.pid.nunique()} of {len(props)} ID properties", flush=True)


def titles(batch=1500, workers=2):
    """Best available label in any language (LABEL_PREF order first) for universe items lacking an English one."""
    from data import load_meta
    u = items().query("universe").qid
    m = load_meta().reindex(u)
    qids = m.index[m.label.isna() & m.en_title.isna() & m.de_title.isna()].tolist()

    def one(chunk):
        vals = " ".join(f"wd:{q}" for q in chunk)
        best = {}
        for r in sparql(f"SELECT ?x ?l WHERE {{ VALUES ?x {{ {vals} }} ?x rdfs:label ?l }}"):
            q, lang = short(r["x"]["value"]), r["l"].get("xml:lang", "")
            rank = LABEL_PREF.index(lang) if lang in LABEL_PREF else len(LABEL_PREF)
            if q not in best or rank < best[q][0]:
                best[q] = (rank, r["l"]["value"])
        return [(q, v) for q, (_, v) in best.items()]

    chunks = [qids[i:i + batch] for i in range(0, len(qids), batch)]
    recs, t0 = [], time.time()
    with ThreadPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(one, chunks)):
            recs += r
            if i % 20 == 0:
                print(f"item labels {i + 1}/{len(chunks)} {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(recs, columns=["qid", "label"]).to_parquet(WD / "item_labels.parquet")
    print(f"item labels done: {len(recs)} of {len(qids)} in {time.time() - t0:.0f}s", flush=True)


STEPS = {"items": items, "stmts": stmts, "labels": labels, "ids": ids, "titles": titles}

if __name__ == "__main__":
    WD.mkdir(parents=True, exist_ok=True)
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in STEPS.items():
        if step in (name, "all"):
            fn()

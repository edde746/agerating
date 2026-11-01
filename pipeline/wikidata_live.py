"""Wikidata API access for items added after the batch collection (used by update.py).

entity_rows() turns a wbgetentities entity into exactly what the batch collection (wikidata.py) stores for an item:
truthy statement rows (qid, pid, value, kind), external-ID values and meta (label, description, en_title, de_title,
sitelinks, n_statements, n_identifiers, revid), so serialize.serialize_frame builds the same input text.
"""
import gzip, json, re, time, urllib.error, urllib.parse, urllib.request
from common import LABEL_PREF, UA

WD_API = "https://www.wikidata.org/w/api.php"
MAX_WAIT = 120  # seconds; longer rate-limit waits fail the call (update.py leaves the chunk for the next run)


class RateLimited(RuntimeError):
    pass


def api(url, params, timeout=60):
    """GET a MediaWiki API (gzip, maxlag=5), retrying on maxlag, 429, 5xx and network errors while honouring
    Retry-After; raises RateLimited once the waits would exceed MAX_WAIT seconds in total."""
    full = url + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": 2, "maxlag": 5})
    waited, i = 0, 0
    while True:
        why = None
        try:
            req = urllib.request.Request(full, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                d = json.loads(gzip.decompress(raw) if r.headers.get("Content-Encoding") == "gzip" else raw)
                if d.get("error", {}).get("code") != "maxlag":
                    if "error" in d:
                        raise RuntimeError(f"API error {d['error']}")
                    return d
                wait, why = int(r.headers.get("Retry-After") or 5), f"maxlag ({d['error'].get('lag')}s)"
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504):
                raise
            wait, why = int(e.headers.get("Retry-After") or 0) or min(60, 5 * 2 ** i), f"HTTP {e.code}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            wait, why = min(60, 3 * 2 ** i), type(e).__name__
        if waited + wait > MAX_WAIT:
            raise RateLimited(f"{url}: {why}, gave up after waiting {waited}s")
        time.sleep(wait)
        waited, i = waited + wait, i + 1


def get_entities(qids):
    """qid -> entity JSON (up to 50); missing ids are dropped."""
    d = api(WD_API, {"action": "wbgetentities", "ids": "|".join(qids),  # all languages: title fallback
                     "props": "info|claims|labels|descriptions|sitelinks"})
    return {q: e for q, e in d["entities"].items() if not e.get("missing")}


def get_labels(qids):
    """qid -> English label, else the language-neutral 'mul' label (what wikidata.labels stores)."""
    out = {}
    for i in range(0, len(qids), 50):
        d = api(WD_API, {"action": "wbgetentities", "ids": "|".join(qids[i:i + 50]), "props": "labels",
                         "languages": "en|mul"})
        for q, e in d["entities"].items():
            lab = e.get("labels", {})
            v = (lab.get("en") or lab.get("mul") or {}).get("value")
            if v is not None:
                out[q] = v
    return out


def _value(dtype, dv):
    """WDQS-style value string and kind for a snak datavalue."""
    t, v = dv["type"], dv["value"]
    if t == "wikibase-entityid":
        val = v.get("id") or f"Q{v['numeric-id']}"
        return val, "item" if val[:1] == "Q" and val[1:].isdigit() else "uri"
    if dtype == "external-id":
        return v, "extid"
    if t == "time":  # WDQS drops the '+' and turns unknown month/day (00) into 01
        s = v["time"].lstrip("+")
        return re.sub(r"^(-?\d+-\d\d)-00", r"\1-01", re.sub(r"^(-?\d+)-00-00", r"\1-01-01", s)), "literal"
    if t == "quantity":
        return v["amount"].lstrip("+"), "literal"
    if t == "monolingualtext":
        return v["text"], "literal"
    if t == "string":
        return v, "uri" if dtype in ("url", "commonsMedia", "geo-shape", "tabular-data") else "literal"
    return json.dumps(v, sort_keys=True), "literal"


def entity_rows(qid, e):
    """(truthy statement rows, external-ID rows, meta dict) for one entity, matching the batch WDQS format."""
    rows, ext = [], []
    n_statements = n_identifiers = 0
    for pid, stmts in e.get("claims", {}).items():
        n_statements += len(stmts)
        best = [s for s in stmts if s.get("rank") == "preferred"] or [s for s in stmts if s.get("rank") == "normal"]
        for s in best:
            sn = s.get("mainsnak", {})
            if sn.get("datatype") == "external-id":
                n_identifiers += 1
            if sn.get("snaktype") != "value":
                continue
            val, kind = _value(sn.get("datatype"), sn["datavalue"])
            rows.append((qid, pid, val, kind))
            if kind == "extid":
                ext.append((qid, pid, val))
    lab = lambda k, lang: e.get(k, {}).get(lang, {}).get("value")
    sl = e.get("sitelinks", {})
    meta = {"qid": qid, "label": lab("labels", "en"), "description": lab("descriptions", "en"),
            "en_title": sl.get("enwiki", {}).get("title"), "de_title": sl.get("dewiki", {}).get("title"),
            "sitelinks": len(sl), "n_statements": n_statements, "n_identifiers": n_identifiers,
            "revid": e.get("lastrevid"), "label_fallback": label_fallback(e.get("labels", {}))}
    return rows, ext, meta


def label_fallback(labels):
    """Same choice as wikidata.titles: first label in LABEL_PREF order, else the first remaining one."""
    for lang in LABEL_PREF:
        if lang in labels:
            return labels[lang]["value"]
    return next(iter(labels.values()))["value"] if labels else None

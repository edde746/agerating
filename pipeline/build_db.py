"""Build the full SQLite database: every Wikidata film / TV item with an official FSK rating or an estimate.

Tables
  title        qid PK, label, year, kind (film|tv|other), sitelinks, enwiki, dewiki, revid (Wikidata revision the row
               was built from), fsk_official, fsk_estimate, p0 p6 p12 p16 p18 (model probabilities), confidence,
               had_article (Wikipedia text was part of the input), source, model, scored_at
  external_id  pid, value, qid: every truthy external ID of each title (IMDb P345, TMDB P4947/P4983, TVDB P4835, ...)
  id_property  pid, name of external-ID properties
  label        qid, label of linked items (people, genres, studios, ...) used to build model input for new titles
  info         key, value: model, target, decision threshold, build time (update.py adds updated_at)
source is official (FSK rating on Wikidata), batch (scored by this build) or update (added by update.py).
fsk_estimate is the smallest age whose cumulative probability reaches TAU (rounds up when the model is unsure);
titles with an official FSK rating on Wikidata carry fsk_official and no estimate.
Usage: python build_db.py [PATH] [--model NAME]   (default data/agerating.db; built next to it, swapped in atomically)
"""
import argparse, os, sqlite3, time
from pathlib import Path
import numpy as np
import pandas as pd
import data as D
from common import BUCKETS, DATA, MODEL_NAME, TARGET_PID, TAU, WD, now
from serialize import facts, has_article_text

DB = DATA / "agerating.db"
P = [f"p{a}" for a in BUCKETS]
SCHEMA = """
CREATE TABLE title (qid TEXT PRIMARY KEY, label TEXT, year INTEGER, kind TEXT, sitelinks INTEGER, enwiki TEXT,
  dewiki TEXT, revid INTEGER, fsk_official INTEGER, fsk_estimate INTEGER, p0 REAL, p6 REAL, p12 REAL, p16 REAL,
  p18 REAL, confidence REAL, had_article INTEGER, source TEXT NOT NULL, model TEXT, scored_at TEXT);
CREATE TABLE external_id (pid TEXT NOT NULL, value TEXT NOT NULL, qid TEXT NOT NULL,
  PRIMARY KEY (pid, value, qid)) WITHOUT ROWID;
CREATE INDEX external_id_qid ON external_id (qid);
CREATE TABLE id_property (pid TEXT PRIMARY KEY, name TEXT);
CREATE TABLE label (qid TEXT PRIMARY KEY, label TEXT NOT NULL) WITHOUT ROWID;
CREATE TABLE info (key TEXT PRIMARY KEY, value TEXT);
"""
TITLE_COLS = ["qid", "label", "year", "kind", "sitelinks", "enwiki", "dewiki", "revid", "fsk_official", "fsk_estimate",
              *P, "confidence", "had_article", "source", "model", "scored_at"]


def decide(probs):
    """Smallest age bucket whose cumulative probability reaches TAU."""
    return np.array(BUCKETS)[(np.cumsum(probs, 1) >= TAU).argmax(1)]


def title_rows(qids, st, meta, labels, text, probs, fallback, source, model_name, scored_at):
    """Title table rows for `qids` (shared by the batch build and update.py).

    meta: indexed by qid, with revid; probs: DataFrame p0..p18 indexed by qid (items without an official rating);
    fallback: Series qid -> label in any language."""
    q = pd.Index(qids, name="qid")
    m = meta.reindex(q)
    d = st[st.pid.isin(["P577", "P571", "P580"]) & st.qid.isin(set(qids))]  # publication, inception, start time
    year = D.year(d.value).groupby(d.qid.values).min().reindex(q)
    off = D.official(st[st.qid.isin(set(qids))], TARGET_PID).reindex(q)
    pr = probs.reindex(q)
    has = pr.iloc[:, 0].notna().values & off.isna().values
    est = pd.Series(np.nan, index=q)
    est[has] = decide(pr.values[has])
    df = pd.DataFrame({"label": m.label.fillna(m.en_title).fillna(m.de_title).fillna(fallback.reindex(q)),
                       "year": year, "kind": D.item_kind(st[st.qid.isin(set(qids))], labels).reindex(q).fillna("other"),
                       "sitelinks": m.sitelinks, "enwiki": m.en_title, "dewiki": m.de_title, "revid": m.revid,
                       "fsk_official": off, "fsk_estimate": est}, index=q)
    for k, c in enumerate(P):
        df[c] = np.where(has, pr.iloc[:, k].values.astype(np.float64), np.nan).round(4)  # float32 would keep 0.62199998
    df["confidence"] = df[P].max(1)
    df["had_article"] = has_article_text(text, list(q)).astype(int).values
    df["source"] = np.where(off.notna(), "official", source)
    df["model"] = np.where(has, model_name, None)
    df["scored_at"] = scored_at
    for c in ("year", "sitelinks", "revid", "fsk_official", "fsk_estimate"):
        df[c] = df[c].astype("Int64")
    return df.reset_index()[TITLE_COLS]


def records(df):
    """DataFrame -> list of tuples with NaN/NA as None (sqlite NULL)."""
    return [tuple(None if (v is pd.NA or (isinstance(v, float) and np.isnan(v))) else (int(v) if isinstance(v, np.integer) else v)
                  for v in row) for row in df.astype(object).itertuples(index=False, name=None)]


def build(path, model_name):
    t0 = time.time()
    st, meta, labels, text = facts()
    u = D.load_items().query("universe").qid.tolist()
    probs = pd.read_parquet(DATA / "pred" / f"{model_name}.parquet")[P]
    fallback = pd.read_parquet(WD / "item_labels.parquet").set_index("qid").label
    titles = title_rows(u, st, meta, labels, text, probs, fallback, "batch", model_name, now())
    missing = titles[(titles.source == "batch") & titles.fsk_estimate.isna()]
    assert not len(missing), f"{len(missing)} universe titles have neither an official rating nor an estimate"
    ids = pd.read_parquet(WD / "ids.parquet")
    ids = ids[ids.qid.isin(set(u))][["pid", "value", "qid"]].drop_duplicates()
    props = pd.read_parquet(WD / "id_properties.parquet")
    lab = labels.rename_axis("qid").reset_index()
    print(f"prepared {len(titles)} titles, {len(ids)} external IDs, {len(lab)} labels ({time.time() - t0:.0f}s)",
          flush=True)
    tmp = path.with_suffix(".building")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.executescript(SCHEMA)
    with con:
        con.executemany(f"INSERT INTO title VALUES ({','.join('?' * len(TITLE_COLS))})", records(titles))
        con.executemany("INSERT INTO external_id VALUES (?,?,?)", ids.itertuples(index=False, name=None))
        con.executemany("INSERT INTO id_property VALUES (?,?)", props.itertuples(index=False, name=None))
        con.executemany("INSERT INTO label VALUES (?,?)", lab.itertuples(index=False, name=None))
        con.executemany("INSERT INTO info VALUES (?,?)", [
            ("model", model_name), ("target", TARGET_PID), ("tau", str(TAU)), ("built_at", now()),
            ("titles", str(len(titles)))])
    con.execute("ANALYZE")
    con.close()
    os.replace(tmp, path)
    print(f"built {path} ({path.stat().st_size / 2**20:.0f} MB) in {time.time() - t0:.0f}s; "
          f"sources {titles.source.value_counts().to_dict()}, kinds {titles.kind.value_counts().to_dict()}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", nargs="?", type=Path, default=DB)
    p.add_argument("--model", default=MODEL_NAME, help="name under data/pred/ and in the title.model column")
    a = p.parse_args()
    build(a.path, a.model)

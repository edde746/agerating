"""Add the Wikidata film / TV items that appeared since the database was built or last updated.

1. Lists the universe on WDQS (every item whose P31 is one of UNIVERSE_CLASSES) and keeps the items that are not in
   the `title` table yet.
2. Per chunk of new items: fetches the entities (50 per call), the labels of linked items missing from the `label`
   table and the en/de Wikipedia articles, builds the model input with serialize.serialize_frame and scores the items
   without an official FSK rating. These are the code paths of the batch build.
3. Inserts the titles and all their external IDs, one transaction per chunk, and sets `updated_at` and `titles` in
   the `info` table.
Existing titles are not refreshed, and items that left the universe stay in the database. Chunks that fail (rate
limits, network errors) are left for the next run; the script then exits with status 1 after saving the other chunks.
Usage: python update.py [--db PATH] [--model DIR] [--device cpu|mps|cuda] [--minutes M] [--limit N]
The last output line is a JSON summary: {"new": ..., "added": ..., "skipped": ..., "left": ..., "titles": ...}
"""
import argparse, json, sqlite3, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import build_db as B
import data as D
import wikidata as W
import wikidata_live as L
from common import MODELS, TARGET_PID, UNIVERSE_CLASSES, now
from serialize import SER_PIDS, serialize_frame
from wikipedia import TEXT_COLS, fetch_titles

CHUNK = 100  # items per transaction
MAX_FAILS = 3  # consecutive failed chunks before the run stops


class Scorer:
    """The model named in the database's info table (models/<name>, or --model), loaded on first use."""

    def __init__(self, con, path, dev):
        self.name = con.execute("SELECT value FROM info WHERE key = 'model'").fetchone()[0]
        self.path, self.dev, self.model = Path(path) if path else MODELS / self.name, dev, None

    def __call__(self, texts):
        import model as M
        if self.model is None:
            self.model = M.load(self.path, self.dev)
        return M.predict(self.model, texts, bs=16)


def linked_labels(con, st):
    """Labels of the item values the serializer uses: from the label table, the rest fetched and stored."""
    want = sorted(set(st.loc[(st.kind == "item") & st.pid.isin(SER_PIDS), "value"]))
    have = {}
    for i in range(0, len(want), 500):
        chunk = want[i:i + 500]
        have |= dict(con.execute(f"SELECT qid, label FROM label WHERE qid IN ({','.join('?' * len(chunk))})",
                                 chunk).fetchall())
    missing = [q for q in want if q not in have]
    if missing:
        got = L.get_labels(missing)
        with con:
            con.executemany("INSERT OR IGNORE INTO label VALUES (?, ?)", list(got.items()))
        have |= got
    return pd.Series(have, dtype=object)


def article_texts(meta):
    """en/de lead and plot of the items' Wikipedia articles (batched API requests), indexed like meta."""
    out = pd.DataFrame(index=meta.index, columns=TEXT_COLS, dtype=object)
    for lang in ("en", "de"):
        titles = meta[f"{lang}_title"].dropna()
        got = fetch_titles(lang, sorted(set(titles))) if len(titles) else {}
        for q, t in titles.items():
            if t in got:
                out.at[q, f"{lang}_lead"], out.at[q, f"{lang}_plot"] = got[t]
    return out


def add_chunk(con, scorer, qids):
    """Fetch, score and insert `qids`; returns (added, skipped) where skipped items are gone, redirected or no
    longer film/TV items."""
    want, ents = set(qids), {}
    for i in range(0, len(qids), 50):
        ents |= {q: e for q, e in L.get_entities(qids[i:i + 50]).items() if q in want}
    rows, ext, metas = [], [], []
    for q, e in ents.items():
        r, x, m = L.entity_rows(q, e)
        if {v for _, p, v, _ in r if p == "P31"} & set(UNIVERSE_CLASSES):
            rows += r
            ext += x
            metas.append(m)
    added = [m["qid"] for m in metas]
    skipped = [q for q in qids if q not in set(added)]
    if not added:
        return added, skipped
    st = pd.DataFrame(rows, columns=["qid", "pid", "value", "kind"])
    meta = pd.DataFrame(metas).set_index("qid")
    labels = linked_labels(con, st)
    text = article_texts(meta)
    off = D.official(st, TARGET_PID)
    todo = [q for q in added if q not in off.index]
    probs = pd.DataFrame(np.empty((0, len(B.P))), columns=B.P)
    if todo:
        inp = serialize_frame(st, meta, labels, TARGET_PID, text=text, qids=todo)
        probs = pd.DataFrame(scorer(inp.tolist()), index=inp.index, columns=B.P)
    title = B.title_rows(added, st, meta, labels, text, probs, meta.label_fallback, "update", scorer.name, now())
    with con:
        con.executemany(f"INSERT INTO title VALUES ({','.join('?' * len(B.TITLE_COLS))})", B.records(title))
        con.executemany("INSERT OR IGNORE INTO external_id VALUES (?, ?, ?)", [(p, v, q) for q, p, v in ext])
    return added, skipped


def main(a):
    t0 = time.time()
    con = sqlite3.connect(a.db, timeout=30)
    have = {q for (q,) in con.execute("SELECT qid FROM title")}
    new = W.qid_order(W.universe() - have)[:a.limit or None]
    print(f"{len(new)} new items ({time.time() - t0:.0f}s)", flush=True)
    scorer = Scorer(con, a.model, a.device)
    added = skipped = failed = fails = 0
    for i in range(0, len(new), CHUNK):
        if a.minutes and time.time() - t0 > a.minutes * 60:
            print(f"time budget of {a.minutes} min used up", flush=True)
            break
        chunk = new[i:i + CHUNK]
        try:
            ok, skip = add_chunk(con, scorer, chunk)
        except Exception:  # rate limits, network or API errors: these items wait for the next run
            traceback.print_exc()
            failed, fails = failed + len(chunk), fails + 1
            if fails >= MAX_FAILS:
                print(f"{fails} chunks failed in a row, stopping", flush=True)
                break
            time.sleep(60)
            continue
        fails, added, skipped = 0, added + len(ok), skipped + len(skip)
        print(f"{i + len(chunk)}/{len(new)}: +{len(ok)} titles, {len(skip)} skipped ({time.time() - t0:.0f}s)",
              flush=True)
    titles = con.execute("SELECT COUNT(*) FROM title").fetchone()[0]
    if added:
        with con:
            con.executemany("INSERT OR REPLACE INTO info VALUES (?, ?)",
                            [("updated_at", now()), ("titles", str(titles))])
        con.execute("ANALYZE")
        con.execute("VACUUM")
    con.close()
    summary = {"new": len(new), "added": added, "skipped": skipped, "failed": failed,
               "left": len(new) - added - skipped, "titles": titles, "minutes": round((time.time() - t0) / 60, 1)}
    print(json.dumps(summary), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", type=Path, default=B.DB)
    p.add_argument("--model", help="model directory (default models/<info.model>)")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
    p.add_argument("--minutes", type=float, help="stop starting new chunks after this many minutes")
    p.add_argument("--limit", type=int, help="process at most this many new items")
    sys.exit(main(p.parse_args()))

"""Build the small lookup database from the full one: external ID -> suggested age.

One table per ID type, each (id INTEGER PRIMARY KEY, age INTEGER NOT NULL) WITHOUT ROWID:
  imdb         P345    IMDb title ID as a number, without "tt" (tt0133093 -> 133093)
  tmdb_movie   P4947   TMDB movie ID
  tmdb_tv      P4983   TMDB TV series ID
  tvdb_series  P4835   TheTVDB series ID
  tvdb_movie   P12196  TheTVDB movie ID
TMDB and TheTVDB number movies and series separately, so the same number can mean different titles in two tables.
age is the official FSK rating where Wikidata has one, else the model's estimate. An ID on several titles takes the
official rating if one of them has it, else the highest estimate. Values that are not plain numbers (URLs, slugs,
Wikidata's "unknown value") are left out.
Usage: python build_ids_db.py [FULL_DB] [OUT]   (default data/agerating.db -> data/agerating-ids.db)
"""
import argparse, os, sqlite3, time
from pathlib import Path
from build_db import DB

TABLES = {"imdb": "P345", "tmdb_movie": "P4947", "tmdb_tv": "P4983", "tvdb_series": "P4835", "tvdb_movie": "P12196"}


def build(full, out):
    t0 = time.time()
    tmp = out.with_suffix(".building")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.execute("PRAGMA page_size = 65536")
    con.execute("ATTACH DATABASE ? AS f", (str(full),))
    pids = ", ".join(f"'{p}'" for p in TABLES.values())
    con.execute(f"""CREATE TEMP VIEW id_age AS
        SELECT e.pid, e.value, COALESCE(MAX(t.fsk_official), MAX(t.fsk_estimate)) AS age
        FROM f.external_id e JOIN f.title t ON t.qid = e.qid
        WHERE e.pid IN ({pids}) GROUP BY e.pid, e.value""")
    with con:
        for table, pid in TABLES.items():
            num = "substr(value, 3)" if pid == "P345" else "value"
            ok = "value GLOB 'tt[0-9]*'" if pid == "P345" else "value GLOB '[0-9]*'"
            con.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, age INTEGER NOT NULL) WITHOUT ROWID")
            con.execute(f"""INSERT INTO {table} SELECT CAST({num} AS INTEGER), MAX(age) FROM id_age
                WHERE pid = ? AND {ok} AND NOT {num} GLOB '*[^0-9]*' GROUP BY 1 ORDER BY 1""", (pid,))
    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}
    con.execute("DETACH DATABASE f")
    con.execute("VACUUM")
    con.close()
    os.replace(tmp, out)
    print(f"built {out} ({out.stat().st_size / 2**20:.1f} MB) in {time.time() - t0:.0f}s: {counts}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("full", nargs="?", type=Path, default=DB)
    p.add_argument("out", nargs="?", type=Path, default=DB.with_name("agerating-ids.db"))
    a = p.parse_args()
    build(a.full, a.out)

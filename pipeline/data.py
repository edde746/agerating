"""Loaders for the collected Wikidata data (wikidata.py) and helpers derived from truthy statements."""
import pandas as pd
from common import WD, TARGET_PID, TARGET_MAP, TV_CLASSES


def load_items():
    """qid, universe (film/TV class), fsk_rated (has an FSK rating: training label)."""
    return pd.read_parquet(WD / "items.parquet")


def load_stmts():
    """Truthy statements of every collected item: qid, pid, value, kind (item|literal|uri|extid).

    The concatenated shards are cached in data/wd/stmts.parquet (rebuilt when a shard is newer)."""
    cache, shards = WD / "stmts.parquet", sorted(WD.glob("stmts_*.parquet"))
    if not cache.exists() or max(p.stat().st_mtime for p in shards) > cache.stat().st_mtime:
        st = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
        st.drop_duplicates().reset_index(drop=True).to_parquet(cache)
    return pd.read_parquet(cache)


def load_meta():
    """Per item: sitelinks, n_statements, n_identifiers, revid, English label/description, en/de Wikipedia titles."""
    m = pd.concat([pd.read_parquet(p) for p in sorted(WD.glob("meta_*.parquet"))]).drop_duplicates("qid")
    for c in ("sitelinks", "n_statements", "n_identifiers", "revid"):
        m[c] = pd.to_numeric(m[c])
    return m.set_index("qid")


def load_labels():
    """qid -> English (else 'mul') label of item-valued statement values."""
    return pd.read_parquet(WD / "labels.parquet").drop_duplicates("qid").set_index("qid")["label"]


def official(st, pid=TARGET_PID):
    """qid -> official age (0/6/12/16/18) in rating system `pid`; items with conflicting values are dropped."""
    r = st[st.pid == pid]
    r = r.assign(a=r.value.map(TARGET_MAP)).dropna(subset=["a"])
    n = r.groupby("qid").a.nunique()
    return r[r.qid.isin(n[n == 1].index)].drop_duplicates("qid").set_index("qid").a.astype(int)


def item_kind(st, labels):
    """qid -> film | tv | other, from P31 classes and their labels."""
    p31 = st[st.pid == "P31"][["qid", "value"]]
    lab = p31.value.map(labels).fillna("").str.lower()
    tv = p31.value.isin(TV_CLASSES) | lab.str.contains("television|tv series|miniseries|web series|anime series")
    film = lab.str.contains("film|movie")
    k = pd.Series("other", index=pd.Index(st.qid.unique(), name="qid"))
    k[k.index.isin(p31.qid[film])] = "film"
    k[k.index.isin(p31.qid[tv])] = "tv"
    tvfilm = p31.qid[lab.str.contains("television film|tv film|made-for-tv")]
    k[k.index.isin(tvfilm)] = "film"  # "television film" matched "television" above
    return k


def groups(st, qids):
    """Split group for grouped cross-validation: first series (P179), else first director (P57), else the item."""
    g = pd.Series({q: f"qid:{q}" for q in qids})
    for pid in ("P57", "P179"):  # later overrides earlier -> series wins
        s = st[(st.pid == pid) & st.qid.isin(qids)].sort_values("value").drop_duplicates("qid")
        g[s.qid.values] = [f"{pid}:{v}" for v in s.value]
    return g


def year(s):
    """Year of WDQS-style date strings (e.g. '2024-02-27T00:00:00Z'); NaN if unparseable."""
    return pd.to_numeric(s.str.extract(r"^([+-]?\d{1,4})-")[0], errors="coerce")

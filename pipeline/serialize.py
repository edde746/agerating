"""Serialize an item's Wikidata facts (+ optional Wikipedia text) into the model's input text.

One code path for training, batch scoring and scoring new titles (update.py). serialize_frame() takes truthy
statement rows (qid, pid, value, kind) as produced by the batch collection (wikidata.py) and by the per-item adapter
(wikidata_live.entity_rows), item meta (label, description, en_title, de_title), linked-item labels and optional
Wikipedia text; it gives the same text for 600k items or for one freshly fetched item. Multi-valued fields list item
values in ascending QID order (WDQS returns no statement order), so both paths agree. The item's own FSK rating
is never included.

Example (abridged):
  Title: Dune: Part Two (2024)
  Description: 2024 film directed by Denis Villeneuve
  Type: film
  Genres: action film, adventure film, science fiction film, ...
  Director: Denis Villeneuve
  Cast: Josh Brolin, Dave Bautista, Javier Bardem, ...
  Duration: 165 min
  Ratings elsewhere: LSF 13+; MPA PG-13; NMHH Category IV; Kijkwijzer 12; ...
  Plot: Following the defeat of House Atreides by House Harkonnen, ...
"""
import functools
import pandas as pd
import data as D
from common import RATING_PROPS, TARGET_PID
from wikipedia import load_texts

FIELDS = [("P31", "Type", 2), ("P136", "Genres", 8), ("P495", "Country", 3), ("P364", "Original language", 2),
          ("P57", "Director", 3), ("P58", "Screenwriter", 3), ("P161", "Cast", 10), ("P272", "Production company", 3),
          ("P750", "Distributor", 3), ("P449", "Network", 2), ("P179", "Part of series", 2), ("P8345", "Franchise", 2),
          ("P144", "Based on", 2), ("P921", "Main subject", 6), ("P840", "Set in", 4), ("P166", "Awards", 5),
          ("P1411", "Nominations", 5), ("P86", "Composer", 2)]
SER_PIDS = {p for p, _, _ in FIELDS} | {"P577", "P2047"} | set(RATING_PROPS)
TEXT_COLS = (("en_plot", "Plot"), ("en_lead", "Summary"), ("de_plot", "Handlung"))  # first one with >= 200 chars
MIN_TEXT, PLOT_CHARS = 200, 2500


def has_article_text(text, qids):
    """qid -> True if a Wikipedia text line would be added."""
    t = text.reindex(qids)
    ok = pd.Series(False, index=pd.Index(qids))
    for col, _ in TEXT_COLS:
        ok |= t[col].fillna("").str.len().ge(MIN_TEXT).values
    return ok


def serialize_frame(st, meta, labels, target_pid=TARGET_PID, with_text=True, text=None, qids=None):
    """qid -> input text. st: truthy statements (qid, pid, value, kind); meta indexed by qid; labels qid -> label."""
    qids = list(meta.index) if qids is None else list(qids)
    st = st[st.pid.isin(SER_PIDS) & st.qid.isin(set(qids))]
    it = st[st.kind == "item"].drop_duplicates(["qid", "pid", "value"])
    it = it.assign(l=it.value.map(labels), n=pd.to_numeric(it.value.str[1:], errors="coerce")).dropna(subset=["l"])
    it = it.sort_values(["qid", "pid", "n"], kind="stable")
    lists = {pid: it[it.pid == pid].groupby("qid", sort=False).l.agg(lambda s, n=n: ", ".join(s.iloc[:n])).to_dict()
             for pid, _, n in FIELDS}
    d = st[st.pid == "P577"]
    year = D.year(d.value).groupby(d.qid.values).min().dropna().to_dict()
    d = st[st.pid == "P2047"]
    dur = pd.to_numeric(d.value, errors="coerce").groupby(d.qid.values).max().dropna().to_dict()
    r = it[it.pid.isin(RATING_PROPS.keys()) & (it.pid != target_pid)]
    rat = (r.pid.map(RATING_PROPS) + " " + r.l).groupby(r.qid.values, sort=False).agg("; ".join).to_dict()
    m = meta.reindex(qids)
    title = m.label.fillna(m.en_title).fillna(m.de_title).to_dict()
    desc = m.description.to_dict()
    txt = {}
    if with_text and text is not None:
        t = text.reindex(qids)
        for col, name in TEXT_COLS:
            for q, v in t[col].items():
                if q not in txt and isinstance(v, str) and len(v) >= MIN_TEXT:
                    txt[q] = f"{name}: {v[:PLOT_CHARS]}"
    out = {}
    for q in qids:
        lines = [f"Title: {title.get(q) if isinstance(title.get(q), str) else q}"
                 + (f" ({int(year[q])})" if q in year else "")]
        if isinstance(desc.get(q), str):
            lines.append(f"Description: {desc[q]}")
        for pid, name, _ in FIELDS:
            if q in lists[pid]:
                lines.append(f"{name}: {lists[pid][q]}")
        if q in dur:
            lines.append(f"Duration: {int(dur[q])} min")
        lines.append("Ratings elsewhere: " + rat.get(q, "none"))
        if q in txt:
            lines.append(txt[q])
        out[q] = "\n".join(lines)
    return pd.Series(out, name="text", dtype=object)


@functools.lru_cache(maxsize=1)
def facts():
    """Batch inputs: truthy statements, meta and labels of all collected items, Wikipedia texts."""
    return D.load_stmts(), D.load_meta(), D.load_labels(), load_texts()


def serialize(qids, with_text=True):
    """Batch input texts for `qids`."""
    st, meta, labels, text = facts()
    return serialize_frame(st, meta, labels, TARGET_PID, with_text, text, qids)

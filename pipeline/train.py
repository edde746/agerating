"""Fine-tune the classifier on every item with an official FSK rating, or evaluate it on one cross-validation fold.

Text dropout: each training step uses the facts-only version of an item (no Wikipedia line) with probability
--text-dropout, so one model serves items with and without an article (98% of the FSK-rated items have one, most
unrated items do not).
Extra labels: --labels FILE (parquet or CSV with columns qid, age) adds items without an official rating as training
data, e.g. TV shows whose FSK rating was looked up elsewhere. Official ratings win where both exist. In CV mode the
extra items get their own folds and are evaluated separately (extra_* columns).
Usage: python train.py                       train on all labelled items -> models/<MODEL_NAME>/
       python train.py --fold K [--split random|group]
                                             5-fold CV, train on 4 folds and evaluate fold K, with and without the
                                             Wikipedia line -> results/eval.csv, results/oof/*.parquet
Options: --epochs 1  --batch-size 8  --lr 1e-4  --text-dropout 0.5  --labels FILE  --name NAME
Runs on a CUDA GPU (bf16, ~6.5 GB peak; about 40 min on an RTX 4090 for 19.7k items) or Apple silicon (MPS).
"""
import argparse, math, random, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
import data as D
import model as M
from common import BUCKETS, MODEL_NAME, MODELS, RESULTS, TARGET_PID, TAU, metrics
from serialize import serialize


def labelled(extra=None):
    """(qids, bucket indices, CV groups, is_extra): every item with an official FSK rating, then the items of the
    `extra` label file that have none."""
    st = D.load_stmts()
    y = D.official(st, TARGET_PID)
    n = len(y)
    if extra:
        e = pd.read_parquet(extra) if str(extra).endswith(".parquet") else pd.read_csv(extra)
        e = e.drop_duplicates("qid").set_index("qid")["age"].astype(int)
        y = pd.concat([y, e[~e.index.isin(y.index)]])
    is_extra = np.arange(len(y)) >= n
    return y.index, np.array([BUCKETS.index(v) for v in y.values]), D.groups(st, list(y.index)).reindex(y.index).values, is_extra


def split(kind, fold, y, g, is_extra):
    """Train/test indices of `fold`. Official and extra items are folded separately, so the official test fold is
    the same with or without extra labels."""
    parts = [folds(kind, y[m], g[m])[fold] for m in (~is_extra, is_extra) if m.any()]
    idx = [np.flatnonzero(m) for m in (~is_extra, is_extra) if m.any()]
    return (np.concatenate([i[tr] for i, (tr, _) in zip(idx, parts)]),
            np.concatenate([i[te] for i, (_, te) in zip(idx, parts)]))


def peak_gb(dev):
    if dev == "cuda":
        return torch.cuda.max_memory_allocated() / 2**30
    return torch.mps.driver_allocated_memory() / 2**30 if dev == "mps" else float("nan")


def folds(kind, y, g, n=5, seed=0):
    sp = StratifiedKFold(n, shuffle=True, random_state=seed) if kind == "random" else \
        StratifiedGroupKFold(n, shuffle=True, random_state=seed)
    return list(sp.split(np.zeros(len(y)), y, g))


def main(a):
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    dev = M.device()
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.85)
    qids, y, groups, is_extra = labelled(a.labels)
    text = serialize(qids).reindex(qids).fillna("").tolist()
    bare = serialize(qids, with_text=False).reindex(qids).fillna("").tolist()
    drop_rng = np.random.default_rng(1)
    tr, te = (np.arange(len(y)), None) if a.fold is None else split(a.split, a.fold, y, groups, is_extra)
    model = M.Clf().to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": [p for p in model.bb.parameters() if p.requires_grad], "lr": a.lr},
                             {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=0.01)
    steps = a.epochs * math.ceil(len(tr) / a.batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / (0.05 * steps)) * max(0.0, (steps - s) / steps))
    print(f"{len(y)} labelled items ({int(is_extra.sum())} from {a.labels}), class counts {np.bincount(y).tolist()}, "
          f"training on {len(tr)} on {dev}", flush=True)
    t0, step = time.time(), 0
    for ep in range(a.epochs):
        model.train()
        order = np.random.permutation(tr)
        for i in range(0, len(order), a.batch_size):
            j = order[i:i + a.batch_size]
            with torch.autocast(dev, dtype=torch.bfloat16):
                batch = [bare[k] if a.text_dropout and drop_rng.random() < a.text_dropout else text[k] for k in j]
                loss = nn.functional.cross_entropy(model(batch), torch.tensor(y[j], device=dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(); step += 1
            if step % 200 == 0:
                print(f"ep{ep} step {step}/{steps} loss {loss.item():.3f} {time.time() - t0:.0f}s "
                      f"peak {peak_gb(dev):.1f}GB", flush=True)
    minutes = round((time.time() - t0) / 60, 1)
    if te is None:
        M.save(model.trainable_state(), MODELS / a.name, target=TARGET_PID, tau=TAU, training={
            "items": len(tr), "extra_labels": int(is_extra.sum()), "epochs": a.epochs, "batch_size": a.batch_size,
            "lr": a.lr, "text_dropout": a.text_dropout, "minutes": minutes, "device": dev,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        print(f"saved {MODELS / a.name} after {minutes} min", flush=True)
        return
    (RESULTS / "oof").mkdir(parents=True, exist_ok=True)
    off, ext = te[~is_extra[te]], te[is_extra[te]]
    row = {"model": a.name, "split": a.split, "fold": a.fold, "epochs": a.epochs, "text_dropout": a.text_dropout,
           "labels": a.labels, "n_test": len(off), "n_extra_test": len(ext), "minutes": minutes}
    for variant, inputs in (("", text), ("no_article_", bare)):  # same test items with the Wikipedia line removed
        p = pd.DataFrame(M.predict(model, [inputs[k] for k in te]), index=te, columns=[f"p{b}" for b in BUCKETS])
        p.set_index(qids[te]).assign(y=y[te], extra=is_extra[te]) \
            .to_parquet(RESULTS / "oof" / f"{a.name}_{variant}{a.split}_f{a.fold}.parquet")
        for prefix, idx in (("", off), ("extra_", ext)):
            if len(idx):
                row |= {f"{variant}{prefix}{k}": v for k, v in metrics(y[idx], p.loc[idx].values.argmax(1)).items()}
    print(row, flush=True)
    out = RESULTS / "eval.csv"
    pd.concat([pd.read_csv(out) if out.exists() else pd.DataFrame(), pd.DataFrame([row])]).to_csv(out, index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fold", type=int)
    p.add_argument("--split", choices=["random", "group"], default="random")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--text-dropout", type=float, default=0.5)
    p.add_argument("--labels", help="extra labels without an official rating: parquet or CSV with qid, age")
    p.add_argument("--name", default=MODEL_NAME)
    main(p.parse_args())

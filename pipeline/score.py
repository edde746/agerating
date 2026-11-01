"""Score every universe item without an official FSK rating.

Usage: python score.py [--model DIR] [--batch-size 32]
Writes data/serial_univ.parquet (model input per qid) and data/pred/<model name>.parquet (p0 p6 p12 p16 p18 per qid).
About 80 min for 578k items on an RTX 4090.
"""
import argparse, json, time
from pathlib import Path
import pandas as pd
import data as D
import model as M
from common import BUCKETS, DATA, MODEL_NAME, MODELS, TARGET_PID
from serialize import serialize, facts


def main(a):
    t0 = time.time()
    path = Path(a.model) if a.model else MODELS / MODEL_NAME
    name = json.loads((path / "config.json").read_text())["name"]
    st = facts()[0]
    off = D.official(st, TARGET_PID)
    q = [x for x in D.load_items().query("universe").qid if x not in off.index]
    s = serialize(q)
    s.to_frame().rename_axis("qid").to_parquet(DATA / "serial_univ.parquet")
    print(f"{len(s)} universe items to score, chars {s.str.len().describe().round(0).to_dict()} "
          f"({time.time() - t0:.0f}s)", flush=True)
    p = M.predict(M.load(path), s.tolist(), bs=a.batch_size)
    (DATA / "pred").mkdir(exist_ok=True)
    pd.DataFrame(p, index=s.index, columns=[f"p{b}" for b in BUCKETS]).to_parquet(DATA / "pred" / f"{name}.parquet")
    print(f"scored {len(s)} items in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model")
    p.add_argument("--batch-size", type=int, default=32)
    main(p.parse_args())

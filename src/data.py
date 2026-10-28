"""Load the downloaded EMBER jsonl shards into numpy arrays (cached as .npz)."""
import glob, json, os

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW, CACHE = os.path.join(ROOT, "data", "raw"), os.path.join(ROOT, "data", "processed.npz")


def slog(x):  # signed log1p: tames file sizes / timestamps for the linear model and the MLP
    return np.sign(x) * np.log1p(np.abs(x))


def _read(split):
    X, y, meta = [], [], []
    for path in sorted(glob.glob(os.path.join(RAW, f"ember2018_{split}_*.jsonl"))):
        with open(path) as fh:
            for line in fh:
                d = json.loads(line)
                if d["y"] < 0:  # unlabeled rows of the EMBER train set
                    continue
                X.append(d["x"]); y.append(int(d["y"]))
                meta.append((d["sha256"], d["appeared"], d.get("avclass") or ""))
    return np.asarray(X, np.float32), np.asarray(y, np.int8), np.asarray(meta, dtype=object)


def load():
    """-> dict(Xtr, ytr, mtr, Xte, yte, mte); meta rows are (sha256, month_first_seen, avclass_family)."""
    if os.path.exists(CACHE):
        return dict(np.load(CACHE, allow_pickle=True))
    out = {}
    for split, k in (("train", "tr"), ("test", "te")):
        out[f"X{k}"], out[f"y{k}"], out[f"m{k}"] = _read(split)
    np.savez_compressed(CACHE, **out)
    return out


if __name__ == "__main__":
    d = load()
    for k in ("tr", "te"):
        print(k, d[f"X{k}"].shape, "malware share", d[f"y{k}"].mean().round(3))

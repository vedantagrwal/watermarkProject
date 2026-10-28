"""Download a small, evenly spread sample of the EMBER-2018 feature shards (HuggingFace mirror cw1521/ember2018).

Each shard is ~2000 PE files x 2381 EMBER features (jsonl, ~70 MB). Re-running skips complete files.
"""
import os, subprocess
from concurrent.futures import ThreadPoolExecutor

BASE = "https://huggingface.co/datasets/cw1521/ember2018/resolve/main/data/ember2018_{}.jsonl"
TRAIN = [1, 35, 70, 105, 150, 185, 220, 300]
TEST = [1, 30, 60]
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def fetch(name):
    url, path = BASE.format(name), os.path.join(OUT, f"ember2018_{name}.jsonl")
    if os.path.exists(path) and not os.path.exists(path + ".part"):
        return f"ok   {name}"
    # curl (not urllib): python.org builds of macOS Python ship without CA certificates
    subprocess.run(["curl", "-fsSL", "-C", "-", "-o", path + ".part", url], check=True)
    os.replace(path + ".part", path)
    return f"got  {name}"


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    names = [f"test_{i}" for i in TEST] + [f"train_{i}" for i in TRAIN]
    with ThreadPoolExecutor(4) as ex:
        for msg in ex.map(fetch, names):
            print(msg, flush=True)

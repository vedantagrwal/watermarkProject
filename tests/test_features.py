"""Smoke test: python tests/test_features.py  (fails loudly if the EMBER column layout drifts)."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from features import N_FEATURES, family_slices, feature_names, hash_bin

names = feature_names()
assert len(names) == N_FEATURES == 2381 and len(set(names)) == 2381, "names must be 2381 and unique"
assert sorted(c for cols in family_slices().values() for c in cols) == list(range(2381))
assert 0 <= hash_bin(1024, "kernel32.dll:VirtualAllocEx") < 1024
assert names[0xE8].endswith("(CALL rel32)")
print("ok")

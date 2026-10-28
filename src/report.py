"""Human-readable malware report for one test sample: verdict + SHAP reasons in plain English + LIME cross-check.

    python src/report.py --n 3            # 3 detected malware + 1 benign, chosen at random
    python src/report.py --index 123      # a specific row of the EMBER test sample
"""
import argparse, os, re
from datetime import datetime, timezone

import matplotlib
import numpy as np
import shap

from data import ROOT, load
from explain import NAMES, FAMS, jaccard, lime_explain, load_booster, shap_values, top_k

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BEHAVIOUR = {
    "VirtualAllocEx": "allocates memory inside another process (code injection)",
    "WriteProcessMemory": "writes into another process (code injection)",
    "CreateRemoteThread": "starts a thread in another process (code injection)",
    "ReadProcessMemory": "reads another process's memory (credential/data theft)",
    "VirtualProtect": "changes memory permissions (unpacking / shellcode)",
    "VirtualAlloc": "allocates executable memory (unpacking / shellcode)",
    "GetProcAddress": "resolves APIs at run time (hides imports)",
    "LoadLibraryA": "loads DLLs dynamically", "LoadLibraryW": "loads DLLs dynamically",
    "IsDebuggerPresent": "anti-debugging check", "CheckRemoteDebuggerPresent": "anti-debugging check",
    "URLDownloadToFileA": "downloads a file from the internet (dropper)", "URLDownloadToFileW": "downloads a file (dropper)",
    "InternetOpenA": "opens an internet session (network C2/download)", "InternetReadFile": "reads data from the network",
    "HttpSendRequestA": "sends HTTP requests (C2 traffic)", "WinExec": "executes another program",
    "ShellExecuteA": "launches programs/URLs", "CreateProcessA": "spawns processes",
    "RegSetValueExA": "writes registry values (persistence/config)", "RegCreateKeyExA": "creates registry keys (persistence)",
    "CreateServiceA": "installs a Windows service (persistence)", "CryptEncrypt": "encrypts data (ransomware / C2)",
    "SetWindowsHookExA": "installs a hook (keylogging)", "GetAsyncKeyState": "polls key state (keylogging)",
    "CreateToolhelp32Snapshot": "enumerates processes", "AdjustTokenPrivileges": "elevates privileges",
    "WSAStartup": "initialises Winsock networking", "connect": "opens a network connection", "socket": "creates a socket",
}


def describe(i, x, s):
    name, v = NAMES[i], x[i]
    m = re.search(r"\(~([^)]*)\)", name)
    if name.startswith("imp_api") and m:
        fns = [h.split(":", 1)[-1] for h in m.group(1).split("|")]
        why = next((BEHAVIOUR[f] for f in fns if f in BEHAVIOUR), "")
        return f"imports a function hashing like `{m.group(1)}`" + (f" - {why}" if why else "") + f" (hash-bin value {v:+.0f})"
    if name.startswith("imp_lib") and m:
        return f"links against a library hashing like `{m.group(1)}` (hash-bin value {v:+.0f})"
    if name == "hdr_timestamp":
        try:
            when = f"{datetime.fromtimestamp(float(v), timezone.utc):%Y-%m-%d}"
        except (OverflowError, OSError, ValueError):
            when = "an impossible date"
        return f"claims a compile timestamp of {when} (forged or zeroed stamps are common in malware)"
    if name.startswith("byte_0x"):
        return f"contains byte `{name[5:]}` at {v:.2%} of its bytes (opcode-frequency proxy)"
    if name.startswith("entropy_bin"):
        return f"has {v:.2%} of its windows in entropy bucket {name[11:13]} / high-nibble {name[-1]} (packed or encrypted data is high-entropy)"
    return f"has `{name}` = {v:.4g}"


def make_report(idx, d, booster, bg, sel, out_dir):
    x, y = d["Xte"][idx], int(d["yte"][idx])
    sha, month, family = d["mte"][idx]
    p = float(booster.predict(x[None])[0])
    sv = shap_values(booster, x[None])[0]
    ev = np.ravel(shap.TreeExplainer(booster).expected_value)
    base = float(ev[-1])
    order = np.argsort(-np.abs(sv))[:10]
    lw, _ = lime_explain(bg, x, booster.predict, sel)
    shap_top = top_k({int(i): sv[i] for i in np.argsort(-np.abs(sv))[:20]}, 10)
    shared = len(set(shap_top) & set(top_k(lw, 10)))
    agree = jaccard(shap_top, top_k(lw, 10))
    risk = "CRITICAL" if p > .95 else "HIGH" if p > .8 else "MEDIUM" if p > .5 else "LOW"
    fam = {f: np.abs(sv[FAMS == f]).sum() / np.abs(sv).sum() for f in dict.fromkeys(FAMS)}

    plt.figure()
    shap.plots.waterfall(shap.Explanation(sv, base, x, feature_names=[n[:48] for n in NAMES]), max_display=10, show=False)
    plt.tight_layout(); png = f"{sha[:12]}_waterfall.png"
    plt.savefig(os.path.join(out_dir, png), dpi=130, bbox_inches="tight"); plt.close("all")

    L = [f"# Malware analysis report - `{sha[:16]}…`", "",
         f"| | |\n|---|---|\n| **Verdict** | {'MALWARE' if p > .5 else 'BENIGN'} |\n| **Malware probability** | {p:.1%} |\n"
         f"| **Risk level** | {risk} |\n| **First seen** | {month} |\n| **Ground-truth label** | {'malware' if y else 'benign'}"
         f"{f' (AV family: {family})' if family not in ('', 'nan', 'None') else ''} |", "",
         "## Why the model decided this (SHAP)", "",
         "Positive = pushes towards *malware*, negative = towards *benign* (log-odds units).", ""]
    for r, i in enumerate(order, 1):
        L.append(f"{r}. **{'+' if sv[i] > 0 else '-'}{abs(sv[i]):.2f}** - the file {describe(i, x, sv[i])}  _[{FAMS[i]}]_")
    L += ["", "## Evidence by feature family", "", "| Family | Share of the decision |\n|---|---|"]
    L += [f"| {f} | {s:.0%} |" for f, s in sorted(fam.items(), key=lambda t: -t[1])]
    L += ["", "## Cross-check with LIME", "",
          f"LIME (a different, model-agnostic method, run on the 60 most important features) picks {shared} of the same top-10 features as SHAP "
          f"(Jaccard overlap {agree:.2f}). LIME's strongest malware-pushing features:", ""]
    L += [f"- {NAMES[i]} (weight {lw[i]:+.3f})" for i in top_k(lw, 5, positive_only=True)]
    L += ["", f"![SHAP waterfall]({png})", "",
          "_Caveat: the hash-decoded names (`~…`) are hints - different APIs can collide in the same hash bin._"]
    path = os.path.join(out_dir, f"{sha[:12]}.md")
    open(path, "w").write("\n".join(L) + "\n")
    return path, p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, nargs="*")
    ap.add_argument("--n", type=int, default=3, help="number of detected malware samples (+1 benign) when --index is not given")
    a = ap.parse_args()
    d, booster = load(), load_booster()
    rng = np.random.default_rng(1)
    p = booster.predict(d["Xte"])
    idxs = a.index or list(rng.choice(np.where((d["yte"] == 1) & (p > .5))[0], a.n, replace=False)) + \
        list(rng.choice(np.where((d["yte"] == 0) & (p < .5))[0], 1))
    out = os.path.join(ROOT, "reports"); os.makedirs(out, exist_ok=True)
    bg = d["Xtr"][rng.choice(len(d["Xtr"]), 3000, replace=False)]
    sel = np.argsort(-booster.feature_importance("gain"))[:60]
    for i in idxs:
        path, pr = make_report(int(i), d, booster, bg, sel, out)
        print(f"{path}  p(malware)={pr:.3f}")

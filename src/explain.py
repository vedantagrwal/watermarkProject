"""Explain the LightGBM detector with SHAP and LIME, and *measure* how good the explanations are:
  - global SHAP importance (per feature and per feature family)
  - SHAP-vs-LIME agreement, LIME run-to-run stability
  - deletion test (faithfulness): neutralise the top-k features an explainer names and see how far the malware score falls
  - LIME applied to the MLP (explainers are model-agnostic, so the deep-learning black box can be opened the same way)
"""
import json, os

import joblib, lightgbm as lgb, matplotlib
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer

from data import ROOT, load
from features import feature_families, feature_names

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES, MOD = os.path.join(ROOT, "results"), os.path.join(ROOT, "models")
NAMES, FAMS = feature_names(), np.array(feature_families())


def load_booster():
    return lgb.Booster(model_file=os.path.join(MOD, "lgbm.txt"))


def shap_values(booster, X):
    sv = shap.TreeExplainer(booster).shap_values(X)
    return sv[1] if isinstance(sv, list) else sv  # log-odds contribution towards "malware"


def lime_explain(bg, x, predict, sel=None, k=20, n=2000, seed=0):
    """LIME on columns `sel` only (all others frozen at x's values); sel=None -> all 2381 columns.
    -> ({column: weight towards malware}, R^2 of the local surrogate)"""
    cols = np.arange(len(x)) if sel is None else np.asarray(sel)

    def f(Z):
        M = np.tile(x, (len(Z), 1)); M[:, cols] = Z
        p = predict(M)
        return np.c_[1 - p, p]
    ex = LimeTabularExplainer(bg[:, cols], feature_names=[NAMES[c] for c in cols], class_names=["benign", "malware"],
                              mode="classification", discretize_continuous=True, random_state=seed)
    ex = ex.explain_instance(x[cols], f, num_features=k, num_samples=n, labels=(1,))
    return {int(cols[j]): w for j, w in ex.as_map()[1]}, ex.score


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def top_k(weights, k, positive_only=False):
    idx = sorted(weights, key=lambda i: -weights[i] if positive_only else -abs(weights[i]))
    return idx[:k]


def main():
    d = load()
    Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    rng = np.random.default_rng(0)
    booster = load_booster()
    predict = lambda X: booster.predict(X)

    # ---- global SHAP on a random test subset ----
    sub = rng.choice(len(Xte), 1500, replace=False)
    Xs, sv = Xte[sub], shap_values(booster, Xte[sub])
    imp = np.abs(sv).mean(0)
    order = np.argsort(-imp)
    fam_share = {f: float(imp[FAMS == f].sum() / imp.sum()) for f in dict.fromkeys(FAMS)}
    with open(os.path.join(RES, "top_features.csv"), "w") as fh:
        fh.write("rank,feature,family,mean_abs_shap\n")
        for r, i in enumerate(order[:50], 1):
            fh.write(f'{r},"{NAMES[i]}",{FAMS[i]},{imp[i]:.5f}\n')

    plt.figure(figsize=(8, 6))
    top = order[:20][::-1]
    plt.barh([NAMES[i][:55] for i in top], imp[top], color="#2a6f97")
    plt.xlabel("mean |SHAP| (log-odds)"); plt.title("Global SHAP importance - top 20 EMBER features")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "shap_global.png"), dpi=150); plt.close()

    plt.figure(figsize=(5.5, 3.2))
    plt.barh(list(fam_share)[::-1], [fam_share[f] for f in list(fam_share)[::-1]], color="#8a5a44")
    plt.xlabel("share of total mean |SHAP|"); plt.title("Which feature family drives decisions?")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "shap_family.png"), dpi=150); plt.close()

    shap.summary_plot(sv, Xs, feature_names=NAMES, max_display=15, show=False, plot_size=(9, 6))
    plt.tight_layout(); plt.savefig(os.path.join(RES, "shap_beeswarm.png"), dpi=150); plt.close()

    # ---- SHAP vs LIME on correctly-detected malware ----
    # Vanilla LIME over all 2381 sparse columns fits a useless surrogate (R^2 ~ 0), so we also run it restricted to
    # the K columns with the highest LightGBM gain (everything else frozen at the file's own values).
    K = 60
    sel = np.argsort(-booster.feature_importance("gain"))[:K]
    p = predict(Xte)
    mal = rng.permutation(np.where((yte == 1) & (p > .5))[0])[:40]
    bg = Xtr[rng.choice(len(Xtr), 3000, replace=False)]
    sv_mal = shap_values(booster, Xte[mal])
    shap_top = [{int(c): sv_mal[r, c] for c in np.argsort(-np.abs(sv_mal[r]))[:20]} for r in range(len(mal))]

    sens = {}  # sensitivity to the size of the LIME feature subset (first 15 files; full space is slow)
    for tag, cols in (("all 2381", None), ("top 30", sel[:30]), ("top 60", sel), ("top 100", np.argsort(-booster.feature_importance("gain"))[:100])):
        out = [lime_explain(bg, Xte[i], predict, cols) for i in mal[:15]]
        sens[tag] = dict(jaccard=float(np.mean([jaccard(top_k(shap_top[r], 10), top_k(w, 10)) for r, (w, _) in enumerate(out)])),
                         surrogate_r2=float(np.mean([r2 for _, r2 in out])))
        print(f"LIME on {tag:9s} vs SHAP: top-10 Jaccard {sens[tag]['jaccard']:.3f}  surrogate R2 {sens[tag]['surrogate_r2']:.3f}")

    lw, jac, sign_agree, stab, r2s = {}, [], [], [], []
    for r, i in enumerate(mal):
        lw[i], r2 = lime_explain(bg, Xte[i], predict, sel)
        r2s.append(r2)
        jac.append(jaccard(top_k(shap_top[r], 10), top_k(lw[i], 10)))
        shared = set(top_k(shap_top[r], 10)) & set(top_k(lw[i], 10))
        if shared:
            sign_agree.append(np.mean([np.sign(shap_top[r][c]) == np.sign(lw[i][c]) for c in shared]))
        if r < 15:  # LIME re-run with another random seed
            stab.append(jaccard(top_k(lw[i], 10), top_k(lime_explain(bg, Xte[i], predict, sel, seed=1)[0], 10)))
    print(f"LIME(top-{K}) vs SHAP top-10 Jaccard {np.mean(jac):.3f} | sign agreement {np.mean(sign_agree):.3f} | "
          f"LIME seed-stability {np.mean(stab):.3f} | surrogate R2 {np.mean(r2s):.3f}")

    # ---- deletion test on malware: neutralise top-k named features, measure score drop ----
    neutral = np.median(Xtr[ytr == 0], axis=0)

    def drop(feats_per_case, k):
        Xm = Xte[mal].copy()
        for r, feats in enumerate(feats_per_case):
            Xm[r, feats[:k]] = neutral[feats[:k]]
        return float((predict(Xte[mal]) - predict(Xm)).mean())
    methods = {
        "SHAP (top-k pushing to malware)": [list(np.argsort(-sv_mal[r])) for r in range(len(mal))],
        f"LIME top-{K} (top-k pushing to malware)": [top_k(lw[i], 20, positive_only=True) for i in mal],
        "Global importance (same k for all files)": [list(order)] * len(mal),
        "Random features": [list(rng.permutation(len(NAMES))) for _ in mal],
    }
    deletion = {m: {k: round(drop(f, k), 4) for k in (5, 10, 20)} for m, f in methods.items()}
    for m, v in deletion.items():
        print(f"deletion {m:42s}", v)

    # ---- LIME on the MLP black box ----
    mlp = joblib.load(os.path.join(MOD, "mlp.joblib"))
    mlp_p = lambda X: mlp.predict_proba(X)[:, 1]
    mlp_jac = [jaccard(top_k(lime_explain(bg, Xte[i], mlp_p, sel)[0], 10), top_k(lw[i], 10)) for i in mal[:15]]
    print(f"LIME(MLP) vs LIME(LightGBM) top-10 Jaccard {np.mean(mlp_jac):.3f}")

    out = dict(family_share=fam_share, lime_subset_size=K, shap_lime_jaccard_top10=float(np.mean(jac)),
               shap_lime_sign_agreement=float(np.mean(sign_agree)), lime_stability_top10=float(np.mean(stab)),
               lime_surrogate_r2=float(np.mean(r2s)), lime_subset_sensitivity=sens, deletion_mean_score_drop=deletion,
               lime_mlp_vs_lgbm_jaccard=float(np.mean(mlp_jac)), n_explained=dict(malware=len(mal), shap_global=len(sub)),
               top10=[(NAMES[i], FAMS[i], round(float(imp[i]), 4)) for i in order[:10]])
    json.dump(out, open(os.path.join(RES, "explain_metrics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()

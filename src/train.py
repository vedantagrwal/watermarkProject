"""Train + evaluate: logistic regression (interpretable baseline), MLP (deep-learning black box), LightGBM (main),
plus a feature-family ablation (PE header / opcode proxy / API calls / strings)."""
import json, os

import joblib, lightgbm as lgb, matplotlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from data import ROOT, load, slog
from features import family_slices

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES, MOD = os.path.join(ROOT, "results"), os.path.join(ROOT, "models")
SEED = 0
LGB_PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=31, colsample_bytree=0.3, subsample=0.8,
                  subsample_freq=1, min_child_samples=20, random_state=SEED, n_jobs=-1, verbose=-1)


def make_lr():
    return make_pipeline(FunctionTransformer(slog), StandardScaler(), LogisticRegression(C=0.05, max_iter=500))


def make_mlp():
    return make_pipeline(FunctionTransformer(slog), StandardScaler(),
                         MLPClassifier((256, 128), alpha=1e-3, early_stopping=True, max_iter=80, random_state=SEED))


def metrics(y, p):
    out = dict(auc=roc_auc_score(y, p), acc=accuracy_score(y, p > .5), f1=f1_score(y, p > .5),
               precision=precision_score(y, p > .5), recall=recall_score(y, p > .5))
    fpr, tpr, _ = roc_curve(y, p)
    for t in (0.01, 0.001):  # EMBER's headline metric: detection rate at a fixed false-positive rate
        out[f"tpr@{t:.1%}fpr"] = float(np.interp(t, fpr, tpr))
    tn, fp, fn, tp = confusion_matrix(y, p > .5).ravel()
    out["confusion"] = dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def main():
    d = load()
    Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    print(f"train {Xtr.shape} malware {ytr.mean():.2%} | test {Xte.shape} malware {yte.mean():.2%}")
    res, curves = {}, {}

    models = {"Logistic regression": make_lr(), "MLP (deep-learning baseline)": make_mlp(),
              "LightGBM (proposed)": lgb.LGBMClassifier(**LGB_PARAMS)}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        res[name] = metrics(yte, p); curves[name] = roc_curve(yte, p)[:2]
        print(f"{name:30s} AUC {res[name]['auc']:.4f}  acc {res[name]['acc']:.4f}  TPR@1%FPR {res[name]['tpr@1.0%fpr']:.4f}")
    joblib.dump(models["Logistic regression"], os.path.join(MOD, "logreg.joblib"))
    joblib.dump(models["MLP (deep-learning baseline)"], os.path.join(MOD, "mlp.joblib"))
    models["LightGBM (proposed)"].booster_.save_model(os.path.join(MOD, "lgbm.txt"))

    # ablation: which feature family carries the signal?
    fam, abl = family_slices(), {}
    for f, cols in fam.items():
        for tag, use in ((f"only {f}", cols), (f"without {f}", [c for c in range(Xtr.shape[1]) if c not in set(cols)])):
            m = lgb.LGBMClassifier(**LGB_PARAMS).fit(Xtr[:, use], ytr)
            abl[tag] = metrics(yte, m.predict_proba(Xte[:, use])[:, 1])
            print(f"ablation {tag:40s} AUC {abl[tag]['auc']:.4f}")
    res["_ablation"] = abl
    res["_data"] = dict(n_train=int(len(ytr)), n_test=int(len(yte)), train_malware=float(ytr.mean().round(4)),
                        test_malware=float(yte.mean().round(4)), n_features=int(Xtr.shape[1]))
    json.dump(res, open(os.path.join(RES, "metrics.json"), "w"), indent=2)

    plt.figure(figsize=(5.5, 4.5))
    for name, (fpr, tpr) in curves.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC {res[name]['auc']:.3f})")
    plt.xscale("log"); plt.xlim(1e-3, 1); plt.xlabel("False-positive rate (log)"); plt.ylabel("True-positive rate")
    plt.title("ROC on held-out EMBER test months"); plt.legend(loc="lower right", fontsize=8); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(RES, "roc.png"), dpi=150)


if __name__ == "__main__":
    main()

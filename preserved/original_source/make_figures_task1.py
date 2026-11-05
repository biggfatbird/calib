import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
s = pd.read_csv("results/noise_draw_summary.csv")
col = {"Logistic regression": "#0072B2", "Random forest": "#D55E00", "RBF SVM": "#009E73"}; mk = {"Logistic regression": "o", "Random forest": "s", "RBF SVM": "^"}
fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
for ax, (met, lab) in zip(axs, [("roc_auc", "ROC-AUC"), ("sensitivity", "Sensitivity (malignant)"), ("brier", "Brier score (lower better)"), ("ece10", "ECE, 10 bins (lower better)")]):
    for m, c in col.items():
        d = s[(s.model == m) & (s.metric == met)].sort_values("sigma")
        ax.plot(d.sigma, d["mean"], color=c, marker=mk[m], label=m); ax.fill_between(d.sigma, d.draw_lo, d.draw_hi, color=c, alpha=.18, lw=0)
    ax.set_xlabel("Noise sigma (x training-feature SD)"); ax.set_title(lab, fontsize=10); ax.grid(alpha=.25)
axs[0].legend(frameon=False, fontsize=8)
fig.suptitle("Mean over 5 splits; bands = 95% interval across 50 independent noise draws (n=143 test rows per split)", fontsize=9, y=1.0)
fig.tight_layout(); fig.savefig("figures/noise_draws_ci.png", dpi=160)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("figs", exist_ok=True)   # figures are written here

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.linewidth": 0.6,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
W = 3.45   # IEEE single-column width in inches

RED, BLUE, GREEN, GREY, ORANGE = "#B03A2E", "#1F5A96", "#2E7D4F", "#8A8A8A", "#D9822B"
TIERS = ["LR", "DT", "GBDT-S", "GBDT-M", "GBDT-M+", "GBDT-L", "TFIDF-LR"]

# ---- deduplicated, mean and sd over three split seeds (Cell 16/17) ----
auc   = dict(zip(TIERS, [0.80333, 0.78793, 0.91407, 0.94233, 0.95157, 0.96667, 0.98810]))
r_lo  = dict(zip(TIERS, [0.40297, 0.48837, 0.61007, 0.66847, 0.70300, 0.68557, 0.91770]))   # 1% FPR
s_lo  = dict(zip(TIERS, [0.0030, 0.0022, 0.0196, 0.0090, 0.0138, 0.0150, 0.0168]))
r_st  = dict(zip(TIERS, [0.27957, 0.41027, 0.425501, 0.56593, 0.59507, 0.53943, 0.29453]))   # 0.1% FPR
s_st  = dict(zip(TIERS, [0.0158, 0.0086, 0.0422, 0.0274, 0.0606, 0.0481, 0.0183]))

def spread(vals, gap):
    """Nudge label y-positions apart so they don't overlap; keeps order."""
    order = np.argsort(vals); out = np.array(vals, float)
    for _ in range(200):
        moved = False
        for a, b in zip(order[:-1], order[1:]):
            if out[b] - out[a] < gap:
                mid = (out[a] + out[b]) / 2
                out[a], out[b] = mid - gap / 2, mid + gap / 2; moved = True
        if not moved: break
    return out

# ================= Fig. 3: slope chart, rank reversal =================
fig, ax = plt.subplots(figsize=(W, 3.0))
xl, xr = 0.0, 1.0
colors = {t: GREY for t in TIERS}; colors["TFIDF-LR"] = RED; colors["GBDT-M"] = BLUE
widths = {t: 0.9 for t in TIERS}; widths["TFIDF-LR"] = 1.8; widths["GBDT-M"] = 1.8
for t in TIERS:
    ax.plot([xl, xr], [r_lo[t], r_st[t]], color=colors[t], lw=widths[t], zorder=3 if colors[t] != GREY else 2)
    ax.errorbar([xl, xr], [r_lo[t], r_st[t]], yerr=[s_lo[t], s_st[t]], fmt="o", ms=3.2,
                color=colors[t], elinewidth=0.7, capsize=1.6, zorder=4)
yl = spread([r_lo[t] for t in TIERS], 0.034)
yr = spread([r_st[t] for t in TIERS], 0.034)
for t, y in zip(TIERS, yl):
    ax.text(xl - 0.05, y, f"{t}  {r_lo[t]:.3f}   AUC {auc[t]:.3f}", ha="right", va="center", fontsize=6.8,
            color=colors[t] if colors[t] != GREY else "#444444",
            fontweight="bold" if colors[t] != GREY else "normal")
for t, y in zip(TIERS, yr):
    ax.text(xr + 0.05, y, f"{r_st[t]:.3f}  {t}", ha="left", va="center", fontsize=6.8,
            color=colors[t] if colors[t] != GREY else "#444444",
            fontweight="bold" if colors[t] != GREY else "normal")
ax.set_xlim(-1.42, 1.78); ax.set_ylim(0.22, 0.97)
ax.set_xticks([xl, xr]); ax.set_xticklabels(["1% FPR\n(review)", "0.1% FPR\n(block)"])
ax.set_ylabel("")
for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
ax.spines["bottom"].set_bounds(xl, xr)
ax.yaxis.set_ticks_position("none"); ax.set_yticks([])
ax.grid(False)
fig.savefig("figs/fig3_rank_reversal.png"); fig.savefig("figs/fig3_rank_reversal.pdf"); plt.close(fig)

# ================= Fig. 4: leakage across split protocols =================
rand = [0.2653, 0.4246, 0.5968, 0.7382, 0.7647, 0.7319, 0.7014]
rand_s = [0.0023, 0.0097, 0.0078, 0.0135, 0.0091, 0.0059, 0.0177]
ded = [r_st[t] for t in TIERS]; ded_s = [s_st[t] for t in TIERS]
endp = [0.2198, 0.1972, 0.2843, 0.5186, 0.4591, 0.4961, 0.5316]
endp_s = [0.0495, 0.0709, 0.0830, 0.3633, 0.2374, 0.3442, 0.2637]

fig, ax = plt.subplots(figsize=(W, 2.35))
x = np.arange(len(TIERS)); bw = 0.27
kw = dict(capsize=1.4, error_kw=dict(elinewidth=0.6, capthick=0.6), edgecolor="white", linewidth=0.3)
ax.bar(x - bw, rand, bw, yerr=rand_s, color="#C9C9C9", label="Random split (duplicates kept)", **kw)
ax.bar(x,       ded,  bw, yerr=ded_s,  color=BLUE,      label="Exact deduplication", **kw)
ax.bar(x + bw,  endp, bw, yerr=endp_s, color=ORANGE,    label="Endpoint held out", **kw)
ax.set_xticks(x); ax.set_xticklabels(TIERS, rotation=28, ha="right")
ax.set_ylabel("Recall at 0.1% FPR"); ax.set_ylim(0, 1.0)
ax.legend(loc="upper left", frameon=False, ncol=1, handlelength=1.2, borderaxespad=0.2)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.yaxis.grid(True, lw=0.3, color="#DDDDDD"); ax.set_axisbelow(True)
fig.savefig("figs/fig4_leakage_protocols.png"); fig.savefig("figs/fig4_leakage_protocols.pdf"); plt.close(fig)

# ================= Fig. 5: latency across two sessions =================
rows = [  # label, node visits text, session A p99, session B p99
    ("TFIDF-LR", "no trees", 1698.5, 1019.1),
    ("GBDT-L",   "10,939 nodes", 448.4, 232.6),
    ("GBDT-M+",  "1,160 nodes", 340.6, 217.4),
    ("GBDT-M",   "1,111 nodes", 204.3, 137.7),
    ("GBDT-S",   "184 nodes", 182.9, 131.7),
    ("DT",       "5 nodes", 262.3, 186.6),
    ("LR",       "no trees", 396.9, 280.3),
]
fig, ax = plt.subplots(figsize=(W, 2.45))
ys = np.arange(len(rows))[::-1]
for y, (lab, nv, a, b) in zip(ys, rows):
    ax.annotate("", xy=(b, y), xytext=(a, y),
                arrowprops=dict(arrowstyle="-|>", color="#9A9A9A", lw=0.8, shrinkA=3, shrinkB=3,
                                mutation_scale=6))
    ax.plot(a, y, "o", color="#303030", ms=3.8, zorder=3)
    ax.plot(b, y, "o", mfc="white", mec="#303030", ms=3.8, mew=0.9, zorder=3)
ax.set_yticks(ys); ax.set_yticklabels([f"{r[0]}  ({r[1]})" for r in rows])
ax.set_xscale("log"); ax.set_xlim(100, 2300)
ax.set_xticks([100, 250, 500, 1000, 2000]); ax.set_xticklabels(["100", "250", "500", "1000", "2000"])
ax.set_xlabel("99th-percentile latency per request (\u00b5s, log scale)")
for D, lab in [(250, "250 \u00b5s deadline"), (1000, "1 ms deadline")]:
    ax.axvline(D, ls="--", lw=0.7, color=RED, alpha=0.8, zorder=1)
    ax.text(D * 1.03, ys[0] + 0.55, lab, color=RED, fontsize=6.5, va="bottom")
ax.set_ylim(-0.6, len(rows) - 0.1)
ax.plot([], [], "o", color="#303030", ms=3.8, label="Session A")
ax.plot([], [], "o", mfc="white", mec="#303030", ms=3.8, label="Session B")
ax.legend(loc="lower right", frameon=False, handletextpad=0.3, borderaxespad=0.1)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.xaxis.grid(True, which="major", lw=0.3, color="#E3E3E3"); ax.set_axisbelow(True)
fig.savefig("figs/fig5_latency_sessions.png"); fig.savefig("figs/fig5_latency_sessions.pdf"); plt.close(fig)
print("results figures written")

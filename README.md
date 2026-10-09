# Green Decision Budgets for Inline E-Commerce Web Attack Detection

Code, results and figures for the paper of the same name.

The study evaluates seven web attack detectors on the CSIC 2010 e-commerce dataset
under the constraints an inline filter actually works under: a very low false-positive
rate, a decision deadline, and a compute cost paid on every request. It introduces no
new detector. It asks how existing ones should be compared.

Everything here runs on a free Google Colab CPU runtime. No GPU, no paid service and no
credentials are needed.

---

## What the study found

1. The model with the highest AUC (a character n-gram TF-IDF model) is among the weakest
   at a 0.1% false-positive rate, and tier rankings change with the operating point.
2. At the strict threshold, differences between mid-sized models do not hold across
   random splits, and single-split bootstrap intervals hide that instability.
3. Exact duplicates make up 58.1% of CSIC 2010. Random splits that keep them put 59.8%
   of test requests word for word in the training set and inflate strict-threshold
   recall by 22.4% on average.
4. Latency-based deadline verdicts for the same code changed between two Colab sessions,
   while the hardware-independent work proxy did not.

---

## Repository layout

```
code/
  01_main_experiment.ipynb        main notebook: download, parse, features, 7 tiers,
                                  matched-FPR evaluation, latency sweeps, figures
  02_duplicate_audit.py           add-on cell: duplicate audit, with vs without duplicates
  03_seed_stability.py            add-on cell: three split seeds, claim stability
  04_leakage_and_deployment.py    add-on cell: three split protocols, deployment table
  05_make_result_figures.py       redraws Figures 3 to 5 from the final numbers
results/
  table2_detection_exact_dedup.csv       paper Table II
  protocol_sensitivity_raw.csv           every protocol x split x tier result
  leakage_sensitivity.csv                paper Figure 4, with relative inflation
  timing_two_sessions.csv                paper Figure 5, both measurement sessions
  claim_stability_paired_bootstrap.csv   paired-bootstrap verdict for each tier pair
figures/
  fig3_rank_reversal.*                   recall at 1% and 0.1% FPR
  fig4_leakage_protocols.*               recall under the three split protocols
  fig5_latency_sessions.*                p99 latency in both sessions
  Methodology_Picture_1.pdf              Figure 1, inline decision path
  Methodology_Picture_2.pdf              Figure 2, evaluation protocol
logs/
  run_*.txt                              raw console output of the reported runs
data/
  README.md                              how to obtain CSIC 2010 (not redistributed here)
```

## Which file backs which claim

| Paper item | File |
| --- | --- |
| Table I, operations per request | `results/table2_detection_exact_dedup.csv`, column `ops_per_request` |
| Table II, detection quality | `results/table2_detection_exact_dedup.csv` |
| Figure 3, rank reversal | `results/table2_detection_exact_dedup.csv` |
| Figure 4, split protocols | `results/leakage_sensitivity.csv` |
| Figure 5, two sessions | `results/timing_two_sessions.csv` |
| Split variance, Section IV-B | `results/protocol_sensitivity_raw.csv`, protocol `exact_dedup` |
| Endpoint-held-out results | `results/protocol_sensitivity_raw.csv`, protocol `endpoint_heldout` |
| Which comparisons are reported | `results/claim_stability_paired_bootstrap.csv` |

Tier names map to the code as follows: LR = `T1_logreg`, DT = `T2_tree_d4`,
GBDT-S = `T3_lgbm_50x8`, GBDT-M = `T4_lgbm_200x31`, GBDT-M+ = `T4f_lgbm_200x31_full`,
GBDT-L = `T5_lgbm_800x127`, TFIDF-LR = `T6_tfidf_lr`. Every results file carries both
names.

---

## Reproducing the results

### 1. Open the notebook

Go to [colab.research.google.com](https://colab.research.google.com), choose
**File > Upload notebook**, and upload `code/01_main_experiment.ipynb`.

### 2. Use a CPU runtime

**Runtime > Change runtime type**, set Hardware accelerator to **None**, then Save.
This matters: the paper measures CPU inference cost, so a GPU runtime would make the
timing results meaningless.

### 3. Run the smoke test first

`SMOKE = True` is already set in Cell 2. Run the cells in order with `Shift + Enter`.
The smoke test uses a small slice of the data and finishes in two to four minutes. Its
job is to prove every cell works, not to produce paper numbers.

### 4. Run the full experiment

Set `SMOKE = False` in Cell 2, then **Runtime > Restart session**, then
**Runtime > Run all**. Expect 20 to 40 minutes including the latency sweeps.

### 5. Run the three add-on analyses

After the notebook has finished, add a new code cell at the bottom, paste in the
contents of `code/02_duplicate_audit.py`, and run it. Repeat for `03_seed_stability.py`
and `04_leakage_and_deployment.py`. Each one reuses the models and caches already in
memory and writes its own CSV files.

### 6. Where the output goes

Everything is written to `MyDrive/ICACS_inline/` in the Google Drive account connected
in Cell 1, including parsed data caches, per-tier results, per-round training history,
raw latency arrays and figures. Finished tiers are skipped on a re-run, so a Colab
disconnect costs at most one tier.

---

## The dataset

CSIC 2010 is not redistributed here. The notebook downloads it automatically from a
public mirror, with no account needed. See `data/README.md` for the source and the
expected file sizes. The notebook checks that parsing yields exactly 36,000 normal and
25,065 anomalous requests and stops if it does not.

## Environment

The reported runs used a free Google Colab CPU runtime with Python 3, LightGBM 4.5.0,
scikit-learn 1.5.2 and NumPy 2.1.3. The notebook pins LightGBM and scikit-learn. If
Colab's Python version has moved on and the pinned scikit-learn no longer installs,
remove that pin and record the version you used; the three scikit-learn tiers (LR, DT
and TFIDF-LR) may then shift by a small amount.

## What will not reproduce exactly

Detection results are deterministic given the seeds and the library versions, and should
reproduce to the digits printed in `results/`.

Latency will not. It depends on which shared virtual CPU Colab allocates. That is itself
one of the paper's findings: the two reported sessions disagreed on which tiers met a
250 microsecond deadline, while the work proxy stayed fixed. Treat the ratios between
tiers as the result, not the absolute microseconds, and read the absolute values as lower
bounds, since the timing harness keeps the fastest of several repeats.

## Maintainer

This repository is maintained by **Muhammad Ahmad Zia**, who built the evaluation
harness and ran the experiments. He lectures in Computer Science at The University of
Lahore and works on sustainable AI, with a focus on the inference side of machine
learning rather than training. More about his research and consulting work at
[ahmadzia.com](https://ahmadzia.com).

Questions about reproducing these results, or about reusing the evaluation protocol on
another corpus, are welcome through the repository issues.

## License

Code and results in this repository: MIT, see `LICENSE`. CSIC 2010 remains under the terms
set by its original authors at the Spanish National Research Council.

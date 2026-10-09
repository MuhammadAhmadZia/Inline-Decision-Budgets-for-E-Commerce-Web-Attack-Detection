# ============================================================================
# Cell 16. Seed stability: do the deduplicated orderings hold across splits?
#
# After removing duplicates the training set is 15,364 rows, so a single random
# split could produce orderings that are an accident of that split. This cell
# repeats the deduplicated experiment under three different split seeds and
# reports whether every claim in the paper survives all three.
#
# Runs on the deduplicated data only, since that is what the paper reports.
# Takes roughly 4 to 6 minutes. Self-contained: needs the notebook to have been
# run top to bottom, but does not depend on Cell 15.
# ============================================================================

SEEDS = [42, 1, 2]

_dup_key = (df.method + ' ' + df.url + ' ' + df.body)
_uniq = np.where((~_dup_key.duplicated()).values)[0]
_y = df.label.values
print(f'deduplicated pool: {len(_uniq):,} unique requests, attack share {_y[_uniq].mean():.3f}')

def _train_one(t, tr, va, te, seed):
    """Train one tier on the given row indices; return validation scores, test scores, node visits."""
    fs, kind = t['featset'], t['kind']
    if fs == 'text':
        vec = TfidfVectorizer(analyzer='char_wb', lowercase=True, **t['params'])
        clf = LogisticRegression(max_iter=3000, C=10).fit(
            vec.fit_transform([TEXTS[i] for i in tr]), _y[tr])
        return (clf.predict_proba(vec.transform([TEXTS[i] for i in va]))[:, 1],
                clf.predict_proba(vec.transform([TEXTS[i] for i in te]))[:, 1], 0.0)
    cols = CHEAP_COLS if fs == 'cheap' else FULL_COLS
    Xtr, Xva, Xte = F.iloc[tr][cols], F.iloc[va][cols], F.iloc[te][cols]
    if kind == 'lr':
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), _y[tr])
        return (clf.predict_proba(sc.transform(Xva))[:, 1],
                clf.predict_proba(sc.transform(Xte))[:, 1], 0.0)
    if kind == 'dt':
        clf = DecisionTreeClassifier(random_state=seed, **t['params']).fit(Xtr, _y[tr])
        return (clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1],
                float(clf.decision_path(Xte).sum() / len(Xte)))
    clf = lgb.LGBMClassifier(random_state=seed, n_jobs=1, verbose=-1, **t['params']).fit(Xtr, _y[tr])
    return (clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1],
            node_visits_lgb(clf, Xte.values))

rows, HITS3 = [], {}
for seed in SEEDS:
    tr, tmp = train_test_split(_uniq, test_size=0.4, random_state=seed, stratify=_y[_uniq])
    va, te = train_test_split(tmp, test_size=0.5, random_state=seed, stratify=_y[tmp])
    overlap = float(_dup_key.iloc[te].isin(set(_dup_key.iloc[tr])).mean())
    print(f'\n--- seed {seed}: train {len(tr):,} | val {len(va):,} | test {len(te):,} '
          f'| train/test request overlap {100*overlap:.1f}% ---')
    for t in TIERS:
        pv, pt, nv = _train_one(t, tr, va, te, seed)
        yva, yte = _y[va], _y[te]
        row = {'seed': seed, 'tier': t['name'], 'auc': round(float(roc_auc_score(yte, pt)), 4),
               'node_visits_per_request': round(nv, 1)}
        for target in FPR_TARGETS:
            thr = float(np.quantile(pv[yva == 0], 1 - target))
            hits = ((pt >= thr).astype(int)[yte == 1] == 1).astype(np.uint8)
            row[f'recall@fpr{target}'] = round(float(hits.mean()), 4)
            HITS3[(seed, t['name'], target)] = hits
        rows.append(row)
        print(f"  {t['name']:24s} AUC {row['auc']:.4f}  "
              f"recall@0.1%FPR {row['recall@fpr0.001']:.4f}  recall@1%FPR {row['recall@fpr0.01']:.4f}")

SEEDDF = pd.DataFrame(rows)
SEEDDF.to_csv(os.path.join(RUN_DIR, 'seed_stability.csv'), index=False)

# ---- per-tier spread across seeds ----
for target in FPR_TARGETS:
    col = f'recall@fpr{target}'
    print('\n' + '=' * 78)
    print(f'RECALL AT {target:.1%} FPR ACROSS SEEDS (deduplicated)')
    print('=' * 78)
    piv = SEEDDF.pivot(index='tier', columns='seed', values=col)
    piv['mean'] = piv.mean(axis=1).round(4)
    piv['sd'] = piv[SEEDS].std(axis=1).round(4)
    piv['range'] = (piv[SEEDS].max(axis=1) - piv[SEEDS].min(axis=1)).round(4)
    print(piv.round(4).sort_values('mean', ascending=False).to_string())

# ---- do the paper's claims hold in every seed? ----
CLAIMS = [('T4_lgbm_200x31', 'T5_lgbm_800x127', 'mid-size beats largest'),
          ('T4_lgbm_200x31', 'T4f_lgbm_200x31_full', 'cheap features beat rich features'),
          ('T4_lgbm_200x31', 'T6_tfidf_lr', 'GBDT beats TF-IDF'),
          ('T4_lgbm_200x31', 'T3_lgbm_50x8', 'mid-size beats smallest')]

print('\n' + '=' * 78)
print('CLAIM STABILITY: paired bootstrap within each seed')
print('=' * 78)
claim_rows = []
for target in FPR_TARGETS:
    print(f'\nAt {target:.1%} FPR:')
    for a, b, label in CLAIMS:
        verdicts, diffs = [], []
        for seed in SEEDS:
            ha, hb = HITS3.get((seed, a, target)), HITS3.get((seed, b, target))
            if ha is None or hb is None: continue
            rs = np.random.RandomState(seed + 97)
            bi = rs.randint(0, len(ha), size=(N_BOOTSTRAP, len(ha))).astype(np.int32)
            d = ha[bi].mean(axis=1) - hb[bi].mean(axis=1)
            lo, hi = np.percentile(d, [2.5, 97.5])
            diff = float(ha.mean() - hb.mean())
            diffs.append(diff)
            verdicts.append('+' if (lo > 0) else ('-' if (hi < 0) else '0'))
            claim_rows.append({'fpr_target': target, 'claim': label, 'seed': seed,
                               'tier_a': a, 'tier_b': b, 'diff': round(diff, 4),
                               'lo': round(float(lo), 4), 'hi': round(float(hi), 4)})
        pattern = ''.join(verdicts)
        if pattern == '+' * len(SEEDS):
            verdict = f'HOLDS in {len(SEEDS)}/{len(SEEDS)} seeds'
        elif pattern == '-' * len(SEEDS):
            verdict = f'REVERSED in {len(SEEDS)}/{len(SEEDS)} seeds (b wins consistently)'
        elif '0' in pattern and '+' not in pattern and '-' not in pattern:
            verdict = 'NO significant difference in any seed'
        else:
            verdict = f'UNSTABLE across seeds ({pattern}) - do not claim a ranking'
        print(f"  {label:36s} diffs {['%+.4f' % d for d in diffs]}  -> {verdict}")

pd.DataFrame(claim_rows).to_csv(os.path.join(RUN_DIR, 'seed_claim_stability.csv'), index=False)
print('\nsaved seed_stability.csv and seed_claim_stability.csv to', RUN_DIR)
print('\nRead the verdicts: "HOLDS in 3/3" can go in the paper as a finding.')
print('"UNSTABLE" means the two tiers must be reported as comparable, not ranked.')

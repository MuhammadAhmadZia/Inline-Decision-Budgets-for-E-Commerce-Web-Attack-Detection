# ============================================================================
# Cell 15. Duplicate audit: how much of the result was memorisation?
#
# CSIC 2010 is generated from templates, so the same request string occurs many
# times. A random split therefore puts identical requests in train and test, and
# because our features are a deterministic function of the request, those test
# rows measure memorisation rather than detection.
#
# This cell re-runs every tier twice, once on the data as-is and once on unique
# requests only, and reports both. Latency is unaffected (it is measured per
# request, not per training set), so the timing results from Cell 10 still stand.
# ============================================================================

dup_key = (df.method + ' ' + df.url + ' ' + df.body)
uniq_mask = (~dup_key.duplicated()).values

print(f'rows total          : {len(df):,}')
print(f'unique request texts: {int(uniq_mask.sum()):,}  ({100*uniq_mask.mean():.1f}% of rows)')
print(f'duplicate rows      : {int((~uniq_mask).sum()):,}  ({100*(~uniq_mask).mean():.1f}%)')
print(f'attack share, all rows {df.label.mean():.3f} | unique rows {df.label[uniq_mask].mean():.3f}')

_y = df.label.values

def _split(mask):
    ix = np.where(mask)[0]
    a, tmp = train_test_split(ix, test_size=0.4, random_state=SEED, stratify=_y[ix])
    b, c = train_test_split(tmp, test_size=0.5, random_state=SEED, stratify=_y[tmp])
    return a, b, c

def _leak(tr, te):
    return float(dup_key.iloc[te].isin(set(dup_key.iloc[tr])).mean())

def _fit_predict(t, tr, va, te):
    """Train one tier on the given indices, return validation and test scores plus node visits."""
    fs, kind = t['featset'], t['kind']
    if fs == 'text':
        vec = TfidfVectorizer(analyzer='char_wb', lowercase=True, **t['params'])
        Vtr = vec.fit_transform([TEXTS[i] for i in tr])
        clf = LogisticRegression(max_iter=3000, C=10).fit(Vtr, _y[tr])
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
        clf = DecisionTreeClassifier(random_state=SEED, **t['params']).fit(Xtr, _y[tr])
        return (clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1],
                float(clf.decision_path(Xte).sum() / len(Xte)))
    clf = lgb.LGBMClassifier(random_state=SEED, n_jobs=1, verbose=-1, **t['params']).fit(Xtr, _y[tr])
    return (clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1],
            node_visits_lgb(clf, Xte.values))

CONDITIONS = {'with_duplicates': np.ones(len(df), bool), 'deduplicated': uniq_mask}
rows, HITS2 = [], {}
rs_boot = np.random.RandomState(SEED + 23)

for cond, mask in CONDITIONS.items():
    tr, va, te = _split(mask)
    leak = _leak(tr, te)
    print(f'\n--- {cond}: train {len(tr):,} | val {len(va):,} | test {len(te):,} '
          f'| test rows whose exact request is in train: {100*leak:.1f}% ---')
    for t in TIERS:
        pv, pt, nv = _fit_predict(t, tr, va, te)
        yva, yte = _y[va], _y[te]
        row = {'condition': cond, 'tier': t['name'], 'train_test_overlap': round(leak, 4),
               'auc': round(float(roc_auc_score(yte, pt)), 4),
               'node_visits_per_request': round(nv, 1)}
        for target in FPR_TARGETS:
            thr = float(np.quantile(pv[yva == 0], 1 - target))
            pred = (pt >= thr).astype(int)
            hits = (pred[yte == 1] == 1).astype(np.uint8)
            bi = rs_boot.randint(0, len(hits), size=(N_BOOTSTRAP, len(hits))).astype(np.int32)
            br = hits[bi].mean(axis=1)
            lo, hi = np.percentile(br, [2.5, 97.5])
            row[f'recall@fpr{target}'] = round(float(hits.mean()), 4)
            row[f'recall_lo@fpr{target}'] = round(float(lo), 4)
            row[f'recall_hi@fpr{target}'] = round(float(hi), 4)
            row[f'fpr_actual@fpr{target}'] = round(float(((pred == 1) & (yte == 0)).sum() /
                                                        max((yte == 0).sum(), 1)), 4)
            HITS2[(cond, t['name'], target)] = hits
        rows.append(row)
        print(f"  {t['name']:24s} AUC {row['auc']:.4f}  "
              f"recall@0.1%FPR {row['recall@fpr0.001']:.4f} "
              f"[{row['recall_lo@fpr0.001']:.4f}, {row['recall_hi@fpr0.001']:.4f}]  "
              f"recall@1%FPR {row['recall@fpr0.01']:.4f}")

D = pd.DataFrame(rows)
D.to_csv(os.path.join(RUN_DIR, 'dedup_comparison.csv'), index=False)

# ---- side-by-side view: what does deduplication cost each tier? ----
print('\n' + '=' * 78)
print('EFFECT OF DEDUPLICATION (recall at 0.1% FPR)')
print('=' * 78)
piv = D.pivot(index='tier', columns='condition', values='recall@fpr0.001')
piv['drop'] = (piv['with_duplicates'] - piv['deduplicated']).round(4)
piv['drop_pct'] = (100 * piv['drop'] / piv['with_duplicates']).round(1)
print(piv.round(4).to_string())
print(f"\nmean recall lost to deduplication: {piv['drop'].mean():.4f} "
      f"({piv['drop_pct'].mean():.1f}% relative)")

# ---- do the paper's orderings survive? paired bootstrap inside each condition ----
print('\n' + '=' * 78)
print('KEY COMPARISONS, PAIRED BOOTSTRAP')
print('=' * 78)
PAIRS_OF_INTEREST = [('T4_lgbm_200x31', 'T5_lgbm_800x127'),
                     ('T4f_lgbm_200x31_full', 'T4_lgbm_200x31'),
                     ('T4_lgbm_200x31', 'T3_lgbm_50x8')]
pair_rows = []
for target in FPR_TARGETS:
    for a, b in PAIRS_OF_INTEREST:
        line = {'fpr_target': target, 'tier_a': a, 'tier_b': b}
        for cond in CONDITIONS:
            ha, hb = HITS2.get((cond, a, target)), HITS2.get((cond, b, target))
            if ha is None or hb is None: continue
            rs = np.random.RandomState(SEED + 31)
            bi = rs.randint(0, len(ha), size=(N_BOOTSTRAP, len(ha))).astype(np.int32)
            d = ha[bi].mean(axis=1) - hb[bi].mean(axis=1)
            lo, hi = np.percentile(d, [2.5, 97.5])
            line[f'{cond}_diff'] = round(float(ha.mean() - hb.mean()), 4)
            line[f'{cond}_lo'] = round(float(lo), 4)
            line[f'{cond}_hi'] = round(float(hi), 4)
            line[f'{cond}_sig'] = bool(lo > 0 or hi < 0)
        flip = (np.sign(line.get('with_duplicates_diff', 0)) !=
                np.sign(line.get('deduplicated_diff', 0)))
        line['sign_flipped'] = bool(flip)
        pair_rows.append(line)
        d1, d2 = line.get('with_duplicates_diff'), line.get('deduplicated_diff')
        tag = '*** SIGN FLIPPED ***' if flip else ('holds' if line.get('deduplicated_sig') else 'not significant after dedup')
        print(f"@{target:<6} {a:22s} - {b:22s}  dup {d1:+.4f}  dedup {d2:+.4f} "
              f"[{line.get('deduplicated_lo'):+.4f}, {line.get('deduplicated_hi'):+.4f}]  {tag}")

PD_ = pd.DataFrame(pair_rows)
PD_.to_csv(os.path.join(RUN_DIR, 'dedup_paired_comparisons.csv'), index=False)
print('\nsaved dedup_comparison.csv and dedup_paired_comparisons.csv to', RUN_DIR)
print('\nReport the DEDUPLICATED column as the paper\'s headline results.')
print('Report the with-duplicates column as a measurement of benchmark inflation.')

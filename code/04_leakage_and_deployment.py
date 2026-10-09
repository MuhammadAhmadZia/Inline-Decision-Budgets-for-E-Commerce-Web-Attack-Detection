# ============================================================================
# Cell 17. Leakage sensitivity and the deployment table
#
# Part A re-runs every tier under three split protocols, three seeds each:
#   1. random        : rows split at random, duplicates left in (common practice)
#   2. exact_dedup   : duplicate requests removed first, then random split
#                      (a duplicate = identical method + full URL + POST body)
#   3. endpoint_heldout: after dedup, leave one application endpoint out.
#                      CSIC has five real endpoints (login, registration, profile
#                      edit, payment, add-to-cart) holding 91% of unique requests,
#                      each with both classes. Train on four, test on the fifth,
#                      rotate. Tests generalisation to an endpoint never seen in
#                      training, which is harsher than a real deployment where a
#                      storefront trains on its own endpoints.
#
#   Why not group by endpoint *template* (path + parameter names)? Because 1,657
#   of CSIC's 1,685 templates are attack-only: attacks mangle parameter names, so
#   each mutation becomes a new template and the template leaks the label.
#
# Part B joins the exact_dedup detection results with the Cell 10 timings and
# marks which tiers are feasible under a 250 us and a 1 ms p99 deadline.
#
# Needs the notebook run top to bottom (including Cell 10). Does not depend on
# Cells 15 or 16. Takes roughly 8 to 12 minutes.
# ============================================================================
from sklearn.model_selection import GroupShuffleSplit

SEEDS17 = [42, 1, 2]
_y = df.label.values
_key = (df.method + ' ' + df.url + ' ' + df.body)
_uniq = np.where((~_key.duplicated()).values)[0]

_path = df.url.str.split('?').str[0].values
_pstats = pd.DataFrame({'path': _path[_uniq], 'y': _y[_uniq]}).groupby('path').y.agg(['mean', 'size'])
ENDPOINTS = _pstats[(_pstats['mean'] > 0.05) & (_pstats['mean'] < 0.95) & (_pstats['size'] >= 300)].index.tolist()
print(f'held-out endpoints ({len(ENDPOINTS)}):', [e.rsplit('/', 1)[-1] for e in ENDPOINTS])

def _split_random(pool, seed):
    tr, tmp = train_test_split(pool, test_size=0.4, random_state=seed, stratify=_y[pool])
    va, te = train_test_split(tmp, test_size=0.5, random_state=seed, stratify=_y[tmp])
    return tr, va, te

def _split_endpoint(pool, fold):
    """Test = every unique request on one endpoint. Train/val = everything else, stratified 75/25."""
    ep = ENDPOINTS[fold]
    in_ep = _path[pool] == ep
    te, rest = pool[in_ep], pool[~in_ep]
    tr, va = train_test_split(rest, test_size=0.25, random_state=SEED, stratify=_y[rest])
    return tr, va, te

PROTOCOLS = {
    'random':           (np.arange(len(df)), _split_random,   SEEDS17),
    'exact_dedup':      (_uniq,              _split_random,   SEEDS17),
    'endpoint_heldout': (_uniq,              _split_endpoint, list(range(len(ENDPOINTS)))),
}

def _fit17(t, tr, va, te, seed):
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

# ------------------------------------------------------------------ Part A
rows = []
for proto, (pool, splitter, reps) in PROTOCOLS.items():
    for seed in reps:
        tr, va, te = splitter(pool, seed)
        exact_ov = float(_key.iloc[te].isin(set(_key.iloc[tr])).mean())
        path_ov = float(pd.Series(_path[te]).isin(set(_path[tr])).mean())
        tag = f'endpoint {ENDPOINTS[seed].rsplit("/", 1)[-1]}' if proto == 'endpoint_heldout' else f'seed {seed}'
        print(f'\n[{proto} | {tag}] train {len(tr):,} val {len(va):,} test {len(te):,} | '
              f'test attack share {_y[te].mean():.3f} | exact overlap {100*exact_ov:.1f}% | '
              f'endpoint overlap {100*path_ov:.1f}%')
        for t in TIERS:
            pv, pt, nv = _fit17(t, tr, va, te, seed)
            yva, yte = _y[va], _y[te]
            row = {'protocol': proto, 'seed_or_fold': seed, 'tier': t['name'],
                   'exact_overlap': round(exact_ov, 4), 'endpoint_overlap': round(path_ov, 4),
                   'auc': round(float(roc_auc_score(yte, pt)), 4),
                   'node_visits_per_request': round(nv, 1)}
            for target in FPR_TARGETS:
                thr = float(np.quantile(pv[yva == 0], 1 - target))
                pred = (pt >= thr).astype(int)
                row[f'recall@fpr{target}'] = round(float(pred[yte == 1].mean()), 4)
                row[f'fpr_actual@fpr{target}'] = round(float(pred[yte == 0].mean()), 5)
            rows.append(row)
            print(f"   {t['name']:24s} AUC {row['auc']:.4f}  rec@0.1% {row['recall@fpr0.001']:.4f}  "
                  f"rec@1% {row['recall@fpr0.01']:.4f}")

P17 = pd.DataFrame(rows)
P17.to_csv(os.path.join(RUN_DIR, 'protocol_sensitivity_raw.csv'), index=False)

agg = (P17.groupby(['protocol', 'tier'])
          [['auc', 'recall@fpr0.001', 'recall@fpr0.01', 'exact_overlap', 'endpoint_overlap']]
          .agg(['mean', 'std']).round(4))
agg.to_csv(os.path.join(RUN_DIR, 'protocol_sensitivity_summary.csv'))

order = ['random', 'exact_dedup', 'endpoint_heldout']
for target in FPR_TARGETS:
    col = f'recall@fpr{target}'
    m = P17.groupby(['tier', 'protocol'])[col].mean().unstack()[order]
    s = P17.groupby(['tier', 'protocol'])[col].std().unstack()[order]
    print('\n' + '=' * 86)
    print(f'LEAKAGE SENSITIVITY: recall at {target:.1%} FPR, mean (sd) over 3 seeds, or 5 held-out endpoints')
    print('=' * 86)
    view = pd.DataFrame({p: [f'{m.loc[t,p]:.4f} ({s.loc[t,p]:.4f})' for t in m.index] for p in order},
                        index=m.index)
    view['inflation_random_vs_dedup'] = [f"{100*(m.loc[t,'random']-m.loc[t,'exact_dedup'])/max(m.loc[t,'random'],1e-9):+.1f}%"
                                         for t in m.index]
    print(view.to_string())
    infl = 100 * (m['random'] - m['exact_dedup']) / m['random']
    print(f"\nmean relative inflation, random vs exact_dedup: {infl.mean():.1f}%")

ov = P17.groupby('protocol')[['exact_overlap', 'endpoint_overlap']].mean()
print('\nmean train/test overlap by protocol:')
print((100 * ov).round(1).to_string())

# ------------------------------------------------------------------ Part B
S = pd.read_csv(os.path.join(RUN_DIR, 'timing_per_sweep.csv'))
tcols = ['p50_us', 'p90_us', 'p99_us']
tim = S.groupby('tier')[tcols].min()                         # fastest sweep = least-noise estimate
for D in [250, 1000]:
    c = f'meets_{D}us_pct'
    if c in S.columns:
        tim[f'violates_{D}us_pct'] = (100 - S.groupby('tier')[c].max()).round(2)

ded = P17[P17.protocol == 'exact_dedup'].groupby('tier').agg(
    recall_strict=('recall@fpr0.001', 'mean'), recall_strict_sd=('recall@fpr0.001', 'std'),
    fpr_strict=('fpr_actual@fpr0.001', 'mean'),
    recall_loose=('recall@fpr0.01', 'mean'), recall_loose_sd=('recall@fpr0.01', 'std'),
    fpr_loose=('fpr_actual@fpr0.01', 'mean'),
    auc=('auc', 'mean'), node_visits=('node_visits_per_request', 'mean'))

DEP = ded.join(tim, how='left')
for D in [250, 1000]:
    DEP[f'feasible_{D}us'] = DEP['p99_us'] <= D
DEP = DEP.round(4).sort_values('recall_strict', ascending=False)
DEP.to_csv(os.path.join(RUN_DIR, 'deployment_table.csv'))

print('\n' + '=' * 86)
print('DEPLOYMENT TABLE (exact_dedup, mean over seeds; latency = fastest of 5 sweeps)')
print('=' * 86)
show = ['recall_strict', 'recall_strict_sd', 'recall_loose', 'recall_loose_sd', 'auc',
        'node_visits', 'p50_us', 'p90_us', 'p99_us', 'violates_250us_pct', 'violates_1000us_pct',
        'feasible_250us', 'feasible_1000us']
print(DEP[[c for c in show if c in DEP.columns]].to_string())

print('\n' + '=' * 86)
print('FEASIBLE FRONTIER')
print('=' * 86)
for D in [250, 1000]:
    feas = DEP[DEP[f'feasible_{D}us']]
    for label, col, sdcol in [('0.1% FPR', 'recall_strict', 'recall_strict_sd'),
                              ('1% FPR', 'recall_loose', 'recall_loose_sd')]:
        if len(feas) == 0:
            print(f'  {label} + {D}us p99: no tier is feasible'); continue
        best = feas[col].max()
        band = feas[feas[col] >= best - feas.loc[feas[col].idxmax(), sdcol]]
        print(f"  {label} + {D}us p99: feasible = {', '.join(feas.index)}")
        print(f"      best mean recall {best:.4f}; within one sd of best (treat as tied): "
              f"{', '.join(band.index)}")

print('\nsaved protocol_sensitivity_raw.csv, protocol_sensitivity_summary.csv and deployment_table.csv to', RUN_DIR)

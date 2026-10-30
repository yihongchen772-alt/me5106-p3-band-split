"""Task 1 predictive comparison: every split model through the identical SFS + SVR pipeline and splits.

python compare.py           compute missing (model, outer bundle) jobs, then summarize
python compare.py --summarize  summarize cached jobs only
Reuses the frozen ewt_verification bundles (84 = 2 windows x 3 seeds x 14 held-out conditions) and EWT views;
frozen folders are read only. Job results are cached in results/cache/<model>/.
"""
from pathlib import Path
import json, time, argparse, collections
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy import stats
import bands as B

v, old = B.v, B.old
OUT = B.HERE / 'results'; CACHE = OUT / 'cache'
MARGIN = 1.0  # equivalence margin in percentage points of macro MAPE (declared in protocol.json)
PAIRS = [('EWT', 'NAT_55.5', 'adaptivity'), ('MEY2.5_55.5', 'NAT_55.5', 'shape at valley'),
         ('NAT_55.5', 'NAT_40', 'location vs existing 40 kHz'), ('NAT_55.5', 'FIX5BW', 'vs existing 5-band'),
         ('NAT_55.5', 'RAW', 'split vs none'), ('MEY2.5_55.5', 'MEY2.5_40', 'location, constant width'),
         ('EWT', 'NAT_40', 'EWT vs existing 40 kHz'), ('EWT', 'FIX5BW', 'EWT vs existing 5-band'),
         ('NAT_55.625', 'EWT', 'LATE reference boundary vs EWT')]
KEY = ['RAW', 'FIX5BW', 'NAT_40', 'NAT_55.5', 'MEY2.5_55.5', 'EWT']


def job(model, path):
    dest = CACHE / model / Path(path).name
    if dest.exists():
        return json.loads(dest.read_text())
    b = json.loads(Path(path).read_text()); w = b['window']; d = v.data(w); start = time.time()
    idx = {int(s): i for i, s in enumerate(d['ids'])}
    tr = np.array([idx[s] for s in b['train_ids']]); te = np.array([idx[s] for s in b['test_ids']])
    X = B.matrix(model, w, b['outer_view']); splits = []
    for s in b['inner_splits']:
        it = np.array([idx[z] for z in s['train_ids']]); iv = np.array([idx[z] for z in s['test_ids']])
        assert set(it) | set(iv) == set(tr) and not set(it) & set(iv) and not set(it) & set(te)
        x = B.matrix(model, w, s['view_key'])
        a, c = old.prior.preprocess(x[it], x[iv])
        wts = np.array([1 / np.sum(d['conditions'][iv] == z) / len(np.unique(d['conditions'][tr])) for z in d['conditions'][iv]])
        splits.append((a, c, it, iv, wts))
    res = []
    for target in ['power', 'speed']:
        y = d[target]
        chosen, _ = old.select_prepared([(a, c, y[it], y[iv], wt) for a, c, it, iv, wt in splits], target)
        if chosen:
            a, c = old.prior.preprocess(X[tr][:, chosen], X[te][:, chosen]); p = old.prior.fit_predict(a, y[tr], c, target)
        else:
            p = np.full(len(te), np.median(y[tr]))
        res.append(dict(target=target, selected=[v.NAMES[j] for j in chosen], observed=y[te].tolist(), predicted=np.asarray(p).tolist()))
    obj = dict(model=model, window=w, seed=b['seed'], fold=b['fold'], condition=b['condition'], test_ids=b['test_ids'],
               results=res, seconds=time.time() - start)
    dest.parent.mkdir(parents=True, exist_ok=True); old.save(dest, obj)
    return obj


def metrics(df, keys):
    per = df.groupby(keys + ['condition'])[['AE', 'APE']].mean()
    macro = per.groupby(keys).mean().rename(columns={'AE': 'macro_MAE', 'APE': 'macro_MAPE'})
    group = df.groupby(keys)[['AE', 'APE']].mean().rename(columns={'AE': 'group_MAE', 'APE': 'group_MAPE'})
    return macro.join(group).reset_index(), per.reset_index()


def paired(m, new, base, w, target):
    delta = (m.loc[w, target, new] - m.loc[w, target, base]).to_numpy()
    rng = np.random.default_rng(441); bs = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(1)
    p = 1.0 if np.allclose(delta, 0) else stats.wilcoxon(delta, zero_method='zsplit').pvalue
    lo90, hi90 = np.quantile(bs, [.05, .95])
    return dict(window=w, target=target, new=new, baseline=base, mean_delta_pp=delta.mean(),
                ci95_low=np.quantile(bs, .025), ci95_high=np.quantile(bs, .975), ci90_low=lo90, ci90_high=hi90,
                wilcoxon_p=p, new_better=int(np.sum(delta < -1e-9)), new_worse=int(np.sum(delta > 1e-9)),
                ties=int(np.sum(abs(delta) <= 1e-9)), equivalent_pm1pp=bool(lo90 > -MARGIN and hi90 < MARGIN))


def summarize():
    paths = sorted(v.BUNDLES.glob('*.json')); rows, sel = [], []
    for model in B.MODELS:
        for p in paths:
            z = json.loads((CACHE / model / p.name).read_text())
            for r in z['results']:
                sel.append(dict(window=z['window'], seed=z['seed'], fold=z['fold'], model=model, target=r['target'], selected=';'.join(r['selected'])))
                for sid, y, pr in zip(z['test_ids'], r['observed'], r['predicted']):
                    rows.append(dict(window=z['window'], seed=z['seed'], fold=z['fold'], condition=z['condition'],
                                     model=model, target=r['target'], sample_id=sid, observed=y, predicted=pr))
    df = pd.DataFrame(rows)
    med = pd.read_csv(v.OUT / 'predictions.csv'); med = med[med.model == 'TRAIN_MEDIAN'][df.columns]
    df = pd.concat([df, med], ignore_index=True)
    assert len(df) == (len(B.MODELS) + 1) * 2 * 3 * 46 * 2
    df['AE'] = abs(df.predicted - df.observed); df['APE'] = 100 * df.AE / df.observed
    df.to_csv(OUT / 'predictions.csv.gz', index=False); sel = pd.DataFrame(sel); sel.to_csv(OUT / 'selections.csv', index=False)

    seed_m, per = metrics(df, ['window', 'seed', 'model', 'target'])
    seed_m.to_csv(OUT / 'metrics_per_seed.csv', index=False)
    g = seed_m.groupby(['window', 'model', 'target'])
    mean = g[['macro_MAPE', 'group_MAPE', 'macro_MAE', 'group_MAE']].mean()
    mean['macro_MAPE_seed_min'] = g.macro_MAPE.min(); mean['macro_MAPE_seed_max'] = g.macro_MAPE.max()
    mean = mean.reset_index(); mean.to_csv(OUT / 'mean_metrics.csv', index=False)
    ens = df.groupby(['window', 'model', 'target', 'condition', 'sample_id']).agg(observed=('observed', 'first'), predicted=('predicted', 'mean')).reset_index()
    ens['AE'] = abs(ens.predicted - ens.observed); ens['APE'] = 100 * ens.AE / ens.observed
    ens_m, _ = metrics(ens, ['window', 'model', 'target']); ens_m.to_csv(OUT / 'ensemble_metrics.csv', index=False)

    m = per.groupby(['window', 'target', 'model', 'condition']).APE.mean()
    m.reset_index().to_csv(OUT / 'condition_errors.csv', index=False)
    pr = [dict(paired(m, new, base, w, t), question=q) for new, base, q in PAIRS for w in old.WINDOWS for t in ['power', 'speed']]
    pr = pd.DataFrame(pr); pr.to_csv(OUT / 'paired_comparisons.csv', index=False)

    # noise references: per-model seed range, and spread over the local 54.5-56.5 kHz grid
    noise = []
    for (w, t), z in mean.groupby(['window', 'target']):
        loc = z[z.model.isin([B.nat(b) for b in B.LOCAL])]
        noise.append(dict(window=w, target=t, median_seed_range_pp=(z.macro_MAPE_seed_max - z.macro_MAPE_seed_min).median(),
                          local_grid_sd_pp=loc.macro_MAPE.std(ddof=1), local_grid_range_pp=np.ptp(loc.macro_MAPE)))
    noise = pd.DataFrame(noise); noise.to_csv(OUT / 'noise_reference.csv', index=False)

    freq = []
    for (w, model, t), z in sel[sel.model.isin(KEY)].groupby(['window', 'model', 'target']):
        c = collections.Counter(n for s in z.selected for n in s.split(';') if n)
        freq += [dict(window=w, model=model, target=t, feature=f, count=k, total=len(z), percent=100 * k / len(z)) for f, k in c.most_common()]
    pd.DataFrame(freq).to_csv(OUT / 'selection_frequency.csv', index=False)

    arch = [pd.read_csv(v.OUT / 'fixed_control' / 'mean_metrics.csv').assign(source='ewt_verification/fixed_control (Linux)'),
            pd.read_csv(v.OUT / 'mean_metrics.csv').query("model=='EWT_STABLE_FREQ'").assign(source='ewt_verification (Linux)'),
            pd.read_csv(old.OLD / 'results' / 'metrics.csv').query("model.str.startswith('FIXED')", engine='python')
              .groupby(['window', 'model', 'target'])[['macro_MAPE', 'group_MAPE', 'macro_MAE']].mean().reset_index()
              .assign(source='window_filter_robustness, 252 paper statistics (Linux)')]
    pd.concat(arch, ignore_index=True)[['source', 'window', 'model', 'target', 'macro_MAPE', 'group_MAPE', 'macro_MAE']].to_csv(OUT / 'archived_reference.csv', index=False)

    pd.set_option('display.width', 220, 'display.max_rows', 200)
    show = mean[mean.model.isin(KEY + ['TRAIN_MEDIAN'])].pivot_table(index='model', columns=['window', 'target'], values='macro_MAPE').round(2)
    print('macro MAPE (mean of 3 seeds)'); print(show.loc[KEY + ['TRAIN_MEDIAN']].to_string())
    showg = mean[mean.model.isin(KEY + ['TRAIN_MEDIAN'])].pivot_table(index='model', columns=['window', 'target'], values='group_MAPE').round(2)
    print('group MAPE'); print(showg.loc[KEY + ['TRAIN_MEDIAN']].to_string())
    print(pr.drop(columns='question').round(3).to_string(index=False)); print(noise.round(2).to_string(index=False))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--summarize', action='store_true'); p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    if not a.summarize:
        paths = sorted(v.BUNDLES.glob('*.json')); assert len(paths) == 84
        todo = [(m, str(q)) for m in B.MODELS for q in paths if not (CACHE / m / q.name).exists()]
        print(len(todo), 'jobs to run', flush=True); t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for i, f in enumerate(as_completed([pool.submit(job, m, q) for m, q in todo]), 1):
                f.result()
                if i % 200 == 0 or i == len(todo):
                    print(f'{i}/{len(todo)} jobs, {time.time() - t0:.0f} s', flush=True)
    summarize()


if __name__ == '__main__':
    main()

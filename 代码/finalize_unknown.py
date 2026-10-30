"""Unknown samples #41, #43, #49 under the package's final protocol (final_report/finalize.py), same machine:
EWT (boundaries re-estimated per inner split, as finalize.py) vs fixed splits (one matrix for every split).
Report only: unknown predictions are never used to choose a split.
"""
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import bands as B

v, old = B.v, B.old
OUT = B.HERE / 'results'; FIN = B.HERE.parent / 'final_report' / 'results'
MODELS = ['EWT', 'NAT_55.5', 'MEY2.5_55.5', 'NAT_40', 'FIX5BW']


def key(ids):
    return int(hashlib.sha256(ids.tobytes() + b'EARLY').hexdigest()[:8], 16)


def main():
    d = v.data('EARLY'); ids, cond, x = d['ids'], d['conditions'], d['x']; unknown = [41, 43, 49]
    z = np.load(FIN / 'unknown_waveforms.npz'); xu = z['x']; assert list(z['ids']) == unknown and xu.shape == (15, 1280)
    bound, _ = v.partition(d['mt'], key(ids))
    mats = {'EWT': (old.aggregate(v.bank(x, old.transform(x, bound)[0])), old.aggregate(v.bank(xu, old.transform(xu, bound)[0])))}
    for m in MODELS[1:]:
        mats[m] = (B.matrix(m, 'EARLY'), old.aggregate(v.bank(xu, B.butter5(xu) if m == 'FIX5BW' else B.components(m, xu))))
    ukeys = np.unique(cond); rows, sel = [], []
    for seed in old.SEEDS:
        folds = [(np.flatnonzero(np.isin(cond, ukeys[a])), np.flatnonzero(np.isin(cond, ukeys[b])))
                 for a, b in KFold(5, shuffle=True, random_state=seed).split(ukeys)]
        wts = [np.array([1 / np.sum(cond[va] == c) / len(ukeys) for c in cond[va]]) for _, va in folds]
        ewt_inner = [old.aggregate(v.bank(x, old.transform(x, v.partition(d['mt'][tr], key(ids[tr]))[0])[0])) for tr, _ in folds]
        for m, (X, U) in mats.items():
            inner = ewt_inner if m == 'EWT' else [X] * len(folds)
            for target in ['power', 'speed']:
                y = d[target]; prep = []
                for (tr, va), w, xx in zip(folds, wts, inner):
                    a, b = old.prior.preprocess(xx[tr], xx[va]); prep.append((a, b, y[tr], y[va], w))
                cc, _ = old.select_prepared(prep, target)
                if cc:
                    a, b = old.prior.preprocess(X[:, cc], U[:, cc]); pred = old.prior.fit_predict(a, y, b, target)
                else:
                    pred = np.full(3, np.median(y))
                sel.append(dict(model=m, target=target, seed=seed, selected=';'.join(v.NAMES[i] for i in cc)))
                rows += [dict(sample_id=s, model=m, target=target, seed=seed, predicted=float(p)) for s, p in zip(unknown, pred)]
    df = pd.DataFrame(rows); df.to_csv(OUT / 'unknown_per_seed.csv', index=False); pd.DataFrame(sel).to_csv(OUT / 'unknown_selections.csv', index=False)
    arch = pd.read_csv(FIN / 'unknown_per_seed.csv').query("method=='EWT'").rename(columns={'method': 'model'})
    arch['model'] = 'EWT_archived_Linux'
    allp = pd.concat([df, arch[df.columns]], ignore_index=True)
    s = allp.groupby(['sample_id', 'target', 'model']).predicted.agg(['mean', 'min', 'max']).reset_index()
    s.to_csv(OUT / 'unknown_summary.csv', index=False)
    print(s.pivot_table(index=['sample_id', 'target'], columns='model', values='mean').round(1).to_string())
    print('full-46 EARLY EWT boundary', bound.tolist())


if __name__ == '__main__':
    main()

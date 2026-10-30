"""Model-free separability of band descriptors vs boundary location (descriptive; no SFS/SVR, no outer folds).

For each window, filter shape (NAT, MEY2.5) and boundary 20-75 kHz (0.5 kHz steps): the 18 band descriptors
(9 per component) of the 46 labelled groups, and their one-way ANOVA omega^2 for
  condition   14 P-v conditions
  speed       3 speed levels
  power|v     power given speed: SS(condition) - SS(speed) against the within-condition error (partial omega^2)
Raw-window descriptors do not depend on the boundary and are reported as reference levels.
"""
import numpy as np
import pandas as pd
import bands as B

v = B.v
LOCS = np.round(np.arange(20, 75.01, 0.5), 2)


def anova(y, cond, speed):
    y = np.asarray(y, float); N = len(y); gm = y.mean()
    ss_t = ((y - gm) ** 2).sum()
    ss_c = sum(np.sum(cond == c) * (y[cond == c].mean() - gm) ** 2 for c in np.unique(cond))
    ss_v = sum(np.sum(speed == s) * (y[speed == s].mean() - gm) ** 2 for s in np.unique(speed))
    k_c, k_v = len(np.unique(cond)), len(np.unique(speed))
    ms_w = (ss_t - ss_c) / (N - k_c)
    w_c = (ss_c - (k_c - 1) * ms_w) / (ss_t + ms_w)
    ms_wv = (ss_t - ss_v) / (N - k_v)
    w_v = (ss_v - (k_v - 1) * ms_wv) / (ss_t + ms_wv)
    ss_p, df_p = ss_c - ss_v, k_c - k_v
    w_p = (ss_p - df_p * ms_w) / (ss_p + (N - df_p) * ms_w)
    return w_c, w_v, w_p


def main():
    rows = []
    for w in ['EARLY', 'LATE']:
        d = v.data(w); x = d['x']; cond = d['conditions']; speed = d['speed']
        raw = B.old.aggregate(v.bank(x, np.empty((len(x), 0, x.shape[-1]))))
        for j, name in enumerate(v.NAMES[:9]):
            rows.append(dict(window=w, shape='RAW', boundary_khz=np.nan, feature=name, **dict(zip(['omega2_condition', 'omega2_speed', 'omega2_power_given_speed'], anova(raw[:, j], cond, speed)))))
        for shape in ['NAT', 'MEY2.5']:
            for b in LOCS:
                model = B.nat(b) if shape == 'NAT' else B.mey(2.5, b)
                X = B.old.aggregate(v.bank(x, B.components(model, x)))
                for j in range(9, 27):
                    rows.append(dict(window=w, shape=shape, boundary_khz=b, feature=v.NAMES[j],
                                     **dict(zip(['omega2_condition', 'omega2_speed', 'omega2_power_given_speed'], anova(X[:, j], cond, speed)))))
    df = pd.DataFrame(rows); df.to_csv(B.HERE / 'results' / 'separability.csv', index=False)
    band = df[df['shape'] != 'RAW']
    best = band.loc[band.groupby(['window', 'shape', 'boundary_khz']).omega2_power_given_speed.idxmax()]
    for (w, s), z in best.groupby(['window', 'shape']):
        top = z.nlargest(3, 'omega2_power_given_speed')[['boundary_khz', 'feature', 'omega2_power_given_speed']]
        print(w, s, 'best single band feature for power|speed:'); print(top.round(3).to_string(index=False))
    print(df[df['shape'] == 'RAW'].groupby('window')[['omega2_power_given_speed', 'omega2_speed']].max().round(3))


if __name__ == '__main__':
    main()

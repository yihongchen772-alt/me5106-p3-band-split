"""Physical origin of the ~55.5 kHz spectral valley (label-free, all 49 samples, raw MAT).

Per track: multitaper PSD (NW 2.5, 4 tapers, 62.5 Hz grid) of
  plateau  the pipeline EARLY window (onset + 2 ms, 1280 samples, laser on)
  tail     end - 7 ms .. end - 1 ms (post-laser decay; the laser is already off)
and of idle-noise windows between samples. PSDs are normalised to unit power in 8-78 kHz, averaged per sample,
smoothed over 1 kHz; valley = minimum in 50-60 kHz (parabolic refinement), depth = shoulders (46-50, 61-65 kHz)
over the minimum. Track times come from the package's research/track_features.csv.
"""
from pathlib import Path
import json
import h5py
import numpy as np
import pandas as pd
from scipy import signal, stats

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; OUT = HERE / 'results'
FS, NFFT, WIN = 160000, 2560, 1280
GRID = [[43, 42, 41, 40, 39, 38, 37], [44, 36, 33, 30, 27, 24, 21], [45, 35, 32, 29, 26, 23, 20], [46, 34, 31, 28, 25, 22, 19],
        [47, 18, 15, 12, 9, 6, 3], [48, 17, 14, 11, 8, 5, 2], [49, 16, 13, 10, 7, 4, 1]]  # course description p.3
POS = {s: (r, c) for r, row in enumerate(GRID) for c, s in enumerate(row)}
f = np.fft.rfftfreq(NFFT, 1 / FS)
BAND = (f >= 8e3) & (f <= 78e3)


def mt_psd(xs):
    tap = signal.windows.dpss(xs.shape[-1], 2.5, Kmax=4, sym=False)
    X = np.fft.rfft((xs - xs.mean(-1, keepdims=True))[:, None, :] * tap[None], n=NFFT, axis=-1)
    p = np.mean(abs(X) ** 2, axis=1)
    return p / p[:, BAND].sum(-1, keepdims=True)


def valley(p):
    s = np.convolve(p, np.ones(16) / 16, mode='same')
    m = np.flatnonzero((f >= 50e3) & (f <= 60e3)); i = m[np.argmin(s[m])]
    y0, y1, y2 = np.log(s[i - 1:i + 2]); off = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    shoulder = s[((f >= 46e3) & (f <= 50e3)) | ((f >= 61e3) & (f <= 65e3))].mean()
    return (f[i] + off * (f[1] - f[0])) / 1e3, 10 * np.log10(shoulder / s[i])


def main():
    with h5py.File(ROOT / 'data' / 'Mini Project 3 Data.mat') as h:
        raw = h['IEPE0_Raw'][()].ravel()
    hp = signal.sosfiltfilt(signal.butter(4, 8000, fs=FS, btype='highpass', output='sos'), raw); del raw
    tr = pd.read_csv(ROOT / 'analysis' / 'research' / 'track_features.csv').sort_values(['sample_id', 'track_id'])
    lab = pd.read_csv(ROOT / 'analysis' / 'sample_labels.csv').set_index('sample_id')

    # the plateau window must be the pipeline's EARLY window: compare with the cached decomposition input
    z = np.load(ROOT / 'analysis' / 'window_filter_robustness' / 'results' / 'decompositions' / 'EARLY_01_1.npz')
    a0 = round(tr.iloc[0].start_s * FS) + 320
    parity = float(np.max(abs(hp[a0:a0 + WIN] - z['x'])) / np.max(abs(z['x'])))

    plate, tail = {}, {}
    for r in tr.itertuples():
        a = round(r.start_s * FS) + 320; e = round(r.end_threshold_s * FS)
        plate.setdefault(r.sample_id, []).append(hp[a:a + WIN])
        tail.setdefault(r.sample_id, []).append(hp[e - int(7e-3 * FS): e - int(1e-3 * FS)])
    # idle: up to 20 windows from the middle of each inter-sample gap (> 0.5 s from any track), spikes rejected
    g = tr.groupby('sample_id').agg(s=('start_s', 'min'), e=('end_threshold_s', 'max')).sort_values('s')
    idle = []
    for e0, s1 in zip(g.e.to_numpy()[:-1], g.s.to_numpy()[1:]):
        lo, hi = round((e0 + .5) * FS), round((s1 - .5) * FS)
        for a in np.linspace(lo, hi - WIN, 20).astype(int):
            idle.append(hp[a:a + WIN])
    idle = np.array(idle); rms = np.sqrt((idle ** 2).mean(1)); idle = idle[rms < 3 * np.median(rms)]

    rows, P_pl, P_tl = [], {}, {}
    for s in sorted(plate):
        P_pl[s] = mt_psd(np.array(plate[s])).mean(0); P_tl[s] = mt_psd(np.array(tail[s])).mean(0)
        fv, dv = valley(P_pl[s]); ft, dt = valley(P_tl[s])
        rows.append(dict(sample_id=s, power_W=lab.loc[s, 'power_W'], speed_mm_s=lab.loc[s, 'scan_speed_mm_s'],
                         row=POS[s][0], col=POS[s][1], valley_plateau_khz=fv, depth_plateau_db=dv,
                         valley_tail_khz=ft, depth_tail_db=dt))
    df = pd.DataFrame(rows); df.to_csv(OUT / 'valley_by_sample.csv', index=False)
    P_idle = mt_psd(idle).mean(0)
    fi, di = valley(P_idle)
    np.savez_compressed(OUT / 'valley_spectra.npz', f=f, ids=np.array(sorted(plate)), plateau=np.array([P_pl[s] for s in sorted(plate)]),
                        tail=np.array([P_tl[s] for s in sorted(plate)]), idle=P_idle)

    known = df.dropna(subset=['power_W'])
    def rho(a, b):
        r = stats.spearmanr(a, b); return dict(rho=float(r.statistic), p=float(r.pvalue))
    summary = dict(
        parity_plateau_vs_pipeline_window=parity, n_idle_windows=int(len(idle)),
        plateau=dict(mean_khz=df.valley_plateau_khz.mean(), sd_khz=df.valley_plateau_khz.std(), min_khz=df.valley_plateau_khz.min(),
                     max_khz=df.valley_plateau_khz.max(), depth_db_median=df.depth_plateau_db.median(), depth_db_min=df.depth_plateau_db.min()),
        tail=dict(mean_khz=df.valley_tail_khz.mean(), sd_khz=df.valley_tail_khz.std(), min_khz=df.valley_tail_khz.min(),
                  max_khz=df.valley_tail_khz.max(), depth_db_median=df.depth_tail_db.median(), depth_db_min=df.depth_tail_db.min()),
        tail_minus_plateau_khz=dict(mean=(df.valley_tail_khz - df.valley_plateau_khz).mean(), sd=(df.valley_tail_khz - df.valley_plateau_khz).std()),
        idle=dict(min_in_50_60_khz=fi, depth_db=di),
        spearman_plateau_valley=dict(power=rho(known.power_W, known.valley_plateau_khz), speed=rho(df.dropna(subset=['speed_mm_s']).speed_mm_s, df.dropna(subset=['speed_mm_s']).valley_plateau_khz),
                                     row=rho(df.row, df.valley_plateau_khz), col=rho(df.col, df.valley_plateau_khz), print_order=rho(df.sample_id, df.valley_plateau_khz)),
        kruskal_speed=float(stats.kruskal(*[z.valley_plateau_khz for _, z in df.dropna(subset=['speed_mm_s']).groupby('speed_mm_s')]).pvalue),
        unknowns=df[df.sample_id.isin([41, 43, 49])][['sample_id', 'valley_plateau_khz', 'depth_plateau_db']].to_dict('records'))
    json.dump(summary, open(OUT / 'valley_summary.json', 'w'), indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == '__main__':
    main()

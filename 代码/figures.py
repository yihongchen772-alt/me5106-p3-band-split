"""Figures for the Task 1 report (reads results/ only; run after compare.py, valley.py, separability.py)."""
from pathlib import Path
import json, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import signal
import bands as B

HERE = Path(__file__).resolve().parent; R = HERE / 'results'; FIG = HERE / 'figures'; FIG.mkdir(exist_ok=True)
INK, MUTED, GRID, LIGHT = '#1f1f1e', '#6b6a64', '#e4e3dd', '#c9c8c1'
COL = {'EWT': '#2a78d6', 'NAT_40': '#eb6834', 'NAT': '#1baf7a', 'NAT_55.5': '#1baf7a', 'FIX5BW': '#eda100',
       'MEY': '#e87ba4', 'MEY2.5_55.5': '#e87ba4', 'RAW': '#8a8984'}
LAB = {'RAW': '不分带', 'FIX5BW': '已有 5 带 Butterworth', 'NAT_40': '已有固定 40 kHz', 'NAT_55.5': '固定 55.5 kHz',
       'MEY2.5_55.5': '固定 55.5 kHz（窄过渡 ±2.5 kHz）', 'EWT': '自适应 EWT', 'TRAIN_MEDIAN': '训练中位数基线'}
CELLS = [('EARLY', 'power', 'EARLY 窗口 · 功率'), ('EARLY', 'speed', 'EARLY 窗口 · 速度'),
         ('LATE', 'power', 'LATE 窗口 · 功率'), ('LATE', 'speed', 'LATE 窗口 · 速度')]
BLUES = LinearSegmentedColormap.from_list('blues', ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])
plt.rcParams.update({'font.size': 9.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 10, 'axes.titlecolor': INK, 'legend.frameon': False, 'axes.unicode_minus': False,
                     'font.sans-serif': ['Arial Unicode MS', 'DejaVu Sans']})

mean = pd.read_csv(R / 'mean_metrics.csv').set_index(['window', 'model', 'target'])
seed = pd.read_csv(R / 'metrics_per_seed.csv')
valley = pd.read_csv(R / 'valley_by_sample.csv')
VLO, VHI = valley.valley_plateau_khz.min(), valley.valley_plateau_khz.max()
views = [json.load(open(p)) for p in glob.glob(str(B.v.VIEWS / '*.json'))]
ewt_b = np.array([b / 1e3 for m in views for b in m['retained_boundaries_hz'] if b > 40000])


def loc(model): return float(model.rsplit('_', 1)[1])


def mm(w, model, t, col='macro_MAPE'): return mean.loc[(w, model, t), col]


def save(fig, name):
    fig.savefig(FIG / name, dpi=170, bbox_inches='tight'); plt.close(fig); print('saved', name)


def fig1():
    z = np.load(R / 'valley_spectra.npz'); f = z['f'] / 1e3
    sm = lambda p: np.convolve(p, np.ones(8) / 8, mode='same')
    fig = plt.figure(figsize=(10, 8.2)); gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 1, .8], hspace=.42)
    a = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1], sharex=a); c = fig.add_subplot(gs[2])
    for p in z['plateau']:
        a.plot(f, 10 * np.log10(sm(p)), color=LIGHT, lw=.5)
    a.plot(f, 10 * np.log10(sm(z['plateau'].mean(0))), color=INK, lw=1.6, label='49 个样本平均')
    a.axvspan(VLO, VHI, color=COL['NAT'], alpha=.18, lw=0)
    a.text(VHI + .8, -21.5, f'谱谷 {VLO:.1f}–{VHI:.1f} kHz\n（49 个样本，激光开启段）', color=INK, fontsize=8.5)
    a.set_xlim(8, 78); a.set_ylim(-40, -18); a.set_ylabel('归一化 PSD [dB]')
    a.set_title('(a) 激光开启段（EARLY 窗口）功率谱：灰线为单个样本，黑线为平均', loc='left')
    a.legend(loc='upper left', fontsize=8)
    n = B.N + 2 * B.PAD; fb = np.abs(np.fft.fftfreq(n, 1 / B.FS))[: n // 2] / 1e3
    h55 = B.old.filters(np.array([55500.]))[:, : n // 2]; h40 = B.old.filters(np.array([40000.]))[:, : n // 2]
    hm = B.meyer_filters(55.5, 2.5)[:, : n // 2]
    edges = [8000, 22000, 36000, 50000, 64000, 78000]
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        w, hh = signal.sosfreqz(signal.butter(4, [lo, hi], btype='bandpass', fs=B.FS, output='sos'), worN=2048, fs=B.FS)
        b.plot(w / 1e3, abs(hh) ** 2, color=COL['FIX5BW'], lw=1, alpha=.9, label=LAB['FIX5BW'] if i == 0 else None)
    for k in range(2):
        b.plot(fb, h40[k], color=COL['NAT_40'], lw=1.6, ls='--', label=LAB['NAT_40'] + '（EWT 原生形状）' if k == 0 else None)
        b.plot(fb, hm[k], color=COL['MEY'], lw=1.6, ls=':', label=LAB['MEY2.5_55.5'] if k == 0 else None)
        b.plot(fb, h55[k], color=COL['NAT'], lw=2.2, label='固定 55.5 kHz（EWT 原生形状，过渡 45.5–65.5 kHz）' if k == 0 else None)
    b.axvspan(VLO, VHI, color=COL['NAT'], alpha=.18, lw=0)
    b.set_ylim(0, 1.5); b.set_yticks([0, .25, .5, .75, 1]); b.set_ylabel('滤波器幅值响应'); b.set_xlabel('频率 [kHz]')
    b.set_title('(b) 各分带方式的滤波器：每条曲线是一个频带分量（低频带 / 高频带）', loc='left')
    b.legend(loc='upper left', fontsize=7.5, ncol=2)
    bins = np.arange(54.25, 57.01, .25)
    c.hist(ewt_b, bins=bins, weights=np.full(len(ewt_b), 100 / len(ewt_b)), color=COL['EWT'], alpha=.85, rwidth=.9,
           label=f'自适应 EWT 在 {len(views)} 个训练子集上估计的边界')
    c.hist(valley.valley_plateau_khz, bins=bins, weights=np.full(len(valley), 100 / len(valley)), histtype='step', color=INK, lw=1.6,
           label='各样本的谱谷中心（49 个样本）')
    c.axvline(55.5, color=COL['NAT'], lw=2, label='固定边界 55.5 kHz')
    c.set_xlim(54.25, 57); c.set_xlabel('频率 [kHz]（放大）'); c.set_ylabel('占比 [%]')
    c.set_title('(c) EWT 每次"自适应"估计出的边界几乎都落在谱谷中心', loc='left'); c.legend(fontsize=7.5, loc='upper right')
    save(fig, 'fig1_spectrum_filters.png')


def fig2():
    z = np.load(R / 'valley_spectra.npz'); f = z['f'] / 1e3
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw=dict(width_ratios=[1.35, .9, 1.1]))
    a = ax[0]; mk = {550: 'o', 700: 's', 850: '^'}
    a.axhspan(ewt_b.min(), ewt_b.max(), color=COL['EWT'], alpha=.15, lw=0)
    a.text(1, ewt_b.max() + .05, 'EWT 边界范围', color=COL['EWT'], fontsize=8)
    for v_, m_ in mk.items():
        q = valley[valley.speed_mm_s == v_]
        a.scatter(q.sample_id, q.valley_plateau_khz, marker=m_, s=34, color=INK, edgecolor='white', lw=.8, label=f'{v_} mm/s 激光开启段', zorder=3)
        a.scatter(q.sample_id, q.valley_tail_khz, marker=m_, s=30, facecolor='none', edgecolor=MUTED, lw=1, zorder=2)
    q = valley[valley.speed_mm_s.isna()]
    a.scatter(q.sample_id, q.valley_plateau_khz, marker='X', s=46, color=INK, label='速度未知（#41, #43）', zorder=3)
    a.scatter([], [], marker='o', facecolor='none', edgecolor=MUTED, label='空心：激光关闭后的衰减尾巴')
    a.axhline(55.5, color=COL['NAT'], lw=1.6)
    a.set_xlabel('样本编号（= 打印顺序）'); a.set_ylabel('谱谷中心频率 [kHz]'); a.set_ylim(54.2, 57)
    a.set_title('(a) 谱谷位置不随参数变化，激光关闭后仍在原处', loc='left'); a.legend(fontsize=7, ncol=2, loc='upper right')
    b = ax[1]; M = np.full((7, 7), np.nan); S = np.zeros((7, 7), int)
    for r in valley.itertuples():
        M[r.row, r.col] = r.valley_plateau_khz; S[r.row, r.col] = r.sample_id
    im = b.imshow(M, cmap=BLUES, vmin=54.5, vmax=56.5)
    for i in range(7):
        for j in range(7):
            b.text(j, i, f'#{S[i, j]}\n{M[i, j]:.1f}', ha='center', va='center', fontsize=6.3, color='white' if M[i, j] > 55.7 else INK)
    b.set_xticks([]); b.set_yticks([]); b.grid(False)
    for s in b.spines.values(): s.set_visible(False)
    cb = fig.colorbar(im, ax=b, fraction=.046, pad=.03); cb.set_label('谱谷中心 [kHz]'); cb.outline.set_visible(False)
    b.set_title('(b) 打印板上的分布（按课程说明排布）', loc='left')
    c = ax[2]; band = (f >= 30) & (f <= 50); sm = lambda p: np.convolve(p, np.ones(16) / 16, mode='same')
    for p, lab, st, cc in [(z['plateau'].mean(0), '激光开启段（49 样本平均）', '-', INK), (z['tail'].mean(0), '激光关闭后衰减尾巴', '--', MUTED),
                           (z['idle'], '空闲噪声（无打印）', ':', COL['RAW'])]:
        p = sm(p); c.plot(f, 10 * np.log10(p / p[band].mean()), ls=st, color=cc, lw=1.7, label=lab)
    c.axvspan(VLO, VHI, color=COL['NAT'], alpha=.18, lw=0)
    c.annotate('8 kHz 整数倍电气谐波\n（48、56 kHz）', xy=(56, 3.5), xytext=(58.5, 6), fontsize=7.5, color=MUTED, arrowprops=dict(arrowstyle='-', color=MUTED, lw=.8))
    c.set_xlim(40, 72); c.set_ylim(-14, 8); c.set_xlabel('频率 [kHz]'); c.set_ylabel('相对 30–50 kHz 平均 [dB]')
    c.set_title('(c) 谷出现在有声激励时，空闲噪声里没有', loc='left'); c.legend(fontsize=7.5, loc='lower left')
    fig.tight_layout(); save(fig, 'fig2_valley_invariance.png')


def fig3():
    models = ['RAW', 'FIX5BW', 'NAT_40', 'NAT_55.5', 'MEY2.5_55.5', 'EWT']
    fig, ax = plt.subplots(2, 4, figsize=(14, 5.6), sharey=True)
    for r, metric, mname in [(0, 'macro_MAPE', '工况宏平均 MAPE [%]'), (1, 'group_MAPE', '组平均 MAPE [%]')]:
        for cidx, (w, t, title) in enumerate(CELLS):
            a = ax[r, cidx]
            for i, m in enumerate(models):
                q = seed[(seed.window == w) & (seed.model == m) & (seed.target == t)][metric]
                a.plot([q.min(), q.max()], [i, i], color=COL[m], lw=2.2, alpha=.55, solid_capstyle='round')
                a.scatter(q.mean(), i, s=60, color=COL[m], edgecolor='white', lw=1.2, zorder=3, marker='D' if m == 'EWT' else 'o')
                a.text(q.mean(), i + .3, f'{q.mean():.1f}', ha='center', fontsize=7.5, color=INK)
            base = mm(w, 'TRAIN_MEDIAN', t, metric)
            a.axvline(base, color=MUTED, ls='--', lw=1.2)
            if cidx == 0 or True:
                a.text(base, len(models) - .35, '中位数基线', color=MUTED, fontsize=7, ha='center')
            a.set_yticks(range(len(models))); a.set_yticklabels([LAB[m] for m in models], fontsize=8.5); a.set_ylim(-.6, len(models) - .1)
            a.set_xlabel(mname, fontsize=8.5)
            if r == 0: a.set_title(title)
    fig.suptitle('图中点 = 3 个随机种子平均，横线 = 种子间最小–最大；越靠左越好', y=1.0, fontsize=9, color=MUTED)
    fig.tight_layout(); save(fig, 'fig3_main_comparison.png')


def fig4():
    nat = sorted({loc(m) for m in B.MODELS if m.startswith('NAT_')}); mey = sorted(B.SWEEP)
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.2), sharex=True)
    for a, (w, t, title) in zip(ax.flat, CELLS):
        a.axvspan(VLO, VHI, color=COL['NAT'], alpha=.15, lw=0)
        lo = [mean.loc[(w, B.nat(b), t), 'macro_MAPE_seed_min'] for b in nat]; hi = [mean.loc[(w, B.nat(b), t), 'macro_MAPE_seed_max'] for b in nat]
        a.fill_between(nat, lo, hi, color=COL['NAT'], alpha=.12, lw=0)
        a.plot(nat, [mm(w, B.nat(b), t) for b in nat], '-o', color=COL['NAT'], ms=4.5, lw=1.8, label='固定边界：EWT 原生滤波器形状（阴影 = 种子范围）')
        a.plot(mey, [mm(w, B.mey(2.5, b), t) for b in mey], '-s', color=COL['MEY'], ms=4.5, lw=1.5, label='固定边界：恒定窄过渡（±2.5 kHz）')
        a.scatter(55.55, mm(w, 'EWT', t), marker='D', s=70, facecolor='white', edgecolor=COL['EWT'], lw=2, zorder=4, label='自适应 EWT')
        a.scatter(40, mm(w, 'NAT_40', t), s=110, facecolor='none', edgecolor=COL['NAT_40'], lw=2, zorder=4, label='已有固定 40 kHz')
        a.axhline(mm(w, 'RAW', t), color=COL['RAW'], ls=':', lw=1.4, label='不分带')
        a.axhline(mm(w, 'FIX5BW', t), color=COL['FIX5BW'], ls='-.', lw=1.3, label='已有 5 带 Butterworth')
        a.axhline(mm(w, 'TRAIN_MEDIAN', t), color=MUTED, ls='--', lw=1.2, label='训练中位数基线')
        a.set_title(title); a.set_ylabel('工况宏平均 MAPE [%]')
    for a in ax[1]: a.set_xlabel('分带边界位置 [kHz]（绿色带 = 实测谱谷范围）')
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, .07, 1, 1)); fig.legend(h, l, loc='lower center', ncol=4, fontsize=8)
    save(fig, 'fig4_location_sweep.png')


def fig5():
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    avg = lambda model: np.mean([mm(w, model, t) for w, t, _ in CELLS])
    a = ax[0]; nat = sorted({loc(m) for m in B.MODELS if m.startswith('NAT_')})
    a.axvspan(VLO, VHI, color=COL['NAT'], alpha=.15, lw=0)
    a.plot(nat, [avg(B.nat(b)) for b in nat], '-o', color=COL['NAT'], ms=4.5, lw=1.8, label='EWT 原生形状')
    a.plot(B.SWEEP, [avg(B.mey(2.5, b)) for b in B.SWEEP], '-s', color=COL['MEY'], ms=4.5, lw=1.5, label='恒定窄过渡 ±2.5 kHz')
    a.scatter(55.55, avg('EWT'), marker='D', s=70, facecolor='white', edgecolor=COL['EWT'], lw=2, zorder=4, label='自适应 EWT')
    for m, ls in [('RAW', ':'), ('FIX5BW', '-.'), ('TRAIN_MEDIAN', '--')]:
        a.axhline(avg(m), color=COL.get(m, MUTED), ls=ls, lw=1.3, label=LAB[m])
    a.set_xlabel('分带边界位置 [kHz]'); a.set_ylabel('4 个窗口×目标的平均宏 MAPE [%]')
    a.set_title('(a) 两种滤波器形状下，谱谷附近都是最稳的区域', loc='left'); a.legend(fontsize=7.2)
    b = ax[1]; widths = B.WIDTHS
    for (w, t, title), mk in zip(CELLS, ['o', 's', '^', 'v']):
        y = [mm(w, B.mey(d, 55.5), t) - mm(w, 'NAT_55.5', t) for d in widths]
        b.plot(widths, y, marker=mk, color=MUTED, lw=1, ms=5, label=title)
    b.plot(widths, [avg(B.mey(d, 55.5)) - avg('NAT_55.5') for d in widths], '-', color=INK, lw=2.4, label='4 项平均')
    b.axhline(0, color=COL['NAT'], lw=1.6); b.text(.52, -.2, '0 = 原生形状（半宽 10.0 kHz）', color=COL['NAT'], fontsize=7.5)
    b.set_xscale('log'); b.set_xticks(widths); b.set_xticklabels([f'±{d:g}' for d in widths])
    b.set_xlabel('边界固定在 55.5 kHz 时的过渡带半宽 [kHz]'); b.set_ylabel('相对原生形状的 MAPE 差 [百分点]')
    b.set_title('(b) 过渡带宽窄影响不大（≤ 2 个百分点）', loc='left'); b.legend(fontsize=7, loc='upper left')
    c = ax[2]; local = sorted(B.LOCAL)
    for (w, t, title), mk in zip(CELLS, ['o', 's', '^', 'v']):
        c.plot(local, [mm(w, B.nat(b), t) - mm(w, 'NAT_55.5', t) for b in local], marker=mk, color=MUTED, lw=1, ms=5, label=title)
        c.scatter(55.55, mm(w, 'EWT', t) - mm(w, 'NAT_55.5', t), marker='D', s=46, facecolor='white', edgecolor=COL['EWT'], lw=1.6, zorder=4)
    c.scatter([], [], marker='D', facecolor='white', edgecolor=COL['EWT'], label='自适应 EWT')
    c.axhspan(-1, 1, color=COL['NAT'], alpha=.1, lw=0); c.axhline(0, color=COL['NAT'], lw=1.4)
    c.plot(ewt_b, np.full(len(ewt_b), -2.6), '|', color=COL['EWT'], ms=10, alpha=.25)
    c.text(54.55, -2.35, 'EWT 估计的边界（每道竖线一个训练子集）', color=COL['EWT'], fontsize=7)
    c.set_xlabel('边界位置 [kHz]（谱谷内的微小移动）'); c.set_ylabel('相对固定 55.5 kHz 的 MAPE 差 [百分点]')
    c.set_ylim(-2.9, 2.3); c.set_title('(c) 边界挪动 <1 kHz 的抖动 ≈ EWT 与固定的差别', loc='left'); c.legend(fontsize=7, loc='upper left', ncol=2)
    fig.tight_layout(); save(fig, 'fig5_robustness.png')


def fig6():
    per = pd.read_csv(R / 'condition_errors.csv'); m = per.groupby(['window', 'target', 'model', 'condition']).APE.mean()
    pr = pd.read_csv(R / 'paired_comparisons.csv').set_index(['window', 'target', 'new', 'baseline'])
    pairs = [('EWT', 'NAT_55.5', '自适应 EWT\n− 固定 55.5', COL['EWT']), ('NAT_55.5', 'NAT_40', '固定 55.5\n− 已有 40 kHz', COL['NAT_40']),
             ('NAT_55.5', 'FIX5BW', '固定 55.5\n− 已有 5 带', COL['FIX5BW']), ('NAT_55.5', 'RAW', '固定 55.5\n− 不分带', COL['RAW'])]
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.4), sharey='row')
    rng = np.random.default_rng(0)
    for a, (w, t, title) in zip(ax.flat, CELLS):
        a.axhspan(-1, 1, color=COL['NAT'], alpha=.12, lw=0)
        for i, (new, base, lab, cc) in enumerate(pairs):
            d = (m.loc[w, t, new] - m.loc[w, t, base]).to_numpy()
            a.scatter(i + rng.uniform(-.13, .13, len(d)), d, s=30, color=cc, edgecolor='white', lw=1, zorder=3, alpha=.9)
            r = pr.loc[(w, t, new, base)]
            a.plot([i + .28, i + .28], [r.ci95_low, r.ci95_high], color=INK, lw=1.6)
            a.plot([i + .2, i + .36], [r.mean_delta_pp] * 2, color=INK, lw=2.4)
            a.text(i + .4, r.mean_delta_pp, f'{r.mean_delta_pp:+.2f}\np={r.wilcoxon_p:.3f}', fontsize=7, va='center', color=INK)
        a.axhline(0, color=MUTED, lw=1)
        a.set_xticks(range(len(pairs))); a.set_xticklabels([p[2] for p in pairs], fontsize=8); a.set_xlim(-.5, len(pairs) - .05)
        a.set_title(title); a.set_ylabel('MAPE 差 [百分点]（< 0：前者更好）')
    fig.text(.5, -.01, '每个点 = 一个留出工况（3 个种子平均）；黑短线 = 均值，黑竖线 = 95% bootstrap 置信区间；绿色带 = ±1 个百分点等效区', ha='center', color=MUTED, fontsize=8)
    fig.tight_layout(); save(fig, 'fig6_paired_conditions.png')


def fig7():
    s = pd.read_csv(R / 'separability.csv'); s = s[s.window == 'EARLY']
    raw = s[s['shape'] == 'RAW'].set_index('feature')
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, items, metric, ref, title in [
            (ax[0], [('mode2_energy_ratio', 'NAT', '-', '低频带能量占比（原生形状）'), ('mode2_energy_ratio', 'MEY2.5', '--', '低频带能量占比（窄过渡）')],
             'omega2_speed', ('raw_log_rms', '原始信号最佳单特征（log RMS）'), '(a) 速度：能量占比在谱谷处有局部峰，但不是全局最高'),
            (ax[1], [('mode1_kurtosis', 'NAT', '-', '高频带峭度（原生形状）'), ('mode1_kurtosis', 'MEY2.5', '--', '高频带峭度（窄过渡）'),
                     ('mode2_crest', 'NAT', ':', '低频带峰值因子（原生形状）')],
             'omega2_power_given_speed', ('raw_kurtosis', '原始信号峭度'), '(b) 给定速度后的功率：高频带峭度在 50–56 kHz 一带最高')]:
        a.axvspan(VLO, VHI, color=COL['NAT'], alpha=.15, lw=0)
        for feat, shape, ls, lab in items:
            z = s[(s['shape'] == shape) & (s.feature == feat)].sort_values('boundary_khz')
            a.plot(z.boundary_khz, z[metric], ls=ls, color=COL['NAT'] if shape == 'NAT' else COL['MEY'], lw=1.8, label=lab)
        a.axhline(raw.loc[ref[0], metric], color=COL['RAW'], ls='-.', lw=1.3, label=ref[1])
        a.set_xlabel('分带边界位置 [kHz]'); a.set_ylabel('ω²（越大越能区分工况）'); a.set_title(title, loc='left'); a.legend(fontsize=7.5)
    fig.text(.5, -.02, 'EARLY 窗口，46 组已知样本；不训练模型的描述性分析。高功率工况都在板子边缘，ω² 中混有位置效应。', ha='center', color=MUTED, fontsize=8)
    fig.tight_layout(); save(fig, 'fig7_separability.png')


def fig8():
    f = pd.read_csv(R / 'selection_frequency.csv'); f = f[f.window == 'EARLY']
    models = ['RAW', 'FIX5BW', 'NAT_40', 'NAT_55.5', 'MEY2.5_55.5', 'EWT']
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for a, t, title in [(ax[0], 'power', '功率模型（每格 = 42 次外层拟合中被选中的比例 %）'), (ax[1], 'speed', '速度模型')]:
        z = f[f.target == t]; top = z.groupby('feature')['count'].sum().nlargest(11).index.tolist()
        M = z.pivot_table(index='feature', columns='model', values='percent').reindex(index=top, columns=models).fillna(0)
        im = a.imshow(np.ma.masked_equal(M.to_numpy(), 0), cmap=BLUES, vmin=0, vmax=100, aspect='auto'); a.set_facecolor('white')
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = M.iat[i, j]
                if val > 0: a.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=7.5, color='white' if val > 55 else INK)
        a.set_xticks(range(len(models))); a.set_xticklabels(['不分带', '已有\n5 带', '已有\n40 kHz', '固定\n55.5 kHz', '固定 55.5\n窄过渡', '自适应\nEWT'], fontsize=8)
        a.set_yticks(range(len(top))); a.set_yticklabels(top, fontsize=8); a.grid(False); a.set_title(title, loc='left')
        for s_ in a.spines.values(): s_.set_visible(False)
    fig.text(.5, -.01, 'mode1 = 高频分量，mode2 = 低频分量（FIX5BW 为 mode1–mode5，从高到低）；raw = 未分解的原始窗口', ha='center', color=MUTED, fontsize=8)
    fig.tight_layout(); save(fig, 'fig8_feature_selection.png')


if __name__ == '__main__':
    for fn in [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8]:
        fn()

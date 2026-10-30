"""Band-split definitions shared by the Task 1 scripts.

Model names
  RAW              no split, 9 descriptors of the raw window
  EWT              adaptive EWT_STABLE: boundaries re-estimated on every training view (cached views)
  NAT_<b>          fixed boundary b kHz, EWT-native Meyer filter (transition (1-g)b..(1+g)b, g=(pi-w)/(pi+w))
  MEY<d>_<b>       fixed boundary b kHz, Meyer filter with constant half-width d kHz (location and width decoupled)
  FIX5BW           the package's existing 5 equal-width Butterworth bands (8-22-36-50-64-78 kHz), cached modes
Every model feeds the same 9-descriptor bank, group median, SFS and SVR as EWT_STABLE_FREQ.
"""
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'ewt_verification'))
import verify as v  # noqa: E402  (also imports ewt_complete/run.py as v.old)

old = v.old
FS, N, PAD = old.FS, old.N, old.PAD
DECOMP = old.OLD / 'results' / 'decompositions'

SWEEP = [30.0, 32.5, 35.0, 37.5, 40.0, 42.5, 45.0, 47.5, 50.0, 52.5, 55.5, 57.5, 60.0, 62.5, 65.0, 67.5, 70.0]
LOCAL = [54.5, 55.0, 55.25, 55.5, 55.625, 55.75, 56.0, 56.5]
WIDTHS = [0.5, 1.0, 2.5, 5.0, 10.0]
SWEEP_WIDTH = 2.5


def nat(b): return f'NAT_{b:g}'
def mey(d, b): return f'MEY{d:g}_{b:g}'


MODELS = (['RAW', 'EWT', 'FIX5BW']
          + [nat(b) for b in sorted(set(SWEEP) | set(LOCAL))]
          + [mey(SWEEP_WIDTH, b) for b in SWEEP]
          + [mey(d, 55.5) for d in WIDTHS if d != SWEEP_WIDTH])


def native_gamma(b_khz, n=N + 2 * PAD):
    """Transition ratio used by pyewt lp_filterbank for a single boundary."""
    w = b_khz * 1e3 * 2 * np.pi / FS
    return (1 - 1 / n) * (np.pi - w) / (np.pi + w)


def beta(x):
    x = np.clip(x, 0, 1)
    return x ** 4 * (35 - 84 * x + 70 * x ** 2 - 20 * x ** 3)


def meyer_filters(b_khz, d_khz, n=N + 2 * PAD):
    """Tight-frame two-band Meyer split (low^2 + high^2 = 1), ordered [high, low] like the EWT code."""
    f = np.abs(np.fft.fftfreq(n, 1 / FS)) / 1e3
    t = beta((f - (b_khz - d_khz)) / (2 * d_khz))
    return np.array([np.sin(np.pi / 2 * t), np.cos(np.pi / 2 * t)])


def butter5(x):
    """Re-implementation of window_filter_robustness/run.py fixed(): 5 equal-width 8-78 kHz Butterworth bands, high first."""
    from scipy import signal
    edges = np.array([8000, 22000, 36000, 50000, 64000, 78000]); pad = x.shape[-1] // 2
    z = np.pad(x, ((0, 0), (pad, pad)), mode='reflect')
    return np.stack([signal.sosfiltfilt(signal.butter(4, [lo, hi], btype='bandpass', fs=FS, output='sos'), z, axis=-1, padtype=None)[:, pad:-pad]
                     for lo, hi in zip(edges[:-1], edges[1:])][::-1], axis=1)


def apply_filters(x, h):
    xp = np.pad(x, ((0, 0), (PAD, PAD)), mode='reflect')
    return np.fft.ifft(np.fft.fft(xp, axis=-1)[:, None, :] * h[None], axis=-1).real[:, :, PAD:-PAD]


def components(model, x, fix5=None):
    """Band components (n, K, N) of windows x for a fixed-split model."""
    if model == 'RAW':
        return np.empty((len(x), 0, x.shape[-1]))
    if model == 'FIX5BW':
        return fix5
    if model.startswith('NAT_'):
        u, _ = old.transform(x, np.array([float(model[4:]) * 1e3]))
        return u
    if model.startswith('MEY'):
        d, b = model[3:].split('_')
        return apply_filters(x, meyer_filters(float(b), float(d)))
    raise ValueError(model)


_X, _FIX5 = {}, {}


def fix5_modes(w):
    if w not in _FIX5:
        d = v.data(w)
        _FIX5[w] = np.array([np.load(DECOMP / f'{w}_{sid:02}_{tid}.npz')['FIXED'] for sid in d['ids'] for tid in range(1, 6)])
    return _FIX5[w]


def matrix(model, w, view_key=None):
    """Group-level (46 x 54) feature matrix; EWT needs the training view key, fixed splits ignore it."""
    if model == 'EWT':
        with np.load(v.VIEWS / f'{w}_{view_key}.npz') as z:
            return z['EWT_STABLE_FREQ']
    if (model, w) not in _X:
        x = v.data(w)['x']
        u = components(model, x, fix5_modes(w) if model == 'FIX5BW' else None)
        _X[model, w] = old.aggregate(v.bank(x, u))
    return _X[model, w]

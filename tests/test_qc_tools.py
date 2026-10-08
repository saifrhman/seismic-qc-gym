import numpy as np
import pytest

from seismic_qc_gym import qc
from seismic_qc_gym.defects import shift_samples
from seismic_qc_gym.fixes import despike, lowcut, notch
from seismic_qc_gym.synth import REGIMES, make_survey


def _gather(regime="A", seed=21):
    return make_survey(REGIMES[regime], 1, np.random.default_rng(seed)).gathers[0]


def _row(table, i):
    return next(r for r in table["rows"] if r[0] == i)


@pytest.mark.parametrize("regime", ["A", "B"])
def test_consistency_sees_polarity_and_timing(regime):
    g = _gather(regime)
    g.data[30] *= -1
    k = int(round(0.024 / g.dt))  # 24 ms late
    g.data[60] = shift_samples(g.data[60], k)
    out = qc.neighbour_consistency(g)
    _, cl, cr, _ = _row(out, 30)
    assert cl < -0.5 and cr < -0.5
    _, cl, cr, shift = _row(out, 60)
    assert cl > 0.3 and cr > 0.3
    assert shift == pytest.approx(k * g.dt * 1000.0, abs=g.dt * 1000.0)
    # healthy traces away from the defects show no shift
    clean_shifts = [abs(r[3]) for r in out["rows"] if r[0] not in (29, 30, 31, 58, 59, 60, 61, 62) and r[3] is not None]
    assert np.median(clean_shifts) < 2.0


def test_first_break_check_recovers_a_header_shift():
    g = _gather("A")
    assert abs(qc.first_break_check(g)["fit"]["intercept_ms"]) < 8.0
    g.offsets = g.offsets + 75.0
    fit = qc.first_break_check(g)["fit"]
    assert fit["intercept_ms"] < -30.0
    assert fit["suggested_offset_correction_m"] == pytest.approx(-75.0, abs=g.group_interval)


def test_spectrum_flags_powerline_and_swell():
    g = _gather("A")
    t = np.arange(g.n_samples) * g.dt
    rms = qc.trace_rms(g.data)
    for i in range(10, 16):
        g.data[i] += (0.5 * rms[i] * np.sqrt(2) * np.sin(2 * np.pi * 50 * t)).astype(np.float32)
    slow = np.sin(2 * np.pi * 2.5 * t)
    for i in range(70, 75):
        g.data[i] += (3 * rms[i] * slow).astype(np.float32)
    out = qc.spectrum_check(g)
    med_line = out["gather_median"]["line_50hz_ratio"]
    assert all(_row(out, i)[3] > 20 * med_line for i in range(10, 16))
    assert all(_row(out, i)[2] > 0.2 for i in range(70, 75))
    assert _row(out, 40)[2] < 0.05 and _row(out, 40)[3] < 20


def test_attributes_see_dead_spike_and_gain():
    g = _gather("A")
    rms = qc.trace_rms(g.data)
    g.data[20] = 0
    g.data[40, 300] += 30 * rms[40]
    g.data[70] *= 4.0
    out = qc.trace_attributes(g)
    assert _row(out, 20)[3] < 0.05 and _row(out, 20)[5] == 1.0
    assert _row(out, 40)[4] > 50
    assert _row(out, 70)[2] > 3 and _row(out, 70)[3] > 3
    assert _row(out, 50)[4] < 15


def test_repairs_undo_their_defects():
    g = _gather("A")
    x0 = g.data[40].astype(float)
    rms = qc.trace_rms(g.data)[40]
    spiky = x0.copy()
    spiky[[200, 480]] += [25 * rms, -30 * rms]
    fixed, n = despike(spiky)
    assert n == 2
    assert np.linalg.norm(fixed - x0) < 0.05 * np.linalg.norm(spiky - x0)
    # filters barely touch a healthy trace
    assert np.linalg.norm(despike(x0)[0] - x0) < 1e-9
    assert np.linalg.norm(lowcut(x0, g.dt) - x0) < 0.15 * np.linalg.norm(x0)
    assert np.linalg.norm(notch(x0, g.dt) - x0) < 0.15 * np.linalg.norm(x0)

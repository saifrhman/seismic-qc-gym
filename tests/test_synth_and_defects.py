import numpy as np
import pytest

from seismic_qc_gym import qc
from seismic_qc_gym.defects import COMPOUNDABLE, DEFECT_TYPES, RUN_DEFECTS, inject_defects, shift_samples
from seismic_qc_gym.synth import REGIMES, make_survey


def test_survey_is_deterministic_by_seed():
    a = make_survey(REGIMES["A"], 2, np.random.default_rng(7))
    b = make_survey(REGIMES["A"], 2, np.random.default_rng(7))
    c = make_survey(REGIMES["A"], 2, np.random.default_rng(8))
    assert np.array_equal(a.gathers[1].data, b.gathers[1].data)
    assert not np.array_equal(a.gathers[1].data, c.gathers[1].data)


@pytest.mark.parametrize("regime", ["A", "B"])
def test_shapes_and_headers(regime):
    r = REGIMES[regime]
    s = make_survey(r, 3, np.random.default_rng(1))
    assert len(s.gathers) == 3
    for g in s.gathers:
        assert g.data.shape == (r.n_traces, r.n_samples)
        assert g.data.dtype == np.float32
        assert np.allclose(np.diff(g.offsets), r.group_interval)
        assert g.offsets[0] == pytest.approx(r.near_offset)


@pytest.mark.parametrize("regime", ["A", "B"])
def test_first_breaks_follow_the_direct_wave(regime):
    s = make_survey(REGIMES[regime], 1, np.random.default_rng(11))
    g = s.gathers[0]
    picks = qc.pick_first_breaks(g)
    true = g.offsets / s.meta["water_velocity"] * 1000.0
    near = g.offsets <= 900
    # within 1.5 samples of the true direct-wave time; interference with the water-bottom
    # reflection at the far end of this range biases some picks by a few ms
    assert np.nanmedian(np.abs(picks[near] - true[near])) < 1.5 * g.dt * 1000.0
    assert abs(qc.first_break_check(g)["fit"]["intercept_ms"]) < 8.0


def test_injection_is_deterministic(clean_a):
    a, ta = inject_defects(clean_a, np.random.default_rng(5), "hard")
    b, tb = inject_defects(clean_a, np.random.default_rng(5), "hard")
    assert [d.to_dict() for d in ta] == [d.to_dict() for d in tb]
    assert np.array_equal(a.gathers[0].data, b.gathers[0].data)


def test_clean_survey_is_not_modified(clean_a):
    before = clean_a.gathers[0].data.copy()
    inject_defects(clean_a, np.random.default_rng(5), "easy")
    assert np.array_equal(clean_a.gathers[0].data, before)


def _single(clean, dtype, seed=3, difficulty="medium"):
    bad, truth = inject_defects(clean, np.random.default_rng(seed), difficulty, defect_types=(dtype,), geometry_prob=1.0)
    return bad, truth


def test_each_trace_defect_matches_its_definition(clean_a):
    for dtype in ("dead_trace", "reversed_polarity", "time_shift", "gain_error"):
        bad, truth = _single(clean_a, dtype)
        assert truth and all(d.defect_type == dtype for d in truth)
        for d in truth:
            t = d.traces[0]
            x0 = clean_a.gather(d.gather_id).data[t]
            x = bad.gather(d.gather_id).data[t]
            if dtype == "dead_trace":
                assert np.all(x == 0)
            elif dtype == "reversed_polarity":
                assert np.allclose(x, -x0)
            elif dtype == "time_shift":
                assert np.allclose(x, shift_samples(x0, d.params["shift_samples"]))
            elif dtype == "gain_error":
                assert np.allclose(x, x0 * d.params["scale"], rtol=1e-5, atol=1e-6)


def test_only_listed_traces_change(clean_a):
    for dtype in ("spike", "noisy_trace", "swell_noise", "powerline_noise"):
        bad, truth = _single(clean_a, dtype)
        hit = {(d.gather_id, t) for d in truth for t in d.traces}
        for g0, g in zip(clean_a.gathers, bad.gathers):
            for t in range(g.n_traces):
                changed = not np.array_equal(g0.data[t], g.data[t])
                assert changed == ((g.gather_id, t) in hit), (dtype, g.gather_id, t)
        if dtype in RUN_DEFECTS:
            assert all(len(d.traces) >= 3 for d in truth)


def test_geometry_error_shifts_headers_only(clean_a):
    bad, truth = _single(clean_a, "geometry_error")
    assert len(truth) == len(clean_a.gathers)  # forced: no trace defects allowed
    for d in truth:
        g0, g = clean_a.gather(d.gather_id), bad.gather(d.gather_id)
        assert np.array_equal(g0.data, g.data)
        assert np.allclose(g.offsets - g0.offsets, d.params["delta_m"])
        assert abs(d.params["delta_m"]) % g0.group_interval == pytest.approx(0.0)


def test_trace_defects_do_not_overlap_except_compounds(clean_a):
    for seed in range(5):
        _, truth = inject_defects(clean_a, np.random.default_rng(seed), "hard")
        seen = {}
        for d in truth:
            if d.defect_type == "geometry_error":
                continue
            for t in d.traces:
                key = (d.gather_id, t)
                if key in seen:
                    assert seen[key] in COMPOUNDABLE and d.defect_type in COMPOUNDABLE
                seen[key] = d.defect_type


def test_compounds_only_in_hard(clean_a):
    for diff in ("easy", "medium"):
        for seed in range(5):
            _, truth = inject_defects(clean_a, np.random.default_rng(seed), diff)
            keys = [(d.gather_id, d.traces) for d in truth if d.defect_type != "geometry_error"]
            assert len(keys) == len(set(keys))


def test_unknown_inputs_rejected(clean_a):
    with pytest.raises(ValueError):
        inject_defects(clean_a, np.random.default_rng(0), "impossible")
    with pytest.raises(ValueError):
        inject_defects(clean_a, np.random.default_rng(0), "easy", defect_types=("not_a_defect",))
    assert set(DEFECT_TYPES) >= set(COMPOUNDABLE)

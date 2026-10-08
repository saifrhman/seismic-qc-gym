import numpy as np
import pytest

from seismic_qc_gym.synth import REGIMES, make_survey


@pytest.fixture
def clean_a():
    return make_survey(REGIMES["A"], 2, np.random.default_rng(123))


@pytest.fixture
def clean_b():
    return make_survey(REGIMES["B"], 2, np.random.default_rng(456))

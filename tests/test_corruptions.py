import numpy as np
import pytest

from detrel.config import load_config
from detrel.corruptions import CORRUPTION_NAMES, MAX_SEVERITY, corrupt, derive_seed

STOCHASTIC = [c for c in CORRUPTION_NAMES if c != "brightness"]
SEVERITIES = list(range(1, MAX_SEVERITY + 1))


@pytest.fixture(scope="module")
def image() -> np.ndarray:
    """Smooth synthetic scene: gradients plus a bright block, so blur and fog have structure."""
    h, w = 64, 96
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([xx * 255 / w, yy * 255 / h, np.full((h, w), 100.0)], axis=-1)
    img[20:40, 30:60] = 230
    return img.astype(np.uint8)


def test_config_corruptions_are_supported():
    assert set(load_config()["corruptions"]["names"]) <= set(CORRUPTION_NAMES)


@pytest.mark.parametrize("name", CORRUPTION_NAMES)
@pytest.mark.parametrize("severity", SEVERITIES)
def test_shape_and_dtype_preserved(image, name, severity):
    out = corrupt(image, name, severity, seed=0)
    assert out.shape == image.shape
    assert out.dtype == np.uint8


@pytest.mark.parametrize("shape", [(32, 32, 3), (427, 640, 3), (640, 427, 3), (33, 97, 3)])
@pytest.mark.parametrize("name", CORRUPTION_NAMES)
def test_odd_shapes(shape, name):
    img = np.random.RandomState(1).randint(0, 256, shape).astype(np.uint8)
    assert corrupt(img, name, MAX_SEVERITY, seed=0).shape == shape


@pytest.mark.parametrize("name", CORRUPTION_NAMES)
def test_severity_zero_is_identity_copy(image, name):
    out = corrupt(image, name, 0, seed=0)
    np.testing.assert_array_equal(out, image)
    assert out is not image


@pytest.mark.parametrize("name", CORRUPTION_NAMES)
def test_input_not_modified(image, name):
    before = image.copy()
    corrupt(image, name, 3, seed=0)
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize("name", CORRUPTION_NAMES)
def test_same_seed_is_deterministic(image, name):
    a = corrupt(image, name, 3, seed=123)
    b = corrupt(image, name, 3, seed=123)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("name", STOCHASTIC)
def test_different_seed_differs(image, name):
    a = corrupt(image, name, 3, seed=1)
    b = corrupt(image, name, 3, seed=2)
    assert not np.array_equal(a, b)


def test_determinism_independent_of_call_order(image):
    first = corrupt(image, "fog", 2, seed=7)
    for name in CORRUPTION_NAMES:  # disturb any hidden state
        corrupt(image, name, 4, seed=99)
    np.random.seed(0)
    np.testing.assert_array_equal(corrupt(image, "fog", 2, seed=7), first)


@pytest.mark.parametrize("name", CORRUPTION_NAMES)
def test_severity_increases_distortion(image, name):
    def dist(s: int) -> float:
        out = corrupt(image, name, s, seed=0).astype(np.float64)
        return float(np.abs(out - image).mean())

    assert dist(1) < dist(5)


def test_derive_seed_stable_and_distinct():
    # Pinned value: changing derive_seed silently would change every corrupted image.
    assert derive_seed(42, 139, "fog", 3) == 3194829124
    seeds = {derive_seed(42, i, c, s) for i in range(5) for c in CORRUPTION_NAMES for s in (1, 5)}
    assert len(seeds) == 5 * len(CORRUPTION_NAMES) * 2
    assert all(0 <= s < 2**32 for s in seeds)


@pytest.mark.parametrize(
    ("img", "name", "severity", "exc"),
    [
        (np.zeros((64, 64, 3), np.float32), "fog", 1, TypeError),
        (np.zeros((64, 64), np.uint8), "fog", 1, ValueError),
        (np.zeros((64, 64, 4), np.uint8), "fog", 1, ValueError),
        (np.zeros((31, 64, 3), np.uint8), "fog", 1, ValueError),
        (np.zeros((64, 64, 3), np.uint8), "rain", 1, ValueError),
        (np.zeros((64, 64, 3), np.uint8), "fog", 6, ValueError),
        (np.zeros((64, 64, 3), np.uint8), "fog", -1, ValueError),
        (np.zeros((64, 64, 3), np.uint8), "fog", 2.0, ValueError),
        (np.zeros((64, 64, 3), np.uint8), "fog", True, ValueError),
    ],
)
def test_invalid_inputs(img, name, severity, exc):
    with pytest.raises(exc):
        corrupt(img, name, severity, seed=0)


# --- Parity with the original package (runs only where imagecorruptions is installed) --------


@pytest.mark.parametrize("name", CORRUPTION_NAMES)
@pytest.mark.parametrize("severity", [1, 3, 5])
def test_parity_with_reference_imagecorruptions(monkeypatch, name, severity):
    monkeypatch.setattr(np, "float_", np.float64, raising=False)  # reference needs it on NumPy 2
    ref = pytest.importorskip("imagecorruptions")

    img = np.random.RandomState(5).randint(0, 256, (120, 160, 3)).astype(np.uint8)
    np.random.seed(11)  # reference draws from the global RandomState
    expected = ref.corrupt(img, corruption_name=name, severity=severity)
    actual = corrupt(img, name, severity, seed=11)

    diff = np.abs(actual.astype(int) - expected.astype(int))
    if name == "snow":
        # OpenCV's float32 RGB->gray rounds differently in the last bit; allow +-1 on few pixels.
        assert diff.max() <= 1 and (diff > 0).mean() < 0.01
    else:
        np.testing.assert_array_equal(actual, expected)

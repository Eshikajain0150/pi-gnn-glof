import numpy as np

from pignn_glof.data.preprocess import minmax_per_band, velocity_features


def test_preprocessing_shapes_and_ranges():
    optical = np.arange(4 * 6 * 3 * 3, dtype=np.float32).reshape(4, 6, 3, 3)
    normalized = minmax_per_band(optical)
    assert normalized.shape == optical.shape
    assert float(normalized.min()) == 0.0
    assert float(normalized.max()) == 1.0
    vx = np.ones((4, 5), dtype=np.float32)
    vy = np.full((4, 5), 2.0, dtype=np.float32)
    features = velocity_features(vx, vy)
    assert features.shape == (4, 5, 3)
    assert np.allclose(features[..., 2], np.sqrt(5.0))

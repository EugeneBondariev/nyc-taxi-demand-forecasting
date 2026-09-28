import logging
from unittest.mock import patch

import pytest

from src.core.config import MAE_THRESHOLD, MAPE_THRESHOLD
from src.monitoring.monitoring import compare_predictions


@pytest.mark.parametrize(
    "mae,mape,should_retrain",
    [
        (MAE_THRESHOLD - 0.1, MAPE_THRESHOLD - 0.01, False),  # both below threshold
        (MAE_THRESHOLD + 0.1, MAPE_THRESHOLD - 0.01, True),  # mae above, mape below
        (MAE_THRESHOLD - 0.1, MAPE_THRESHOLD + 0.01, True),  # mae below, mape above
        (MAE_THRESHOLD + 0.1, MAPE_THRESHOLD + 0.01, True),  # both above threshold
    ],
)
def test_compare_predictions(mae, mape, should_retrain, caplog):
    with (
        patch("src.monitoring.monitoring.run_training_pipeline"),
        patch("src.monitoring.monitoring.detect_drift", return_value=False),
        caplog.at_level(logging.WARNING),
    ):
        compare_predictions(mae, mape)
    assert ("Retraining triggered" in caplog.text) == should_retrain

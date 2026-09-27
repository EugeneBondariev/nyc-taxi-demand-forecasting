import logging
from unittest.mock import patch

import pytest

from src.config import MAE_THRESHOLD, MAPE_THRESHOLD
from src.monitoring import compare_predictions


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
    with patch("src.monitoring.run_training_pipeline"), caplog.at_level(logging.WARNING):
        compare_predictions(mae, mape)
    assert ("Trigger retraining" in caplog.text) == should_retrain

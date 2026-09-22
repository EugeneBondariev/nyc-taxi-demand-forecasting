import pytest
from unittest.mock import patch
from src.monitoring import compare_predictions
from src.config import MAE_THRESHOLD, MAPE_THRESHOLD


@pytest.mark.parametrize(
    "mae,mape,should_retrain",
    [
        (MAE_THRESHOLD - 0.1, MAPE_THRESHOLD - 0.01, False),  # both below threshold
        (MAE_THRESHOLD + 0.1, MAPE_THRESHOLD - 0.01, True),  # mae above, mape below
        (MAE_THRESHOLD - 0.1, MAPE_THRESHOLD + 0.01, True),  # mae below, mape above
        (MAE_THRESHOLD + 0.1, MAPE_THRESHOLD + 0.01, True),  # both above threshold
    ],
)
def test_compare_predictions(mae, mape, should_retrain, capsys):
    with patch("src.monitoring.run_training_pipeline"):
        compare_predictions(mae, mape)
        output = capsys.readouterr().out
        assert ("Trigger retraining" in output) == should_retrain

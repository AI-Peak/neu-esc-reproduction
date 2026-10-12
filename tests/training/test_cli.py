import importlib.util
from pathlib import Path
import pytest


@pytest.mark.parametrize("variant", ["st_sentiment", "st_topic", "mtl_smart", "mtl_mlm_smart"])
def test_cli_resume_new_process_prediction_and_validation_aggregation(tmp_path, variant):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("module02_cli_smoke", root / "reports/method/cli_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.run_smoke(tmp_path / "cli", variant=variant)
    assert result["status"] == "PASS"
    assert result["validation_aggregation"] == "PASS"
    assert result["validation_metadata"]["variant"] == variant

"""
SageMaker training entrypoint (BYOC).

Contract:
- Input data channel "train": /opt/ml/input/data/train
- Model output: /opt/ml/model (SageMaker will package this as model.tar.gz)

This wrapper reuses the existing project training logic:
    python -m src.training.train
and then copies the produced artifact to SM_MODEL_DIR.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


def main() -> None:
    # SageMaker contract paths
    model_dir = _env_path("SM_MODEL_DIR", "/opt/ml/model")
    train_channel_dir = _env_path("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")

    # Project root (assumes container WORKDIR is repo root)
    project_root = Path.cwd()

    # Expected input file placed in the "train" channel
    input_features = train_channel_dir / "features_train.csv.gz"
    if not input_features.exists():
        raise FileNotFoundError(
            f"Expected training file not found: {input_features}. "
            "Put your prepared features file in the 'train' channel with that name."
        )

    # Copy to the path your existing training script expects
    prep_dir = project_root / "data" / "prep"
    prep_dir.mkdir(parents=True, exist_ok=True)
    local_features = prep_dir / "features_train.csv.gz"
    shutil.copy2(input_features, local_features)

    # Run existing training module (unchanged)
    cmd = [
        "python",
        "-m",
        "src.training.train",
        "--prep-path",
        "data/prep/features_train.csv.gz",
        "--model-out",
        "artifacts/models/final_model.joblib",
    ]
    subprocess.run(cmd, check=True)

    # Verify output exists
    artifacts_model = project_root / "artifacts" / "models" / "final_model.joblib"
    if not artifacts_model.exists():
        raise FileNotFoundError(
            f"Training finished but model artifact not found: {artifacts_model}"
        )

    # Copy to SM_MODEL_DIR so SageMaker packages it
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifacts_model, model_dir / "final_model.joblib")

    print(f"[sagemaker-train] OK -> {model_dir / 'final_model.joblib'}")


if __name__ == "__main__":
    main()

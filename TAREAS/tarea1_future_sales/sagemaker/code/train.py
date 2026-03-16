"""
SageMaker training entrypoint (BYOC).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


def main() -> None:
    model_dir = _env_path("SM_MODEL_DIR", "/opt/ml/model")
    train_channel_dir = _env_path("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")

    project_root = Path.cwd()

    input_features = train_channel_dir / "features_train.csv.gz"
    if not input_features.exists():
        raise FileNotFoundError(
            f"Expected training file not found: {input_features}"
        )

    prep_dir = project_root / "data" / "prep"
    prep_dir.mkdir(parents=True, exist_ok=True)

    local_features = prep_dir / "features_train.csv.gz"
    shutil.copy2(input_features, local_features)

    absolute_prep_path = str(local_features.resolve())
    absolute_model_path = str((project_root / "artifacts" / "models" / "final_model.joblib").resolve())

    cmd = [
        "python",
        "-m",
        "src.training.train",
        "--prep-path",
        absolute_prep_path,
        "--model-out",
        absolute_model_path,
    ]

    subprocess.run(cmd, check=True)

    artifacts_model = Path(absolute_model_path)
    if not artifacts_model.exists():
        raise FileNotFoundError(
            f"Training finished but model artifact not found: {artifacts_model}"
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifacts_model, model_dir / "final_model.joblib")

    print(f"[sagemaker-train] OK -> {model_dir / 'final_model.joblib'}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

INPUT_DIR = Path("/opt/ml/processing/input")
OUTPUT_DIR = Path("/opt/ml/processing/output")
TARGET_COLUMN = "item_cnt_month"


def load_data() -> pd.DataFrame:
    """Carga el archivo preprocesado comprimido."""
    input_file = INPUT_DIR / "features_train.csv.gz"
    logger.info("Cargando archivo desde %s", input_file)

    df = pd.read_csv(input_file, compression="gzip")
    logger.info("Shape original: %s", df.shape)

    return df


def build_splits(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide el dataset en train, validation, test y genera submission_features."""
    logger.info("Construyendo particiones")

    target_candidates = [TARGET_COLUMN, "target", "label", "y"]
    target_col = next((col for col in target_candidates if col in df.columns), None)

    if target_col:
        logger.info("Variable objetivo detectada: %s", target_col)
    else:
        logger.info("No se detectó variable objetivo; se usarán todas las columnas como features")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        shuffle=True,
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        shuffle=True,
    )

    if target_col:
        submission_df = test_df.drop(columns=[target_col]).copy()
    else:
        submission_df = test_df.copy()

    logger.info("train shape: %s", train_df.shape)
    logger.info("validation shape: %s", validation_df.shape)
    logger.info("test shape: %s", test_df.shape)
    logger.info("submission_features shape: %s", submission_df.shape)

    return train_df, validation_df, test_df, submission_df


def save_batch_jsonl(test_df: pd.DataFrame, output_dir: Path) -> None:
    """Guarda un archivo JSONL para Batch Transform compatible con serve.py."""
    batch_dir = output_dir / "batch"
    batch_dir.mkdir(parents=True, exist_ok=True)

    batch_path = batch_dir / "batch.jsonl"

    feature_df = test_df.drop(columns=[TARGET_COLUMN], errors="ignore")

    with open(batch_path, "w", encoding="utf-8") as f:
        for _, row in feature_df.iterrows():
            payload = {"instances": [row.to_dict()]}
            f.write(json.dumps(payload) + "\n")

    logger.info("Archivo batch transform guardado en: %s", batch_path)


def save_outputs(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    submission_df: pd.DataFrame,
) -> None:
    """Guarda los archivos de salida en carpetas separadas para SageMaker Pipelines."""
    train_dir = OUTPUT_DIR / "train"
    validation_dir = OUTPUT_DIR / "validation"
    test_dir = OUTPUT_DIR / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Para compatibilidad con la imagen vieja de training
    train_df.to_csv(
        train_dir / "features_train.csv.gz",
        index=False,
        compression="gzip",
    )

    # Estos se quedan como CSV para el resto del pipeline
    validation_df.to_csv(validation_dir / "validation.csv", index=False)
    test_df.to_csv(test_dir / "test.csv", index=False)

    # Input adicional para Batch Transform en formato JSONL
    save_batch_jsonl(test_df, OUTPUT_DIR)

    logger.info("Archivos guardados correctamente:")
    logger.info(" - %s", train_dir / "features_train.csv.gz")
    logger.info(" - %s", validation_dir / "validation.csv")
    logger.info(" - %s", test_dir / "test.csv")
    logger.info(" - %s", OUTPUT_DIR / "batch" / "batch.jsonl")


def main() -> None:
    df = load_data()
    train_df, validation_df, test_df, submission_df = build_splits(df)
    save_outputs(train_df, validation_df, test_df, submission_df)

    logger.info("Preprocessing terminado exitosamente")


if __name__ == "__main__":
    main()
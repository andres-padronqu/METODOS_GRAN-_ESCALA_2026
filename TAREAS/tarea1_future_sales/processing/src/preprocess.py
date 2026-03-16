from __future__ import annotations

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


def load_data() -> pd.DataFrame:
    """Carga el archivo preprocesado comprimido."""
    input_file = INPUT_DIR / "features_train.csv.gz"
    logger.info("Cargando archivo desde %s", input_file)

    df = pd.read_csv(input_file, compression="gzip")
    logger.info("Shape original: %s", df.shape)

    return df


def build_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide el dataset en train, validation, test y genera submission_features."""
    logger.info("Construyendo particiones")

    # Intenta detectar una variable objetivo común
    target_candidates = ["item_cnt_month", "target", "label", "y"]
    target_col = next((col for col in target_candidates if col in df.columns), None)

    if target_col:
        logger.info("Variable objetivo detectada: %s", target_col)
    else:
        logger.info("No se detectó variable objetivo; se usarán todas las columnas como features")

    # Primer split: train 70%, temp 30%
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        shuffle=True,
    )

    # Segundo split: validation 15%, test 15%
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        shuffle=True,
    )

    # submission_features = test sin la variable objetivo, si existe
    if target_col:
        submission_df = test_df.drop(columns=[target_col]).copy()
    else:
        submission_df = test_df.copy()

    logger.info("train shape: %s", train_df.shape)
    logger.info("validation shape: %s", validation_df.shape)
    logger.info("test shape: %s", test_df.shape)
    logger.info("submission_features shape: %s", submission_df.shape)

    return train_df, validation_df, test_df, submission_df


def save_outputs(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    submission_df: pd.DataFrame,
) -> None:
    """Guarda los cuatro archivos de salida."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / "train.csv"
    validation_path = OUTPUT_DIR / "validation.csv"
    test_path = OUTPUT_DIR / "test.csv"
    submission_path = OUTPUT_DIR / "submission_features.csv"

    train_df.to_csv(train_path, index=False)
    validation_df.to_csv(validation_path, index=False)
    test_df.to_csv(test_path, index=False)
    submission_df.to_csv(submission_path, index=False)

    logger.info("Archivos guardados correctamente:")
    logger.info(" - %s", train_path)
    logger.info(" - %s", validation_path)
    logger.info(" - %s", test_path)
    logger.info(" - %s", submission_path)


def main() -> None:
    df = load_data()
    train_df, validation_df, test_df, submission_df = build_splits(df)
    save_outputs(train_df, validation_df, test_df, submission_df)

    logger.info("Preprocessing terminado exitosamente")


if __name__ == "__main__":
    main()
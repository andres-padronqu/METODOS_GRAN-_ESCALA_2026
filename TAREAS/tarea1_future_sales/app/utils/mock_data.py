from __future__ import annotations

import numpy as np
import pandas as pd


def load_mock_forecasts() -> pd.DataFrame:
    """Create mock forecast data for the Streamlit MVP."""
    rng = np.random.default_rng(42)

    months = pd.date_range("2015-01-01", periods=12, freq="MS")
    rows = []

    for shop_id in range(1, 6):
        for item_id in range(1001, 1011):
            category_id = (item_id % 5) + 1
            base = rng.uniform(5, 30)

            for month in months:
                actual = max(0, base + rng.normal(0, 4))
                prediction = max(0, actual + rng.normal(0, 3))
                naive_prediction = max(0, base + rng.normal(0, 5))

                rows.append(
                    {
                        "shop_id": shop_id,
                        "shop_name": f"Shop {shop_id}",
                        "item_id": item_id,
                        "item_name": f"Item {item_id}",
                        "item_category_id": category_id,
                        "item_category_name": f"Category {category_id}",
                        "month": month,
                        "actual": round(actual, 2),
                        "prediction": round(prediction, 2),
                        "naive_prediction": round(naive_prediction, 2),
                    }
                )

    return pd.DataFrame(rows)


def load_mock_feedback() -> pd.DataFrame:
    """Create mock business feedback data."""
    return pd.DataFrame(
        [
            {
                "item_id": 1001,
                "shop_id": 1,
                "item_category_name": "Category 2",
                "issue_type": "Predicción alta",
                "comment": "La predicción parece alta para la temporada.",
                "created_by": "analista_planeacion",
                "status": "open",
            },
            {
                "item_id": 1004,
                "shop_id": 3,
                "item_category_name": "Category 5",
                "issue_type": "Producto sin venta reciente",
                "comment": "El producto ya casi no se vende en esta tienda.",
                "created_by": "finanzas",
                "status": "open",
            },
        ]
    )
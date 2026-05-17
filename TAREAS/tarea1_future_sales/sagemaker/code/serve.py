"""
SageMaker inference server.

Endpoints required:
GET  /ping
POST /invocations
"""

from __future__ import annotations

import json
import os
from flask import Flask, Response, request

from predictor import Predictor


app = Flask(__name__)

predictor = Predictor(model_dir=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))


@app.route("/ping", methods=["GET"])
def ping():
    """
    Health check endpoint.
    """
    return Response(response="OK", status=200, mimetype="text/plain")


@app.route("/invocations", methods=["POST"])
def invocations():
    """
    Prediction endpoint.
    """
    if request.content_type != "application/json":
        return Response(
            response=json.dumps({"error": "Content-Type must be application/json"}),
            status=415,
            mimetype="application/json",
        )

    payload = request.get_json()

    result = predictor.predict(payload)

    return Response(
        response=json.dumps(result),
        status=200,
        mimetype="application/json",
    )


if __name__ == "__main__":
    port = int(os.environ.get("SAGEMAKER_BIND_TO_PORT", "8080"))
    app.run(host="0.0.0.0", port=port)

import os
import sys

MODELS = {
    "tooth": {
        "id": "tooth-bqo56-ez1w4/1",
        "label": "Tooth detection (coco track)",
    },
    "pano": {
        "id": "pano-fcjgf-nn0ku/1",
        "label": "Panoramic X-ray (yolo track)",
    },
}

DEFAULT_API_URL = "https://serverless.roboflow.com"
# SELF_HOSTED_API_URL = "http://localhost:9001"

OUTPUT_DIR = "outputs"


def load_dotenv_if_available():
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def get_api_key() -> str:
    load_dotenv_if_available()
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print(
            "ROBOFLOW_API_KEY is not set.\n"
            "Export your key: export ROBOFLOW_API_KEY=<your_api_key>\n"
            "Or copy .env.example to .env and add your key.",
            file=sys.stderr,
        )
        sys.exit(1)
    return api_key


def resolve_model_id(model_name: str) -> str:
    if model_name not in MODELS:
        choices = ", ".join(MODELS.keys())
        print(f"Unknown model '{model_name}'. Choose one of: {choices}", file=sys.stderr)
        sys.exit(1)
    return MODELS[model_name]["id"]

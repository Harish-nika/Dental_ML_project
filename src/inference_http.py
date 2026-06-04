from pathlib import Path

from inference_sdk import InferenceHTTPClient

from src.annotate import annotate_and_save, load_image
from src.config import DEFAULT_API_URL, get_api_key


def run_http_inference(
    image_source: str,
    model_id: str,
    output_path: Path,
    display: bool = False,
    api_url: str = DEFAULT_API_URL,
) -> Path:
    image = load_image(image_source)
    client = InferenceHTTPClient(api_url=api_url, api_key=get_api_key())
    results = client.infer(image, model_id=model_id)
    return annotate_and_save(image, results, output_path, display=display)

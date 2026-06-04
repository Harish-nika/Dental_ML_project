from pathlib import Path

from inference import get_model

from src.annotate import annotate_and_save, load_image
from src.config import get_api_key


def run_local_inference(
    image_source: str,
    model_id: str,
    output_path: Path,
    display: bool = False,
) -> Path:
    api_key = get_api_key()
    image = load_image(image_source)
    model = get_model(model_id=model_id, api_key=api_key)
    results = model.infer(image)[0]
    return annotate_and_save(image, results, output_path, display=display)

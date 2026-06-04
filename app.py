#!/usr/bin/env python3
import argparse
from pathlib import Path

from src.config import MODELS, OUTPUT_DIR, resolve_model_id


def build_parser() -> argparse.ArgumentParser:
    model_choices = ", ".join(f"{k} ({v['label']})" for k, v in MODELS.items())
    parser = argparse.ArgumentParser(
        description="Run Roboflow inference on dental images (tooth or pano model)."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODELS.keys()),
        help=f"Model to use: {model_choices}",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to a local image or HTTP(S) URL",
    )
    parser.add_argument(
        "--backend",
        choices=["http", "local"],
        default="http",
        help="http: serverless Roboflow API (default); local: inference package",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for annotated image (default: outputs/<model>_annotated.png)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show annotated image interactively",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    model_id = resolve_model_id(args.model)
    output_path = Path(
        args.output or f"{OUTPUT_DIR}/{args.model}_annotated.png"
    )

    if args.backend == "http":
        from src.inference_http import run_http_inference

        result_path = run_http_inference(
            image_source=args.image,
            model_id=model_id,
            output_path=output_path,
            display=args.display,
        )
    else:
        from src.inference_local import run_local_inference

        result_path = run_local_inference(
            image_source=args.image,
            model_id=model_id,
            output_path=output_path,
            display=args.display,
        )

    print(f"Saved annotated image to {result_path}")


if __name__ == "__main__":
    main()

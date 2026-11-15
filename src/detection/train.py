"""Fine-tune a pretrained YOLO11 model on the exported pilot dataset."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/yolo/dataset.yaml"))
    parser.add_argument("--weights", default="yolo11s.pt", help="Pretrained starting weights")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="mps", help="mps, cpu or a CUDA device")
    parser.add_argument("--name", default="pilot_yolo11s")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Dataset file not found: {args.data} (run src.detection.export_yolo first)")
    model = YOLO(args.weights)
    model.train(
        data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.image_size,
        batch=args.batch, device=args.device, project=str(Path("runs").resolve()),
        name=args.name, patience=20, workers=0, seed=0, exist_ok=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

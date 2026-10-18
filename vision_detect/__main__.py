"""Command line for download, training, evaluation, export, and serving."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from vision_detect.paths import (
    DEFAULT_CHECKPOINT,
    DEFAULT_IMAGE_DIR,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    SUBSET_PATH,
)


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index", type=Path, default=SUBSET_PATH)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vision-detect")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Download COCO annotations and write the subset index")
    prepare.add_argument("--train-images", type=int, default=160)
    prepare.add_argument("--val-images", type=int, default=40)
    prepare.add_argument("--output", type=Path, default=SUBSET_PATH)

    download = sub.add_parser("download", help="Download subset images (no login)")
    _add_data_args(download)

    train = sub.add_parser("train", help="Fine-tune SSDLite or Faster R-CNN")
    _add_data_args(train)
    train.add_argument("--arch", choices=["ssdlite", "fasterrcnn"], default="ssdlite")
    train.add_argument("--epochs", type=int, default=6)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--lr", type=float, default=2e-5)
    train.add_argument("--device", default="auto")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT)

    evaluate = sub.add_parser("evaluate", help="COCO-style mAP and latency")
    _add_data_args(evaluate)
    evaluate.add_argument("--arch", choices=["ssdlite", "fasterrcnn"], default="ssdlite")
    evaluate.add_argument("--checkpoint", type=Path, default=None)
    evaluate.add_argument("--pretrained", action="store_true")
    evaluate.add_argument("--onnx", type=Path, default=None)
    evaluate.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--output", type=Path, default=None)

    export = sub.add_parser("export", help="Export SSDLite to ONNX and check parity")
    export.add_argument("--arch", choices=["ssdlite"], default="ssdlite")
    export.add_argument("--checkpoint", type=Path, default=None)
    export.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    export.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    export.add_argument("--version", default=None)
    export.add_argument("--dataset", default="COCO 2017 person subset")
    export.add_argument("--reference-image", type=Path, default=None)

    cnn = sub.add_parser("cnn-baseline", help="Train the crop-classification CNN")
    _add_data_args(cnn)
    cnn.add_argument("--epochs", type=int, default=6)
    cnn.add_argument("--batch-size", type=int, default=64)
    cnn.add_argument("--lr", type=float, default=1e-3)
    cnn.add_argument("--size", type=int, default=64)
    cnn.add_argument("--seed", type=int, default=42)
    cnn.add_argument("--output", type=Path, default=None)

    predict = sub.add_parser("predict", help="Write sample overlays and browser images")
    _add_data_args(predict)
    predict.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    predict.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    predict.add_argument("--docs", type=Path, default=None)
    predict.add_argument("--web", type=Path, default=None)
    predict.add_argument("--limit", type=int, default=4)
    predict.add_argument("--max-side", type=int, default=800)
    predict.add_argument("--score-threshold", type=float, default=0.4)

    serve = sub.add_parser("serve", help="Run the FastAPI service")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def _hardware() -> dict:
    processor = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                processor = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "platform": platform.platform(),
        "processor": processor,
        "cpu_count": __import__("os").cpu_count(),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        from vision_detect.data.prepare import build_subset

        build_subset(args.train_images, args.val_images, output_path=args.output)
        return
    if args.command == "download":
        from vision_detect.data.download import download_images

        download_images(args.index, args.images)
        return
    if args.command == "train":
        from vision_detect.engine.train import train_command

        train_command(args)
        return
    if args.command == "evaluate":
        _evaluate(args)
        return
    if args.command == "export":
        from vision_detect.engine.export import export_command

        export_command(args)
        return
    if args.command == "cnn-baseline":
        from vision_detect.engine.cnn_baseline import train_cnn

        train_cnn(args)
        return
    if args.command == "predict":
        from vision_detect.engine.predict import default_docs, default_web, render_samples

        if args.docs is None:
            args.docs = default_docs()
        if args.web is None:
            args.web = default_web()
        render_samples(args)
        return
    if args.command == "serve":
        import uvicorn

        uvicorn.run("vision_detect.api.app:app", host=args.host, port=args.port)
        return
    raise SystemExit(f"Unknown command {args.command}")


def _evaluate(args) -> None:
    import cv2
    from torch.utils.data import DataLoader

    from vision_detect.api.inference import OnnxPersonDetector
    from vision_detect.data.dataset import CocoPersonSubset, collate_detections
    from vision_detect.data.records import load_subset, records_for_split
    from vision_detect.engine.evaluate import evaluate_map, pytorch_latency_ms
    from vision_detect.engine.train import resolve_device
    from vision_detect.models.detector import build_model, load_checkpoint

    device = resolve_device(args.device)
    dataset = CocoPersonSubset(args.index, args.images, "val", training=False)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_detections
    )
    report: dict = {"hardware": _hardware(), "val_images": len(dataset)}
    if args.pretrained:
        model = build_model(args.arch, pretrained=True).to(device)
        report["pretrained"] = evaluate_map(model, loader, device)
        report["pretrained_latency"] = pytorch_latency_ms(model, dataset, device)
    if args.checkpoint:
        model = load_checkpoint(args.arch, args.checkpoint).to(device)
        report["checkpoint"] = evaluate_map(model, loader, device)
        report["checkpoint_latency"] = pytorch_latency_ms(model, dataset, device)
    if args.onnx:
        detector = OnnxPersonDetector(args.onnx, args.metadata if args.metadata.exists() else None)
        records = records_for_split(load_subset(args.index), "val")
        first = cv2.imread(str(Path(args.images) / records[0]["file_name"]))
        detector.predict(first)
        times = []
        for record in records[:20]:
            image = cv2.imread(str(Path(args.images) / record["file_name"]))
            times.append(detector.predict(image)["latency_ms"])
        times.sort()
        report["onnx_latency"] = {
            "median_ms": times[len(times) // 2],
            "min_ms": times[0],
            "max_ms": times[-1],
            "images": len(times),
            "includes": "opencv resize, onnxruntime, nms",
        }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

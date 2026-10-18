"""Fine-tune a torchvision detector on the COCO person subset."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from vision_detect.data.dataset import CocoPersonSubset, collate_detections
from vision_detect.engine.evaluate import evaluate_map
from vision_detect.models.detector import build_model


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def run_epoch(model, loader, optimizer, device: torch.device) -> float:
    model.train()
    total = 0.0
    steps = 0
    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        if not torch.isfinite(loss):
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
        steps += 1
    return total / max(steps, 1)


def train_command(args) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    print(f"device={device} arch={args.arch} epochs={args.epochs} batch={args.batch_size}")

    train_set = CocoPersonSubset(args.index, args.images, "train", training=True, seed=args.seed)
    val_set = CocoPersonSubset(args.index, args.images, "val", training=False)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_detections,
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_detections
    )

    model = build_model(args.arch, pretrained=True)
    model.to(device)
    baseline = evaluate_map(model, val_loader, device)
    print(f"pretrained val mAP={baseline['map']:.4f} mAP50={baseline['map_50']:.4f}")
    best_map = baseline["map"] if baseline["map"] is not None else -1.0
    best_state = _clone_state(model)
    best_epoch = 0
    history: list[dict] = []

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    for epoch in range(1, args.epochs + 1):
        loss = run_epoch(model, train_loader, optimizer, device)
        metrics = evaluate_map(model, val_loader, device)
        row = {"epoch": epoch, "loss": loss, **metrics}
        history.append(row)
        score = metrics["map"] if metrics["map"] is not None else -1.0
        print(
            f"epoch {epoch} loss={loss:.4f} mAP={metrics['map']} mAP50={metrics['map_50']}"
        )
        if score > best_map:
            best_map = score
            best_epoch = epoch
            best_state = _clone_state(model)

    model.load_state_dict(best_state)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output)
    document = {
        "arch": args.arch,
        "seed": args.seed,
        "lr": args.lr,
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "device": str(device),
        "train_images": len(train_set),
        "val_images": len(val_set),
        "pretrained": baseline,
        "epochs": history,
        "best_epoch": best_epoch,
        "best_map": best_map,
        "checkpoint": str(output),
    }
    history_path = output.with_suffix(".history.json")
    history_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(output), "best_epoch": best_epoch, "best_map": best_map}))
    return document

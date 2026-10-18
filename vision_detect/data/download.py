"""Download the COCO images named by the subset index. No login required."""

from __future__ import annotations

import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from vision_detect.data.records import load_subset
from vision_detect.paths import DEFAULT_IMAGE_DIR, SUBSET_PATH


def _download_one(url: str, dest: Path, timeout: int = 60) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "vision-detect/0.1"})
    last_error: Exception | None = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as handle:
                while True:
                    chunk = response.read(1 << 16)
                    if not chunk:
                        break
                    handle.write(chunk)
            temporary.replace(dest)
            return
        except Exception as exc:  # noqa: BLE001 - retry network errors
            last_error = exc
            if temporary.exists():
                temporary.unlink()
    raise RuntimeError(f"Failed to download {url}") from last_error


def download_images(
    index_path: Path | str = SUBSET_PATH,
    image_dir: Path | str = DEFAULT_IMAGE_DIR,
    splits: tuple[str, ...] = ("train", "val"),
    workers: int = 8,
) -> int:
    subset = load_subset(index_path)
    directory = Path(image_dir)
    directory.mkdir(parents=True, exist_ok=True)
    pending = [
        record
        for record in subset["images"]
        if record["split"] in splits and not (directory / record["file_name"]).exists()
    ]
    if not pending:
        print(f"Images already present in {directory}")
        return 0
    print(f"Downloading {len(pending)} images to {directory}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_download_one, record["coco_url"], directory / record["file_name"])
            for record in pending
        ]
        done = 0
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 20 == 0 or done == len(pending):
                print(f"  {done}/{len(pending)}")
    return len(pending)

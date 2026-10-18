# vision-detect

[![CI](https://github.com/naveecode/vision-detect/actions/workflows/ci.yml/badge.svg)](https://github.com/naveecode/vision-detect/actions/workflows/ci.yml)
[![Pages](https://github.com/naveecode/vision-detect/actions/workflows/pages.yml/badge.svg)](https://github.com/naveecode/vision-detect/actions/workflows/pages.yml)

Person detection with a fine-tuned SSDLite detector, an ONNX export, a FastAPI service, and an in-browser demo. The code is MIT. Training uses PyTorch and torchvision. There is no Ultralytics dependency.

**Demo (after GitHub Pages is enabled):** [naveecode.github.io/vision-detect](https://naveecode.github.io/vision-detect/)

The detector finds people. Upload a photo in the browser, or POST it to `/detect`. Boxes are drawn with the same score threshold (0.4) and the same NMS IoU (0.55) in Python and in the page.

## Problem

Job descriptions for ML and AI engineering roles ask for OpenCV, CNNs, and object detection. This repo is a small end-to-end pipeline that covers those pieces without a login-gated dataset or an AGPL library:

1. A documented person subset of COCO 2017, downloaded with no account.
2. Fine-tuning of torchvision's SSDLite320 MobileNetV3-Large.
3. COCO-style mean average precision with torchmetrics and pycocotools.
4. OpenCV drawing, plus OpenCV and Albumentations augmentations.
5. An ONNX graph checked against PyTorch, served by FastAPI and by onnxruntime-web.

## Architecture

```mermaid
flowchart LR
  subgraph data [Data]
    ann[COCO 2017 annotations]
    index[Person subset index]
    photos[Images, not committed]
  end
  subgraph train [Train and measure]
    aug[OpenCV and Albumentations]
    ssd[SSDLite320 MobileNetV3]
    score[Person mAP]
  end
  subgraph ship [Ship]
    onnx[ONNX, NMS outside the graph]
    api[FastAPI /detect]
    page[onnxruntime-web]
  end
  ann --> index --> photos --> aug --> ssd --> score
  ssd --> onnx
  onnx --> api
  onnx --> page
```

The training graph keeps torchvision's 91 COCO classes so the pretrained head is not rebuilt. Only person boxes (class index 1) are supervised. The exported graph returns the top 300 person scores and decoded boxes. Non-maximum suppression runs in NumPy for the API and in JavaScript for the page, with the same IoU.

A Faster R-CNN MobileNetV3 variant shares the training loop (`--arch fasterrcnn`). It is not exported. RoIAlign does not run reliably in onnxruntime-web, and the browser demo is the artifact this repo ships.

## Results

Measured on this machine: Linux, Intel Xeon, 4 CPUs, no GPU, torch 2.14.1+cpu, torchvision 0.29.1+cpu. Validation is 40 images. Training is 160 images, AdamW, learning rate 2e-5, weight decay 1e-4, batch size 4, seed 42, 6 epochs, every layer trainable. The checkpoint is the best of the pretrained weights and the six epochs. Epoch 6 won.

These numbers are **person AP on an easy 40-image slice** (one to three non-crowd people, large boxes). They are not the official COCO val2017 80-class mAP. torchvision lists 21.3 box mAP for this SSDLite checkpoint on that full benchmark. Full-precision values are in [`docs/metrics.json`](docs/metrics.json).

| Model | person mAP | mAP@0.50 | mAP@0.75 | AR@100 | Latency, batch 1 |
| --- | ---: | ---: | ---: | ---: | --- |
| COCO-pretrained SSDLite | 0.677 | 0.941 | 0.741 | 0.754 | 29.4 ms PyTorch |
| Fine-tuned, epoch 6 | 0.723 | 0.960 | 0.817 | 0.782 | 28.1 ms PyTorch, 6.0 ms ONNX |

PyTorch latency is the median of 20 single-image forwards, including torchvision's resize and NMS. ONNX latency is the median of 20 images through OpenCV resize, onnxruntime, and NumPy NMS (5.97 ms, range 5.8–7.9 ms). The two timings are not the same work.

| Epoch | Loss | person mAP | mAP@0.50 | mAP@0.75 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.557 | 0.688 | 0.953 | 0.770 |
| 2 | 2.101 | 0.682 | 0.935 | 0.748 |
| 3 | 1.972 | 0.690 | 0.936 | 0.761 |
| 4 | 1.757 | 0.702 | 0.925 | 0.789 |
| 5 | 1.770 | 0.708 | 0.944 | 0.815 |
| 6 | 1.602 | 0.723 | 0.960 | 0.817 |

Six CPU epochs moved person mAP from 0.677 to 0.723 and training loss from 2.56 to 1.60. That is a real gain on this slice and a short run. A free Colab GPU, 15–30 epochs, and a larger subset are the way to push it further. See [Reproduce a stronger number](#reproduce-a-stronger-number).

The shipped ONNX file is the epoch-6 weights, version `ssdlite-coco-person-e6`, 13,840,340 bytes (under 25 MB, not quantized). On a random tensor, the largest absolute difference versus the traced PyTorch module was 9.2e-5. On validation photo `000000021167.jpg` at score 0.3, the official SSD and the export both returned 2 person boxes, with top-box IoU 1.0 and a top-score gap of 0.0. SHA-256: `e60019591e470457e179acfd29d6a090f2e6900b0a351d48bb92e5b8920bd22c`.

### CNN baseline

A 3-convolution network classifies 64×64 crops as person or background. It does not draw boxes, so its accuracy is not mAP. On a balanced validation set (98 crops, half positive) it reached **0.684 accuracy** after 6 epochs. Median batch-1 latency was 0.41 ms. It is an ablation: a small CNN can tell a person crop from background, and it still cannot localize.

### Sample predictions

Score threshold 0.4. Photos are resized copies (longest side 800 px) of COCO val2017 images under CC BY 2.0. The dog in the second photo is not boxed.

| | |
| --- | --- |
| ![Two people, scores 0.999 and 0.994](docs/predictions/000000021167_pred.jpg) | COCO [21167](http://farm4.staticflickr.com/3469/3397730036_36219f4c6e_z.jpg). Attribution License. Two standing people, scores 0.9985 and 0.9939. |
| ![One person walking a dog, score 0.994](docs/predictions/000000022705_pred.jpg) | COCO [22705](http://farm3.staticflickr.com/2338/1940631815_acd4f4458e_z.jpg). Attribution License. One person, score 0.9939. |
| ![Three people, scores 0.986, 0.813, and 0.467](docs/predictions/000000008021_pred.jpg) | COCO [8021](http://farm7.staticflickr.com/6117/6240308255_77782d0723_z.jpg). Attribution License. Three people, including a partial figure at the right edge (0.467). |
| ![Two people, scores 0.985 and 0.762](docs/predictions/000000000872_pred.jpg) | COCO [872](http://farm9.staticflickr.com/8447/7805810128_605424213d_z.jpg). Attribution License. Two people, scores 0.9851 and 0.7619. |

## Dataset and license

[Penn-Fudan](https://www.cis.upenn.edu/~jshi/ped_html/) is a common teaching set, but the official page does not grant a license, and the bundled readme says the copyright holders retain their rights and the images may not be reposted. This repo does not use it.

The index is a deterministic person subset of [COCO 2017](https://cocodataset.org/):

- Annotations: Creative Commons Attribution 4.0. Cite Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014. The zip is checked against MD5 `f4bbac642086de4f52a3fdda2de5fa2c`.
- Images stay under their Flickr licenses. The repo stores annotations for all 200 images and redistributes only four CC BY photos (resized) plus detection overlays. Non-commercial and no-derivatives images are used for local training only.

Selection, in ascending COCO image id: non-crowd person boxes, 1–3 people, short side at least 240, every box at least 1% of the image, the largest box between 8% and 85%. Train is the first 160 qualifying train2017 images. Val is 40 val2017 images, with at least 8 redistributable images reserved first. Person category id and the torchvision class index are both 1.

`https://images.cocodataset.org` fails TLS verification here (the certificate is for `s3.amazonaws.com`). Downloads use path-style `https://s3.amazonaws.com/images.cocodataset.org/...`.

## Quickstart

Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[train,api,dev]"

python -m vision_detect prepare    # annotations only, writes the subset index
python -m vision_detect download   # 200 JPEGs into data/coco-person/images
python -m vision_detect train      # 6 epochs, CPU or CUDA
python -m vision_detect evaluate \
  --pretrained \
  --checkpoint outputs/ssdlite_person.pth \
  --onnx web/model/detector.onnx \
  --output outputs/metrics.json
python -m vision_detect serve      # http://127.0.0.1:8000
```

`prepare` and `download` are not required to run the tests or the API. The subset index and the ONNX model are already in the repo.

```bash
curl -s localhost:8000/health
curl -s localhost:8000/version
curl -s -F "file=@web/samples/000000021167.jpg" localhost:8000/detect
```

`POST /detect` returns boxes in xyxy pixels, labels, scores, image size, latency, and the model version. Optional query `score_threshold` is in `[0, 1]`. Uploads larger than 8 MB are rejected.

Docker serves the ONNX model and does not install PyTorch:

```bash
docker build -t vision-detect .
docker run --rm -p 8000:8000 vision-detect
```

The browser demo is the `web/` directory. From a checkout:

```bash
python -m http.server 8080 --directory web
```

Open `http://127.0.0.1:8080/`. The page loads `onnxruntime-web` 1.30.0 from jsDelivr. GitHub Pages does not send the headers a threaded WASM runtime needs, so the page forces one thread.

### Reproduce a stronger number

[`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) is written for a free Colab GPU. It does not pin the CPU wheel, so Colab's CUDA build stays installed. The notebook trains 12 epochs at batch 16. For a number you can compare with the published 21.3, evaluate on full COCO val2017 rather than this 40-image slice. A larger subset is `python -m vision_detect prepare --train-images N`.

## Design decisions

- **SSDLite, not Faster R-CNN, in the shipped model.** The person head exports to a 13.8 MB graph that matches torchvision boxes on the parity photo. Faster R-CNN remains available for training.
- **NMS stays outside ONNX.** The browser and the API share one IoU and one score threshold. The graph returns decoded boxes in the 320×320 input, clamped to that square.
- **Normalization is inside the graph.** Input is RGB NCHW float32 in `[0, 1]`. SSDLite applies `(x - 0.5) / 0.5`, which is not ImageNet mean and standard deviation.
- **The image is stretched to 320×320.** That matches the SSDLite training transform. It is not letterboxed.
- **API and browser resize with OpenCV and canvas.** mAP and the parity check use torchvision's bilinear interpolate. Parity is exact on the shared 320×320 tensor. The OpenCV path is not claimed to be bit-identical.
- **The 91-class head is kept.** Person-only labels use index 1. Rebuilding the head would drop the COCO classification weights.
- **Augmentation falls back.** Training uses OpenCV brightness and Albumentations horizontal flip plus random brightness/contrast. If an augmentation deletes every box, the original sample is used.
- **CI does not train.** GitHub Actions installs the CPU wheel, runs ruff, and runs pytest on tiny fixtures plus the committed ONNX file. It does not download COCO images.

## Tests

```bash
ruff check .
pytest -q
```

Coverage includes box IoU and NMS, the COCO parser on a tiny fixture, OpenCV and Albumentations transforms, ONNX versus PyTorch on a tiny module, the shipped ONNX file, and the FastAPI contract (health, detect, bad uploads, the score threshold).

## Layout

| Path | Role |
| --- | --- |
| `vision_detect/data` | Subset index, loader, augmentations |
| `vision_detect/models` | SSDLite, Faster R-CNN, exportable person head, crop CNN |
| `vision_detect/engine` | Train, mAP, ONNX export, sample rendering |
| `vision_detect/api` | FastAPI service |
| `web/` | GitHub Pages demo and `model/detector.onnx` |
| `docs/predictions` | Overlays used in this README |
| `docs/metrics.json` | Unrounded measurements |
| `notebooks/train_colab.ipynb` | GPU training notes |

## License

Code: [MIT](LICENSE), Copyright 2026 naveecode.

Annotations in `vision_detect/data/coco_person_subset.json`: CC BY 4.0, COCO Consortium. See [NOTICE](NOTICE). Sample photographs: CC BY 2.0, credited above and in `web/samples/manifest.json`.

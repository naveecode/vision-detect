const INPUT_SIZE = 320;
const ORT_VERSION = "1.30.0";

const stage = document.getElementById("stage");
const stageCtx = stage.getContext("2d");
const statusEl = document.getElementById("status");
const countEl = document.getElementById("count");
const latencyEl = document.getElementById("latency");
const modelEl = document.getElementById("model");
const creditEl = document.getElementById("credit");
const thresholdInput = document.getElementById("threshold");
const thresholdValue = document.getElementById("threshold-value");
const samplesEl = document.getElementById("samples");
const fileInput = document.getElementById("file");

let session = null;
let metadata = {
  nms_iou: 0.55,
  default_score_threshold: 0.4,
  class_name: "person",
  model_version: "",
};
let currentImage = null;
let currentCredit = "";

function setStatus(text) {
  statusEl.textContent = text;
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const union = areaA + areaB - inter;
  return union > 0 ? inter / union : 0;
}

function nms(boxes, scores, iouThreshold) {
  const order = scores.map((_, index) => index).sort((a, b) => scores[b] - scores[a]);
  const suppressed = new Array(scores.length).fill(false);
  const keep = [];
  for (const index of order) {
    if (suppressed[index]) continue;
    keep.push(index);
    for (const other of order) {
      if (other === index || suppressed[other]) continue;
      if (iou(boxes[index], boxes[other]) > iouThreshold) suppressed[other] = true;
    }
  }
  return keep;
}

function preprocess(image) {
  const canvas = document.createElement("canvas");
  canvas.width = INPUT_SIZE;
  canvas.height = INPUT_SIZE;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(image, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const pixels = context.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  const plane = INPUT_SIZE * INPUT_SIZE;
  const chw = new Float32Array(3 * plane);
  for (let index = 0; index < plane; index += 1) {
    const offset = index * 4;
    chw[index] = pixels[offset] / 255;
    chw[plane + index] = pixels[offset + 1] / 255;
    chw[2 * plane + index] = pixels[offset + 2] / 255;
  }
  return chw;
}

function decode(boxes, scores, width, height, threshold) {
  const candidates = [];
  const candidateScores = [];
  for (let index = 0; index < scores.length; index += 1) {
    if (scores[index] < threshold) continue;
    const offset = index * 4;
    candidates.push([boxes[offset], boxes[offset + 1], boxes[offset + 2], boxes[offset + 3]]);
    candidateScores.push(scores[index]);
  }
  const keep = nms(candidates, candidateScores, metadata.nms_iou || 0.55);
  return keep
    .map((index) => {
      const box = candidates[index];
      const scaled = [
        (box[0] * width) / INPUT_SIZE,
        (box[1] * height) / INPUT_SIZE,
        (box[2] * width) / INPUT_SIZE,
        (box[3] * height) / INPUT_SIZE,
      ];
      scaled[0] = Math.min(Math.max(scaled[0], 0), width);
      scaled[2] = Math.min(Math.max(scaled[2], 0), width);
      scaled[1] = Math.min(Math.max(scaled[1], 0), height);
      scaled[3] = Math.min(Math.max(scaled[3], 0), height);
      return { box: scaled, score: candidateScores[index] };
    })
    .filter((item) => item.box[2] > item.box[0] && item.box[3] > item.box[1]);
}

function draw(image, detections) {
  stage.width = image.naturalWidth;
  stage.height = image.naturalHeight;
  stageCtx.clearRect(0, 0, stage.width, stage.height);
  stageCtx.drawImage(image, 0, 0);
  const label = metadata.class_name || "person";
  stageCtx.lineWidth = Math.max(2, Math.round(stage.width / 320));
  stageCtx.font = `${Math.max(14, Math.round(stage.width / 42))}px "Avenir Next", sans-serif`;
  for (const detection of detections) {
    const [x1, y1, x2, y2] = detection.box;
    stageCtx.strokeStyle = "#0c6b58";
    stageCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const text = `${label} ${detection.score.toFixed(2)}`;
    const textY = Math.max(18, y1 - 6);
    stageCtx.fillStyle = "#0c6b58";
    stageCtx.fillText(text, x1 + 2, textY);
  }
}

async function runImage(image) {
  if (!session || !image) return;
  const tensor = new ort.Tensor("float32", preprocess(image), [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const started = performance.now();
  const outputs = await session.run({ images: tensor });
  const elapsed = performance.now() - started;
  const threshold = Number(thresholdInput.value);
  const detections = decode(
    outputs.boxes.data,
    outputs.scores.data,
    image.naturalWidth,
    image.naturalHeight,
    threshold,
  );
  draw(image, detections);
  countEl.textContent = String(detections.length);
  latencyEl.textContent = `${elapsed.toFixed(1)} ms`;
  creditEl.textContent = currentCredit;
  setStatus(
    detections.length
      ? `${detections.length} ${detections.length === 1 ? "person" : "people"} above ${threshold.toFixed(2)}.`
      : "No people above the threshold.",
  );
}

function loadHtmlImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Could not load ${src}`));
    image.src = src;
  });
}

async function useFile(file) {
  currentCredit = "";
  const url = URL.createObjectURL(file);
  currentImage = await loadHtmlImage(url);
  document.querySelectorAll(".samples button").forEach((button) => {
    button.setAttribute("aria-pressed", "false");
  });
  await runImage(currentImage);
}

async function useSample(sample, button) {
  currentCredit = `COCO image ${sample.coco_id}. ${sample.license_name}. ${sample.flickr_url || sample.coco_url}`;
  currentImage = await loadHtmlImage(`samples/${sample.file}`);
  document.querySelectorAll(".samples button").forEach((item) => {
    item.setAttribute("aria-pressed", item === button ? "true" : "false");
  });
  await runImage(currentImage);
}

async function main() {
  try {
    ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    const metaResponse = await fetch("model/metadata.json");
    if (metaResponse.ok) {
      metadata = await metaResponse.json();
      modelEl.textContent = metadata.model_version || metadata.model_name;
      if (metadata.default_score_threshold) {
        thresholdInput.value = String(metadata.default_score_threshold);
        thresholdValue.textContent = Number(metadata.default_score_threshold).toFixed(2);
      }
    }
    session = await ort.InferenceSession.create("model/detector.onnx", {
      executionProviders: ["wasm"],
    });
    const sampleResponse = await fetch("samples/manifest.json");
    const samples = sampleResponse.ok ? (await sampleResponse.json()).samples || [] : [];
    samples.forEach((sample, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = `Sample ${index + 1}`;
      button.addEventListener("click", () => useSample(sample, button));
      samplesEl.appendChild(button);
    });
    setStatus(samples.length ? "Model ready. Choose a sample or upload a photo." : "Model ready. Upload a photo.");
    if (samples.length) {
      samplesEl.firstChild.click();
    }
  } catch (error) {
    setStatus(`Could not start the model. ${error.message}`);
  }
}

fileInput.addEventListener("change", () => {
  if (fileInput.files && fileInput.files[0]) useFile(fileInput.files[0]);
});

thresholdInput.addEventListener("input", () => {
  thresholdValue.textContent = Number(thresholdInput.value).toFixed(2);
  if (currentImage) runImage(currentImage);
});

stage.addEventListener("dragover", (event) => event.preventDefault());
stage.addEventListener("drop", (event) => {
  event.preventDefault();
  const file = event.dataTransfer.files && event.dataTransfer.files[0];
  if (file) useFile(file);
});

main();

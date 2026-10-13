"""Standalone worker for the optional vision environment.

This file deliberately imports neither ``adanvil`` nor any project dependency: it runs in
the separate environment made by ``adanvil vision-setup`` (torch, transformers, OpenCV).

Operations (one JSON object each, answered by one JSON object):

* ``{"op": "info"}``: the model name, the embedding size and the SigLIP logit scale and bias.
* ``{"op": "embed_images", "paths": [...]}``: unit-length image embeddings.
* ``{"op": "embed_texts", "texts": [...]}``: unit-length text embeddings.
* ``{"op": "detect_faces", "paths": [...]}``: faces per picture, boxes in 0..1 of the picture.

``--serve`` keeps the models loaded and answers one request per line; ``--prefetch`` downloads
the weights (used by ``adanvil vision-setup``).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

MODEL = "google/siglip2-base-patch16-224"
MODEL_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/26cc381e4d2594bb9f47a26eb8fd96c94a13660d/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
FACE_SCORE_MIN = 0.7
FACE_LONG_SIDE = 640  # pictures are detected at this long side (YuNet sees 10 to 300 px faces)
TEXT_LENGTH = 64  # SigLIP texts are padded to this many tokens
BATCH = 8


def _models_dir() -> Path:
    value = os.environ.get("ADANVIL_VISION_DIR")
    if value:
        return Path(value)
    cache = Path(os.environ.get("ADANVIL_CACHE", Path.home() / ".cache" / "adanvil"))
    return cache / "vision-models"


def _threads() -> int:
    try:
        override = int(os.environ.get("ADANVIL_VISION_THREADS", ""))
    except ValueError:
        override = 0
    if override >= 1:
        return override
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # Windows, macOS
        return max(1, os.cpu_count() or 1)


def _prepare_environment() -> None:
    directory = _models_dir()
    directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(directory / "hf"))
    threads = str(_threads())
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _fetch_yunet() -> Path:
    """The YuNet face detector (MIT), downloaded once and checked against its SHA-256."""

    path = _models_dir() / YUNET_FILE
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == YUNET_SHA256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(YUNET_URL, headers={"User-Agent": "adanvil-vision-setup"})
    with urllib.request.urlopen(request, timeout=120) as reply:  # noqa: S310 (pinned https URL)
        data = reply.read()
    if hashlib.sha256(data).hexdigest() != YUNET_SHA256:
        raise RuntimeError("the downloaded YuNet file does not match its pinned SHA-256")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return path


def _missing(exc: BaseException) -> str:
    return (
        f"Vision worker dependencies are unavailable: {exc}. Install them with adanvil vision-setup"
    )


def _load_rgb(path: str, long_side: int | None = None):
    from PIL import Image, ImageOps

    with Image.open(path) as opened:
        picture = ImageOps.exif_transpose(opened).convert("RGB")
    if long_side is not None and max(picture.size) > long_side:
        scale = long_side / max(picture.size)
        picture = picture.resize(
            (max(1, round(picture.width * scale)), max(1, round(picture.height * scale))),
            Image.Resampling.BILINEAR,
        )
    return picture


def _model(models: dict):
    if "siglip" not in models:
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(_threads())
        model = AutoModel.from_pretrained(MODEL, revision=MODEL_REVISION).eval()
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
        models["siglip"] = (model, tokenizer)
    return models["siglip"]


def _features(output):
    """The tensor of ``get_*_features``: older transformers return it, newer ones a wrapper."""

    tensor = getattr(output, "pooler_output", None)
    return output if tensor is None else tensor


def _normalise(tensor):
    import torch

    return torch.nn.functional.normalize(tensor, dim=-1)


def _rows(tensor) -> list[list[float]]:
    return [[round(float(value), 6) for value in row] for row in tensor.numpy()]


def _info(models: dict) -> dict:
    import torch

    model, _tokenizer = _model(models)
    return {
        "ok": True,
        "model": f"{MODEL}@{MODEL_REVISION[:8]}",
        "dim": int(model.config.text_config.hidden_size),
        "logit_scale": float(torch.exp(model.logit_scale.detach())),
        "logit_bias": float(model.logit_bias.detach()),
    }


def _embed_images(op: dict, models: dict) -> dict:
    try:
        import numpy
        import torch
        from PIL import Image

        model, _tokenizer = _model(models)
        size = int(model.config.vision_config.image_size)
        paths = [str(path) for path in op["paths"]]
        vectors = []
        for start in range(0, len(paths), BATCH):
            batch = []
            for path in paths[start : start + BATCH]:
                picture = _load_rgb(path, long_side=1024).resize(
                    (size, size), Image.Resampling.BILINEAR
                )
                array = numpy.asarray(picture, dtype="float32") / 255.0
                batch.append((array - 0.5) / 0.5)
            pixels = torch.from_numpy(numpy.stack(batch)).permute(0, 3, 1, 2).contiguous()
            with torch.no_grad():
                vectors.append(_normalise(_features(model.get_image_features(pixel_values=pixels))))
        return {"ok": True, "embeddings": _rows(torch.cat(vectors)) if vectors else []}
    except ImportError as exc:
        return {"ok": False, "error": _missing(exc)}
    except Exception as exc:  # the parent receives the useful operation error
        return {"ok": False, "error": f"embed_images failed: {exc}"}


def _embed_texts(op: dict, models: dict) -> dict:
    try:
        import torch

        model, tokenizer = _model(models)
        texts = [str(text).lower() for text in op["texts"]]  # SigLIP was trained on lower case
        vectors = []
        for start in range(0, len(texts), 32):
            tokens = tokenizer(
                texts[start : start + 32],
                padding="max_length",
                max_length=TEXT_LENGTH,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                vectors.append(
                    _normalise(_features(model.get_text_features(input_ids=tokens["input_ids"])))
                )
        return {"ok": True, "embeddings": _rows(torch.cat(vectors)) if vectors else []}
    except ImportError as exc:
        return {"ok": False, "error": _missing(exc)}
    except Exception as exc:
        return {"ok": False, "error": f"embed_texts failed: {exc}"}


def _detector(models: dict):
    if "yunet" not in models:
        import cv2

        cv2.setNumThreads(_threads())
        models["yunet"] = cv2.FaceDetectorYN.create(
            str(_fetch_yunet()), "", (320, 320), FACE_SCORE_MIN, 0.3, 200
        )
    return models["yunet"]


def _detect_faces(op: dict, models: dict) -> dict:
    try:
        import cv2
        import numpy

        detector = _detector(models)
        results = []
        for path in op["paths"]:
            picture = _load_rgb(str(path), long_side=FACE_LONG_SIDE)
            width, height = picture.size
            bgr = cv2.cvtColor(numpy.asarray(picture), cv2.COLOR_RGB2BGR)
            detector.setInputSize((width, height))
            _count, found = detector.detect(bgr)
            faces = []
            for row in found if found is not None else []:
                x, y, w, h = (float(value) for value in row[:4])
                left, top = max(0.0, x / width), max(0.0, y / height)
                right, bottom = min(1.0, (x + w) / width), min(1.0, (y + h) / height)
                if right > left and bottom > top:
                    faces.append(
                        [
                            round(left, 5),
                            round(top, 5),
                            round(right, 5),
                            round(bottom, 5),
                            round(float(row[14]), 4),
                        ]
                    )
            results.append({"size": [width, height], "faces": faces})
        return {"ok": True, "pictures": results}
    except ImportError as exc:
        return {"ok": False, "error": _missing(exc)}
    except Exception as exc:
        return {"ok": False, "error": f"detect_faces failed: {exc}"}


def run_op(op: dict, models: dict) -> dict:
    """Run one operation with the models already loaded in ``models``."""

    kind = op.get("op")
    try:
        if kind == "info":
            return _info(models)
        if kind == "embed_images":
            return _embed_images(op, models)
        if kind == "embed_texts":
            return _embed_texts(op, models)
        if kind == "detect_faces":
            return _detect_faces(op, models)
    except Exception as exc:
        return {"ok": False, "error": f"Vision worker operation crashed: {exc}"}
    return {"ok": False, "error": f"Unknown vision worker operation: {kind!r}"}


def prefetch() -> int:
    """Download every model file now, so the first real call needs no network."""

    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL,
        revision=MODEL_REVISION,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
            "generation_config.json",
        ],
    )
    _fetch_yunet()
    result = _info({})  # load once: a broken install fails here, not in the first search
    if not result.get("ok"):
        print(result.get("error"), file=sys.stderr)
        return 1
    print(f"vision models ready: {result['model']}")
    return 0


def serve(source, sink) -> int:
    """Persistent mode: one JSON request per line in, one JSON reply per line out.

    ``{"id": 3, "op": {...}}`` is answered by ``{"id": 3, "result": {...}}``; the models stay
    loaded. ``{"cmd": "quit"}`` or the end of the input stops the worker. A ``{"ready": true}``
    line is written first.
    """

    def send(document: dict) -> None:
        sink.write(json.dumps(document, ensure_ascii=False) + "\n")
        sink.flush()

    models: dict = {}
    send({"ready": True})
    for line in source:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError as exc:
            send({"id": None, "result": {"ok": False, "error": f"Invalid request: {exc}"}})
            continue
        if request.get("cmd") == "quit":
            break
        send({"id": request.get("id"), "result": run_op(request.get("op") or {}, models)})
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    _prepare_environment()
    if argv == ["--prefetch"]:
        return prefetch()
    if argv == ["--serve"]:
        sink = os.fdopen(os.dup(1), "w", encoding="utf-8")
        os.dup2(2, 1)  # anything a library prints goes to the log, not into the protocol
        sys.stdout = sys.stderr
        return serve(sys.stdin, sink)
    print("usage: python worker.py --serve | --prefetch", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

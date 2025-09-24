#!/usr/bin/env python3
"""
Minimal Orchestrator: Architecture Image Evaluation (COT + Objective metrics)

Steps
- Read Q&A.csv: [A: implicit category, B: prompt, C: question]
- Generate image (DALL·E 3)
- Analyze image (GPT-4o)
- COT score (GPT-4o, 1-10)
- Compute PIQE (GPU by default; fallback to CPU)
- Optional: inject other objective metrics (NIQE/BRISQUE/IS) from JSON

Env
- API_BASE_URL
- UIUIAPI_API_KEY

Usage
  python orchestrator.py --csv Q&A.csv --out results --device cuda

Optional objective metrics JSON (list or dict by row_id):
[
  {"row_id": 1, "niqe": 7.8, "brisque": 22.5, "inception_score": 6.1},
  ...
]
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import base64
import requests
import numpy as np
from io import BytesIO
from PIL import Image
import torch
import torchvision
from torchvision.models import Inception_V3_Weights

# Import circulation analysis module
from circulation_analysis import CirculationAnalyzer
from perspective_analysis import PerspectiveAnalyzer


def get_env(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


def robust_read_csv(csv_path: str, max_rows: int = 300) -> List[List[str]]:
    """Read CSV with robust encoding fallbacks."""
    encodings = ["utf-8-sig", "utf-8", "gbk", "latin-1"]
    last_error = None
    for enc in encodings:
        try:
            with open(csv_path, "r", encoding=enc, newline="") as f:
                reader = csv.reader(f)
                rows = []
                for i, row in enumerate(reader, 1):
                    if not row:
                        continue
                    rows.append(row)
                    if i >= max_rows:
                        break
                return rows
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"Failed to read CSV {csv_path}: {last_error}")


def category_by_row(row_id: int) -> str:
    if 1 <= row_id <= 50:
        return "Object Counting"
    if 51 <= row_id <= 100:
        return "Spatial Relations"
    if 101 <= row_id <= 150:
        return "Attribute Binding"
    if 151 <= row_id <= 200:
        return "Complex Compositions"
    if 201 <= row_id <= 250:
        return "Fine-grained Actions & Dynamic Layouts"
    if 251 <= row_id <= 300:
        return "Negation Handling"
    return "Unknown"


def api_post(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int = 60, retries: int = 3, backoff: float = 1.0) -> Dict[str, Any]:
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            else:
                text = resp.text
                if attempt < retries - 1:
                    time.sleep(backoff * (attempt + 1))
                    continue
                raise RuntimeError(f"API {url} failed: {resp.status_code} {text}")
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
                continue
            raise
    raise RuntimeError("Exhausted API retries")


def api_get(url: str, headers: Dict[str, str], timeout: int = 60, retries: int = 3, backoff: float = 1.0) -> Dict[str, Any]:
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            else:
                text = resp.text
                if attempt < retries - 1:
                    time.sleep(backoff * (attempt + 1))
                    continue
                raise RuntimeError(f"API {url} failed: {resp.status_code} {text}")
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
                continue
            raise
    raise RuntimeError("Exhausted API retries")


def is_midjourney_model(model: str) -> bool:
    m = (model or "").strip().lower()
    return m in {"mj", "midjourney", "mj_imagine", "niji", "niji_journey"} or m.startswith("mj_")


def generate_image_mj(api_base: str, api_key: str, text_prompt: str, model: str, timeout: int = 600, retries: int = 5, backoff: float = 2.0) -> str:
    """Submit a Midjourney Imagine task via midjourney-proxy and poll until SUCCESS.

    Returns the final imageUrl.
    """
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # Choose botType based on model hint
    model_l = (model or "").strip().lower()
    bot_type = "NIJI_JOURNEY" if ("niji" in model_l) else "MID_JOURNEY"

    submit_url = f"{api_base.rstrip('/')}/mj/submit/imagine"
    payload = {
        "botType": bot_type,
        "prompt": text_prompt,
        "base64Array": [],
        "accountFilter": {
            "channelId": "",
            "instanceId": "",
            "modes": [],
            "remark": "",
            "remix": True,
            "remixAutoConsidered": True,
        },
        "notifyHook": "",
        "state": "",
    }

    submit_resp = api_post(submit_url, headers, payload, timeout=min(timeout, 60), retries=retries, backoff=backoff)
    # Common structure: { code, description, result: taskId }
    task_id = None
    if isinstance(submit_resp, dict):
        task_id = submit_resp.get("result")
    if not task_id:
        raise RuntimeError(f"Midjourney imagine submit returned no task id: {submit_resp}")

    # Poll task status until SUCCESS or FAILURE or timeout
    start_time = time.time()
    poll_interval = 2.0
    last_status = None
    while True:
        if time.time() - start_time > timeout:
            raise RuntimeError(f"Midjourney task {task_id} timed out after {timeout}s (last status: {last_status})")
        fetch_url = f"{api_base.rstrip('/')}/mj/task/{task_id}/fetch"
        data = api_get(fetch_url, headers, timeout=30, retries=retries, backoff=backoff)
        # Expected fields per doc
        status = (data.get("status") or "").upper()
        last_status = status
        if status == "SUCCESS":
            image_url = data.get("imageUrl") or data.get("image_url")
            if not image_url:
                raise RuntimeError(f"Midjourney task {task_id} succeeded but no imageUrl in response: {data}")
            return str(image_url)
        if status in {"FAILURE", "CANCEL"}:
            fail_reason = data.get("failReason") or data.get("description") or "Unknown"
            raise RuntimeError(f"Midjourney task {task_id} failed: {fail_reason}")
        # Keep waiting when NOT_START / SUBMITTED / MODAL / IN_PROGRESS
        time.sleep(poll_interval)


def prompt_analysis_system() -> str:
    return (
        "You are a senior architectural image analysis expert. "
        "Rely strictly on visible evidence in the image. Be structured, precise, and quantitative. "
        "Do not hallucinate or assume unseen details. Output JSON only (no backticks, no extra text). "
        "JSON schema: {\n"
        "  \"answer\": string,            # concise, direct answer to the question\n"
        "  \"evidence\": [string],        # key visual observations supporting the answer\n"
        "  \"counts\": object|null,       # optional counts (e.g., rooms, windows)\n"
        "  \"relations\": [object]|null, # optional spatial/attribute relations\n"
        "  \"uncertainty\": number        # 0.0-1.0 (1.0 = very uncertain)\n"
        "}"
    )


def prompt_evaluator_system() -> str:
    return (
        "You are an architectural evaluation expert. "
        "Perform a structured Chain-of-Thought (COT) consistency assessment between the prompt, the question, and the image-derived answer. "
        "Be objective and concise. Respond in JSON only (no backticks, no extra text). "
        "JSON schema: {\n"
        "  \"score\": integer,            # 1-10\n"
        "  \"rationale\": string,        # brief rationale for the score\n"
        "  \"checks\": {                 # brief bullet-like findings\n"
        "     \"prompt_parse\": string,\n"
        "     \"question_focus\": string,\n"
        "     \"answer_verification\": string,\n"
        "     \"consistency_check\": string\n"
        "  }\n"
        "}"
    )


def generate_image(api_base: str, api_key: str, text_prompt: str, model: str, timeout: int = 300, retries: int = 5, backoff: float = 2.0) -> str:
    # Branch for Midjourney proxy flow
    if is_midjourney_model(model):
        return generate_image_mj(api_base, api_key, text_prompt, model, timeout=max(timeout, 300), retries=retries, backoff=backoff)

    # Default: OpenAI-compatible images API
    url = f"{api_base.rstrip('/')}/v1/images/generations"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "prompt": text_prompt,
        "n": 1,
        "quality": "hd",
        "style": "vivid",
        "size": "1792x1024",  # allowed: 1024x1024 | 1024x1792 | 1792x1024
    }
    data = api_post(url, headers, payload, timeout=timeout, retries=retries, backoff=backoff)
    # Try common response shapes
    # 1) OpenAI-like: { data: [ { url } ] } or { data: [ { b64_json } ] }
    if isinstance(data.get("data"), list) and data["data"]:
        item = data["data"][0]
        # direct URL
        if isinstance(item, dict) and "url" in item and item["url"]:
            return str(item["url"])
        # nested image_url
        if isinstance(item, dict) and "image_url" in item:
            image_url = item["image_url"]
            if isinstance(image_url, str) and image_url:
                return image_url
            if isinstance(image_url, dict) and image_url.get("url"):
                return str(image_url["url"])
        # base64 payload
        if isinstance(item, dict) and "b64_json" in item and item["b64_json"]:
            mime = item.get("mime_type") or item.get("content_type") or "image/png"
            return f"data:{mime};base64,{item['b64_json']}"
        # images: [base64,...]
        if isinstance(item, dict) and isinstance(item.get("images"), list) and item["images"]:
            b64_data = item["images"][0]
            mime = item.get("mime_type") or "image/png"
            return f"data:{mime};base64,{b64_data}"
    # 2) Some providers: { images: [ base64 or url ] }
    if isinstance(data.get("images"), list) and data["images"]:
        first = data["images"][0]
        if isinstance(first, str):
            if first.startswith("http") or first.startswith("data:image/"):
                return first
            # assume base64
            return f"data:image/png;base64,{first}"
        if isinstance(first, dict):
            if first.get("url"):
                return str(first["url"])
            if first.get("b64_json"):
                return f"data:image/png;base64,{first['b64_json']}"
    # 3) Fall back to common top-level fields
    for key in ("url", "image_url", "b64_json"):
        if data.get(key):
            val = data[key]
            if key == "b64_json":
                return f"data:image/png;base64,{val}"
            if isinstance(val, dict) and val.get("url"):
                return str(val["url"])
            return str(val)
    # If we reached here, structure is unknown
    raise RuntimeError(f"Image generation returned unexpected structure: keys={list(data.keys())}")


def analyze_image(api_base: str, api_key: str, image_url: str, question: str, timeout: int = 120, retries: int = 5, backoff: float = 2.0) -> str:
    url = f"{api_base.rstrip('/')}/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": prompt_analysis_system()},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Question (answer in JSON only): {question}"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        "max_tokens": 700,
        "temperature": 0.1,
    }
    data = api_post(url, headers, payload, timeout=timeout, retries=retries, backoff=backoff)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("Image analysis returned no choices")
    return (choices[0]["message"]["content"] or "").strip()


def evaluate_cot(api_base: str, api_key: str, prompt: str, question: str, answer: str, timeout: int = 120, retries: int = 5, backoff: float = 2.0) -> Tuple[int, str]:
    url = f"{api_base.rstrip('/')}/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    user_text = (
        f"PROMPT: {prompt}\n"
        f"QUESTION: {question}\n"
        f"IMAGE_ANSWER_JSON: {answer}\n\n"
        "Assess consistency with a 4-step COT: prompt_parse → question_focus → answer_verification → consistency_check. "
        "Return JSON only matching the schema in system message."
    )
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": prompt_evaluator_system()},
            {"role": "user", "content": user_text},
        ],
        "max_tokens": 900,
        "temperature": 0.05,
    }
    data = api_post(url, headers, payload, timeout=timeout, retries=retries, backoff=backoff)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("Evaluation returned no choices")
    content = (choices[0]["message"]["content"] or "").strip()
    score = extract_score(content)
    return score, content


def extract_score(json_text: str) -> int:
    """Parse JSON from evaluator and extract integer score (1-10)."""
    try:
        data = json.loads(json_text)
        v = int(data.get("score"))
        if 1 <= v <= 10:
            return v
    except Exception:
        pass
    return 5


def load_objective_metrics(path: Optional[str]) -> Dict[int, Dict[str, float]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Accept list or dict
    if isinstance(data, list):
        return {int(item["row_id"]): item for item in data if "row_id" in item}
    elif isinstance(data, dict):
        # assume keys are row_id strings
        out = {}
        for k, v in data.items():
            try:
                out[int(k)] = v
            except Exception:
                continue
        return out
    else:
        return {}


def normalize_objective(m: Dict[str, float]) -> Optional[float]:
    if not m:
        return None
    # Extract metrics if present
    niqe = m.get("niqe")
    brisque = m.get("brisque")
    iscore = m.get("inception_score")
    piqe = m.get("piqe")
    persp = m.get("perspective_score")
    circulation = m.get("overall_circulation_score")

    def clip01(x):
        return max(0.0, min(1.0, x))

    parts = []
    weights = []

    if niqe is not None:
        parts.append(1.0 - clip01((niqe or 0.0) / 25.0))
        weights.append(0.20)
    if brisque is not None:
        parts.append(1.0 - clip01((brisque or 0.0) / 100.0))
        weights.append(0.20)
    if iscore is not None:
        parts.append(clip01(((iscore or 0.0) - 2.0) / (10.0 - 2.0)))
        weights.append(0.20)
    if piqe is not None:
        # PIQE ~ [0,100], lower is better
        parts.append(1.0 - clip01((piqe or 0.0) / 100.0))
        weights.append(0.15)
    if persp is not None:
        parts.append(clip01(persp or 0.0))
        weights.append(0.10)
    if circulation is not None:
        parts.append(clip01(circulation or 0.0))
        weights.append(0.15)  # New circulation weight

    if not parts:
        return None
    # Weighted average
    s_w = sum(weights)
    value = sum(p * w for p, w in zip(parts, weights)) / (s_w if s_w else 1.0)
    return value


def composite_score(*args, **kwargs) -> float:  # deprecated
    raise NotImplementedError("Composite score is disabled; report individual objective metrics only.")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download_image_rgb01(image_url: str) -> np.ndarray:
    """Download image and return float32 RGB array in [0,1], shape (H, W, 3)."""
    resp = requests.get(image_url, timeout=60)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def compute_piqe_score(image_rgb01: np.ndarray, device: str = "cuda") -> float:
    """Compute PIQE (no-reference) score using a standard implementation.

    Requires one of the following backends installed (in this order):
    - pyiqa (recommended): pip install pyiqa torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
    - piqe (if available on PyPI)
    """
    # Try pyiqa first
    try:
        import torch  # type: ignore
        import pyiqa  # type: ignore

        # try cuda then cpu
        dev = device
        if dev == "cuda":
            try:
                _ = pyiqa.create_metric('piqe', as_loss=False, device='cuda')
                dev = 'cuda'
            except Exception:
                dev = 'cpu'
        metric = pyiqa.create_metric('piqe', as_loss=False, device=dev)
        # input expects NCHW tensor in [0,1]
        img = torch.from_numpy(image_rgb01).permute(2, 0, 1).unsqueeze(0).to(dev)
        img = img.clamp(0, 1).float()
        with torch.no_grad():
            score = metric(img).item()
        return float(score)
    except Exception:
        pass

    # Try a module named 'piqe' (if exists)
    try:
        # hypothetical API: from piqe import piqe
        from piqe import piqe as piqe_fn  # type: ignore

        # many libs expect uint8 BGR/RGB; convert to uint8 RGB
        img_uint8 = (np.clip(image_rgb01, 0, 1) * 255.0).astype(np.uint8)
        score = float(piqe_fn(img_uint8))
        return score
    except Exception:
        pass

    raise RuntimeError(
        "PIQE backend not available. Please install 'pyiqa' (recommended) or a PIQE implementation."
    )


def save_image_with_id(image_url: str, out_images_dir: str, row_id: int) -> str:
    """Download image and save as <row_id>.<ext> inside out_images_dir. Return saved path.
    Attempts to keep original format; falls back to PNG if needed.
    """
    ensure_dir(out_images_dir)
    # Support data URI or http(s)
    if isinstance(image_url, str) and image_url.startswith("data:image/"):
        header, b64data = image_url.split(",", 1)
        # header example: data:image/png;base64
        mime = "image/png"
        try:
            mime = header.split(":", 1)[1].split(";", 1)[0] or "image/png"
        except Exception:
            pass
        raw = base64.b64decode(b64data)
        img = Image.open(BytesIO(raw))
        fmt = (img.format or 'PNG').upper()
    else:
        resp = requests.get(image_url, timeout=60)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
        fmt = (img.format or 'PNG').upper()
    ext = 'png' if fmt == 'PNG' else ('jpg' if fmt in ['JPEG', 'JPG'] else fmt.lower())
    filename = f"{row_id}.{ext}"
    save_path = os.path.join(out_images_dir, filename)
    try:
        if ext in ['jpg', 'jpeg']:
            img = img.convert('RGB')
        img.save(save_path, format=fmt)
    except Exception:
        # Fallback to PNG
        save_path = os.path.join(out_images_dir, f"{row_id}.png")
        img.convert('RGB').save(save_path, format='PNG')
    return save_path


def find_existing_image_path(out_images_dir: str, row_id: int) -> Optional[str]:
    """Return path of an existing image for row_id under out_images_dir, if any."""
    common_exts = ("png", "jpg", "jpeg", "webp", "bmp", "gif")
    for ext in common_exts:
        candidate = os.path.join(out_images_dir, f"{row_id}.{ext}")
        if os.path.isfile(candidate):
            return candidate
    # Fallback scan
    try:
        for name in os.listdir(out_images_dir):
            stem, _ext = os.path.splitext(name)
            if stem == str(row_id):
                full = os.path.join(out_images_dir, name)
                if os.path.isfile(full):
                    return full
    except Exception:
        pass
    return None


def file_to_data_url(path: str) -> str:
    """Encode a local image file as data URL suitable for image_url input."""
    ext = os.path.splitext(path)[1].lower().lstrip('.')
    mime_map = {
        'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png',
        'webp': 'image/webp', 'bmp': 'image/bmp', 'gif': 'image/gif'
    }
    mime = mime_map.get(ext, 'image/png')
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')
    return f"data:{mime};base64,{b64}"


def load_image_rgb01_from_file(path: str) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    return (np.asarray(img).astype(np.float32) / 255.0)


def compute_brisque_score(image_rgb01: np.ndarray, device: str = "cuda") -> float:
    """Compute BRISQUE using pyiqa; fallback to CPU automatically."""
    try:
        import pyiqa  # type: ignore
        dev = device
        if dev == "cuda":
            try:
                _ = pyiqa.create_metric('brisque', as_loss=False, device='cuda')
                dev = 'cuda'
            except Exception:
                dev = 'cpu'
        metric = pyiqa.create_metric('brisque', as_loss=False, device=dev)
        import torch  # local import guard
        img = torch.from_numpy(image_rgb01).permute(2, 0, 1).unsqueeze(0).to(dev)
        img = img.clamp(0, 1).float()
        with torch.no_grad():
            score = metric(img).item()
        return float(score)
    except Exception:
        pass
    raise RuntimeError("BRISQUE backend not available. Please install 'pyiqa'.")


def compute_niqe_score(image_rgb01: np.ndarray, device: str = "cuda") -> float:
    """Compute NIQE using pyiqa; fallback to CPU automatically."""
    try:
        import pyiqa  # type: ignore
        dev = device
        if dev == "cuda":
            try:
                _ = pyiqa.create_metric('niqe', as_loss=False, device='cuda')
                dev = 'cuda'
            except Exception:
                dev = 'cpu'
        metric = pyiqa.create_metric('niqe', as_loss=False, device=dev)
        import torch  # local import guard
        img = torch.from_numpy(image_rgb01).permute(2, 0, 1).unsqueeze(0).to(dev)
        img = img.clamp(0, 1).float()
        with torch.no_grad():
            score = metric(img).item()
        return float(score)
    except Exception:
        pass
    raise RuntimeError("NIQE backend not available. Please install 'pyiqa'.")


_INCEPTION_MODEL = None
_INCEPTION_TRANSFORMS = None


def compute_inception_prob(image_rgb01: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Return softmax probabilities (1000-dim) from InceptionV3 (ImageNet)."""
    global _INCEPTION_MODEL, _INCEPTION_TRANSFORMS
    try:
        dev = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        if _INCEPTION_MODEL is None:
            weights = Inception_V3_Weights.IMAGENET1K_V1
            # Use aux_logits=True to match pretrained weights; handle output accordingly
            _INCEPTION_MODEL = torchvision.models.inception_v3(weights=weights, aux_logits=True).to(dev)
            _INCEPTION_MODEL.eval()
            _INCEPTION_TRANSFORMS = weights.transforms()  # includes resize to 299, normalization

        # Convert numpy [0,1] RGB to PIL for transforms
        img_uint8 = (np.clip(image_rgb01, 0, 1) * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8, mode='RGB')
        tensor = _INCEPTION_TRANSFORMS(pil_img).unsqueeze(0).to(dev)
        with torch.no_grad():
            out = _INCEPTION_MODEL(tensor)
            # Handle different return types across torchvision versions
            if hasattr(out, 'logits'):
                logits = out.logits
            elif isinstance(out, (tuple, list)):
                logits = out[0]
            else:
                logits = out
            probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
        return probs.astype(np.float64)
    except Exception as e:
        raise RuntimeError(f"Inception prob failed: {e}")


def load_existing_results(out_root: str) -> Dict[int, Dict[str, Any]]:
    """Load existing results from final.json or any orchestrator_*.json under out_root."""
    existing: Dict[int, Dict[str, Any]] = {}
    try:
        final_path = os.path.join(out_root, 'final.json')
        if os.path.exists(final_path):
            with open(final_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for rec in data or []:
                rid = int(rec.get('row_id'))
                existing[rid] = rec
            return existing
        # fallback: collect any orchestrator_*.json
        for name in os.listdir(out_root):
            if name.endswith('.json') and name.startswith('orchestrator_'):
                with open(os.path.join(out_root, name), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for rec in data or []:
                    rid = int(rec.get('row_id'))
                    existing[rid] = rec
    except Exception:
        pass
    return existing


def copy_import_images(images_dir: str, out_images_dir: str) -> Dict[int, str]:
    """Copy numeric-named images from images_dir to out_images_dir as <row_id>.<ext>.
    Return mapping row_id->dest_path.
    """
    ensure_dir(out_images_dir)
    mapping: Dict[int, str] = {}
    if not images_dir:
        return mapping
    for name in os.listdir(images_dir):
        src = os.path.join(images_dir, name)
        if not os.path.isfile(src):
            continue
        stem, ext = os.path.splitext(name)
        try:
            rid = int(stem)
        except Exception:
            continue
        ext_norm = ext.lower().lstrip('.') or 'png'
        dest = os.path.join(out_images_dir, f"{rid}.{ext_norm}")
        if not os.path.exists(dest):
            try:
                with open(src, 'rb') as fi, open(dest, 'wb') as fo:
                    fo.write(fi.read())
            except Exception:
                continue
        mapping[rid] = dest
    return mapping


def run_perspective_only(args) -> int:
    """
    Special mode: Only recompute perspective analysis for existing results.
    This mode loads existing results, recomputes perspective metrics, and updates the output files.
    """
    print("🔄 Running perspective-only mode...")
    
    # Determine input and output directories
    if args.out == 'results':
        # Try to find existing results directories
        possible_dirs = ['DALL-E-3', 'gpt-image-1-results', 'mj_imagine-results', 'SD15-results', 'sora_image-results']
        input_dirs = [d for d in possible_dirs if os.path.isdir(d)]
        if not input_dirs:
            print("❌ No existing results directories found. Please specify --out with an existing results directory.")
            return 1
    else:
        input_dirs = [args.out]
    
    # Process each results directory
    total_updated = 0
    for input_dir in input_dirs:
        if not os.path.isdir(input_dir):
            print(f"⚠️  Directory {input_dir} not found, skipping...")
            continue
            
        print(f"\n📂 Processing {input_dir}...")
        
        # Load existing results
        existing_results = load_existing_results(input_dir)
        if not existing_results:
            print(f"  No existing results found in {input_dir}")
            continue
        
        print(f"  Found {len(existing_results)} existing results")
        
        # Initialize perspective analyzer
        perspective_analyzer = PerspectiveAnalyzer()
        updated_count = 0
        
        # Process each result
        for row_id, result in existing_results.items():
            if not (args.start_row <= row_id <= args.end_row):
                continue
            
            # Construct image path directly from row_id
            images_dir = os.path.join(input_dir, 'images')
            image_filename = f"{row_id}.png"
            image_path = os.path.join(images_dir, image_filename)
            
            if not os.path.exists(image_path):
                print(f"  ⚠️  Row {row_id}: Image file not found at {image_path}, skipping...")
                continue
            
            try:
                # Recompute perspective analysis
                perspective_result = perspective_analyzer.analyze_perspective(image_path)
                perspective_score = perspective_result.get('perspective_score', 0.5)
                
                # Update objective metrics
                objective_metrics = result.get('objective_metrics') or {}
                objective_metrics['perspective_score'] = perspective_score
                result['objective_metrics'] = objective_metrics
                
                updated_count += 1
                print(f"  ✅ Row {row_id}: Updated perspective score = {perspective_score:.3f}")
                
            except Exception as e:
                print(f"  ❌ Row {row_id}: Perspective analysis failed: {e}")
                continue
        
        # Save updated results
        if updated_count > 0:
            # Convert back to list format
            updated_results = [existing_results[k] for k in sorted(existing_results.keys())]
            
            # Save updated files
            final_json = os.path.join(input_dir, 'final.json')
            summary_csv = os.path.join(input_dir, 'summary.csv')
            
            save_json(final_json, updated_results)
            write_summary_csv(summary_csv, updated_results)
            
            print(f"  💾 Updated {updated_count} results in {input_dir}")
            print(f"     - Updated: {final_json}")
            print(f"     - Updated: {summary_csv}")
            
            total_updated += updated_count
        else:
            print(f"  ℹ️  No results were updated in {input_dir}")
    
    print(f"\n🎉 Perspective-only mode completed!")
    print(f"   Total results updated: {total_updated}")
    
    return 0


def run_circulation_only(args) -> int:
    """
    Special mode: Only recompute circulation analysis for existing results.
    This mode loads existing results, recomputes circulation metrics, and updates the output files.
    """
    print("🔄 Running circulation-only mode...")
    
    # Determine input and output directories
    if args.out == 'results':
        # Try to find existing results directories
        possible_dirs = ['DALL-E-3', 'gpt-image-1-results', 'mj_imagine-results', 'SD15-results', 'sora_image-results']
        input_dirs = [d for d in possible_dirs if os.path.isdir(d)]
        if not input_dirs:
            print("❌ No existing results directories found. Please specify --out with an existing results directory.")
            return 1
    else:
        input_dirs = [args.out]
    
    rows = robust_read_csv(args.csv, max_rows=300)
    
    # Process each results directory
    total_updated = 0
    for input_dir in input_dirs:
        if not os.path.isdir(input_dir):
            print(f"⚠️  Directory {input_dir} not found, skipping...")
            continue
            
        print(f"\n📂 Processing {input_dir}...")
        
        # Load existing results
        existing_results = load_existing_results(input_dir)
        if not existing_results:
            print(f"  No existing results found in {input_dir}")
            continue
        
        print(f"  Found {len(existing_results)} existing results")
        
        # Initialize circulation analyzer
        circulation_analyzer = CirculationAnalyzer()
        updated_count = 0
        
        # Process each result
        for row_id, result in existing_results.items():
            if not (args.start_row <= row_id <= args.end_row):
                continue
                
            # Get prompt and question from CSV or existing result
            prompt = result.get('prompt', '')
            question = result.get('question', '')
            
            # If not in result, try to get from CSV
            if not prompt or not question:
                try:
                    csv_row = rows[row_id - 1]  # CSV is 0-indexed, row_id is 1-indexed
                    prompt = prompt or (csv_row[1].strip() if len(csv_row) > 1 else "")
                    question = question or (csv_row[2].strip() if len(csv_row) > 2 else "")
                except (IndexError, KeyError):
                    print(f"  ⚠️  Row {row_id}: Could not find prompt/question, skipping...")
                    continue
            
            if not prompt or not question:
                print(f"  ⚠️  Row {row_id}: Empty prompt or question, skipping...")
                continue
            
            try:
                # Recompute circulation analysis
                circulation_result = circulation_analyzer.analyze_circulation(prompt, question)
                
                # Update objective metrics
                objective_metrics = result.get('objective_metrics') or {}
                objective_metrics.update(circulation_result)
                result['objective_metrics'] = objective_metrics
                
                updated_count += 1
                print(f"  ✅ Row {row_id}: Updated circulation score = {circulation_result.get('overall_circulation_score', 'N/A'):.3f}")
                
            except Exception as e:
                print(f"  ❌ Row {row_id}: Circulation analysis failed: {e}")
                continue
        
        # Save updated results
        if updated_count > 0:
            # Convert back to list format
            updated_results = [existing_results[k] for k in sorted(existing_results.keys())]
            
            # Save updated files
            final_json = os.path.join(input_dir, 'final.json')
            summary_csv = os.path.join(input_dir, 'summary.csv')
            
            save_json(final_json, updated_results)
            write_summary_csv(summary_csv, updated_results)
            
            print(f"  💾 Updated {updated_count} results in {input_dir}")
            print(f"     - Updated: {final_json}")
            print(f"     - Updated: {summary_csv}")
            
            total_updated += updated_count
        else:
            print(f"  ℹ️  No results were updated in {input_dir}")
    
    print(f"\n🎉 Circulation-only mode completed!")
    print(f"   Total results updated: {total_updated}")
    
    return 0


def repack_existing_results(in_root: str, out_root: Optional[str] = None, model_name: Optional[str] = None) -> str:
    """Repack existing results under in_root into standardized structure.
    - Merge JSONs to final.json
    - Write summary.csv
    - Copy images/ and cot/
    Return out_root path.
    """
    if not os.path.isdir(in_root):
        raise RuntimeError(f"Input directory not found: {in_root}")
    if not out_root:
        if model_name:
            out_root = f"{model_name}-results"
        else:
            out_root = os.path.basename(os.path.normpath(in_root)) + "-repacked"
    ensure_dir(out_root)
    # Merge results
    merged_map = load_existing_results(in_root)
    merged = [merged_map[k] for k in sorted(merged_map.keys())]
    # Copy images and cot if present
    for sub in ("images", "cot"):
        src = os.path.join(in_root, sub)
        if os.path.isdir(src):
            dst = os.path.join(out_root, sub)
            ensure_dir(dst)
            for name in os.listdir(src):
                s = os.path.join(src, name)
                d = os.path.join(dst, name)
                if os.path.isfile(s) and not os.path.exists(d):
                    with open(s, 'rb') as fi, open(d, 'wb') as fo:
                        fo.write(fi.read())
    # Write outputs
    final_json = os.path.join(out_root, 'final.json')
    summary_csv = os.path.join(out_root, 'summary.csv')
    save_json(final_json, merged)
    write_summary_csv(summary_csv, merged)
    return out_root


def run(args) -> int:
    # Special mode: only repack an existing results directory
    if getattr(args, 'repack_in', None):
        out_root = repack_existing_results(args.repack_in)
        print(f"Repacked results written to: {out_root}")
        return 0
    
    # Special mode: only recompute circulation analysis
    if getattr(args, 'circulation_only', False):
        return run_circulation_only(args)
    
    # Special mode: only recompute perspective analysis
    if getattr(args, 'perspective_only', False):
        return run_perspective_only(args)
    # Default API and KEY fallbacks (overridden by env vars if provided)
    api_base = get_env("API_BASE_URL", "https://sg.uiuiapi.com")
    api_key = get_env("UIUIAPI_API_KEY", "sk-twSzMg0U0sTIAZNbs6cI71dxcIVl9cCCNIYUA1K94xBOYOOs")

    rows = robust_read_csv(args.csv, max_rows=300)
    # Convert to (prompt, question) list with row ids
    records: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, 1):
        if not (args.start_row <= idx <= args.end_row):
            continue
        # Expect at least 3 columns, B=prompt, C=question
        prompt = row[1].strip() if len(row) > 1 else ""
        question = row[2].strip() if len(row) > 2 else ""
        if not prompt or not question:
            continue
        records.append({
            "row_id": idx,
            "category": category_by_row(idx),
            "prompt": prompt,
            "question": question,
        })

    if not records:
        print("No valid questions in the specified range.")
        return 0

    print(f"Processing {len(records)} rows: {args.start_row}-{args.end_row}")

    # Load optional objective metrics
    objective_by_row = load_objective_metrics(args.objective_json)

    # Determine output root name
    out_root = args.out
    if out_root == 'results':
        # derive default by mode
        if getattr(args, 'images_dir', None):
            base = os.path.basename(os.path.normpath(args.images_dir)) or 'import'
            out_root = f"{base}-results"
        else:
            out_root = f"{args.model}-results"

    results: List[Dict[str, Any]] = []
    ensure_dir(out_root)
    # Prepare base prefix for this run (used for JSON/CSV/LOG)
    start_end = f"{args.start_row}-{args.end_row}"
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_prefix = os.path.join(out_root, f"orchestrator_{start_end}_{run_ts}")
    log_path = f"{base_prefix}_log.txt"
    cot_dir = os.path.join(out_root, "cot")
    img_dir = os.path.join(out_root, "images")
    ensure_dir(cot_dir)
    ensure_dir(img_dir)

    # Load resume state
    existing_map = load_existing_results(out_root)
    done_ids = set(existing_map.keys())

    # Import images if provided
    import_map: Dict[int, str] = {}
    if getattr(args, 'images_dir', None):
        import_map = copy_import_images(args.images_dir, img_dir)

    inception_probs: List[np.ndarray] = []
    for i, rec in enumerate(records, 1):
        rid = rec["row_id"]
        print(f"[{i}/{len(records)}] Row {rid} - {rec['category']}")
        if rid in done_ids:
            print(f"  Skipping row {rid}: already done")
            continue
        try:
            # 1) Prepare image: reuse existing file if present; otherwise generate
            prev = existing_map.get(rid) or {}
            image_url = ""
            local_image_path: Optional[str] = None

            if import_map.get(rid):
                local_image_path = import_map[rid]
            else:
                # Check for an already saved image under output images directory
                existing_img = find_existing_image_path(img_dir, rid)
                if existing_img:
                    local_image_path = existing_img

            if local_image_path:
                image_url = prev.get("image_url") or file_to_data_url(local_image_path)
                write_log(log_path, f"Row {rid}: Reusing existing image {local_image_path}.\n")
            else:
                image_url = generate_image(
                    api_base, api_key, rec["prompt"], args.model,
                    timeout=args.image_timeout, retries=args.api_retries, backoff=args.api_backoff
                )
                # Save generated image as <row_id>.ext
                local_image_path = save_image_with_id(image_url, img_dir, rid)
            time.sleep(args.delay)

            # 2) Analyze image
            if image_url:
                answer = analyze_image(
                    api_base, api_key, image_url, rec["question"],
                    timeout=args.chat_timeout, retries=args.api_retries, backoff=args.api_backoff
                )
            else:
                answer = json.dumps({
                    "answer": "",
                    "evidence": [],
                    "counts": None,
                    "relations": None,
                    "uncertainty": 1.0
                })
            time.sleep(args.delay)

            # 3) COT evaluation (skip if COT markdown already exists)
            cot_md_path = os.path.join(cot_dir, f"cot_row_{rid}.md")
            cot_exists = os.path.exists(cot_md_path)

            if not cot_exists and image_url:
                cot, cot_text = evaluate_cot(
                    api_base, api_key, rec["prompt"], rec["question"], answer,
                    timeout=args.chat_timeout, retries=args.api_retries, backoff=args.api_backoff
                )
            elif cot_exists:
                # Reuse prior COT if present; otherwise mark as skipped
                prev_cot = prev.get("cot_score")
                cot = int(prev_cot) if isinstance(prev_cot, (int, float)) else None
                cot_text = prev.get("cot_reason") or json.dumps({
                    "score": cot if isinstance(cot, int) else 5,
                    "rationale": "COT markdown already exists; evaluation skipped.",
                    "checks": {
                        "prompt_parse": "skipped",
                        "question_focus": "skipped",
                        "answer_verification": "skipped",
                        "consistency_check": "skipped"
                    }
                })
                write_log(log_path, f"Row {rid}: Skipped COT evaluation (cot markdown exists).\n")
            else:
                cot, cot_text = 5, json.dumps({
                    "score": 5,
                    "rationale": "Image URL unavailable; COT evaluation skipped.",
                    "checks": {
                        "prompt_parse": "skipped",
                        "question_focus": "skipped",
                        "answer_verification": "skipped",
                        "consistency_check": "skipped"
                    }
                })
            time.sleep(args.delay)

            # Write row-level COT markdown for QA
            if not cot_exists:
                save_cot_markdown(cot_dir, rec, image_url, answer, cot if cot is not None else 5, cot_text)

            # Log summary
            write_log(log_path, f"Row {rid} | COT={cot} | Prompt len={len(rec['prompt'])} | Answer len={len(answer)}\n")

            # 4) Circulation analysis (new)
            circulation_metrics = {}
            try:
                circulation_analyzer = CirculationAnalyzer()
                circulation_result = circulation_analyzer.analyze_circulation(rec["prompt"], rec["question"])
                circulation_metrics = circulation_result
                write_log(log_path, f"  Circulation analysis completed for row {rid}: score={circulation_result.get('overall_circulation_score', 'N/A'):.3f}\n")
            except Exception as e:
                err = f"  Circulation analysis failed for row {rid}: {e}"
                print(err)
                write_log(log_path, err + "\n")
                # Default circulation metrics
                circulation_metrics = {
                    'circulation_efficiency': 0.5,
                    'circulation_convenience': 0.5,
                    'circulation_dynamics': 0.5,
                    'overall_circulation_score': 0.5
                }

            # 5) Perspective analysis (new)
            perspective_metrics = {}
            try:
                perspective_analyzer = PerspectiveAnalyzer()
                perspective_result = perspective_analyzer.analyze_perspective(image_path)
                perspective_metrics = {'perspective_score': perspective_result.get('perspective_score', 0.5)}
                write_log(log_path, f"  Perspective analysis completed for row {rid}: score={perspective_result.get('perspective_score', 'N/A'):.3f}\n")
            except Exception as e:
                err = f"  Perspective analysis failed for row {rid}: {e}"
                print(err)
                write_log(log_path, err + "\n")
                # Default perspective metrics
                perspective_metrics = {'perspective_score': 0.5}

            # 6) Objective metrics (optional)
            objective = objective_by_row.get(rid, {}) if objective_by_row else {}
            # Merge circulation and perspective metrics into objective metrics
            objective.update(circulation_metrics)
            objective.update(perspective_metrics)
            # Always compute/overwrite PIQE/NIQE/BRISQUE for consistency if not provided
            if "piqe" not in objective:
                try:
                    img = load_image_rgb01_from_file(local_image_path)
                    piqe_score = compute_piqe_score(img, device=args.device)
                    objective = dict(objective or {})
                    objective["piqe"] = float(piqe_score)
                except Exception as e:
                    err = f"  PIQE computation failed for row {rid}: {e}"
                    print(err)
                    write_log(log_path, err + "\n")
            if "niqe" not in objective:
                try:
                    local_img = img if 'img' in locals() else load_image_rgb01_from_file(local_image_path)
                    niqe_score = compute_niqe_score(local_img, device=args.device)
                    objective["niqe"] = float(niqe_score)
                except Exception as e:
                    err = f"  NIQE computation failed for row {rid}: {e}"
                    print(err)
                    write_log(log_path, err + "\n")
            if "brisque" not in objective:
                try:
                    local_img = img if 'img' in locals() else load_image_rgb01_from_file(local_image_path)
                    brisque_score = compute_brisque_score(local_img, device=args.device)
                    objective["brisque"] = float(brisque_score)
                except Exception as e:
                    err = f"  BRISQUE computation failed for row {rid}: {e}"
                    print(err)
                    write_log(log_path, err + "\n")

            # Inception softmax for IS (computed after loop)
            try:
                local_img = img if 'img' in locals() else load_image_rgb01_from_file(local_image_path)
                p = compute_inception_prob(local_img, device=args.device)
                inception_probs.append(p)
            except Exception as e:
                err = f"  Inception prob failed for row {rid}: {e}"
                print(err)
                write_log(log_path, err + "\n")
                inception_probs.append(None)
            result = {
                **rec,
                "image_url": image_url,
                "answer": answer,
                "cot_score": cot,
                "cot_reason": cot_text,
                "objective_metrics": objective or None,
            }
            results.append(result)
        except Exception as e:
            msg = f"Row {rid} failed: {e}"
            print(msg)
            write_log(log_path, msg + "\n")
            continue

        # Periodic save
        if i % max(1, args.save_every) == 0:
            save_json(os.path.join(args.out, f"orchestrator_partial_{i}.json"), results)

    # Final save
    # Compute IS from collected probs
    try:
        eps = 1e-16
        valid_probs = [p for p in inception_probs if isinstance(p, np.ndarray)]
        if valid_probs:
            p_y = np.mean(valid_probs, axis=0)
            for idx, p in enumerate(inception_probs):
                if isinstance(p, np.ndarray):
                    iscore = float(np.exp(np.sum(p * (np.log(p + eps) - np.log(p_y + eps)))))
                else:
                    iscore = None
                # write back
                if results[idx].get("objective_metrics") is None:
                    results[idx]["objective_metrics"] = {}
                if iscore is not None:
                    results[idx]["objective_metrics"]["inception_score"] = iscore
        else:
            write_log(log_path, "No valid inception probabilities collected; IS unavailable.\n")
    except Exception as e:
        write_log(log_path, f"IS computation failed: {e}\n")

    # Merge with existing and write final
    merged_map = dict(existing_map)
    for rec in results:
        merged_map[int(rec['row_id'])] = rec
    merged = [merged_map[k] for k in sorted(merged_map.keys())]

    json_path = f"{base_prefix}.json"
    csv_path = f"{base_prefix}.csv"
    final_json = os.path.join(out_root, 'final.json')
    summary_csv = os.path.join(out_root, 'summary.csv')

    save_json(json_path, results)
    save_csv(csv_path, results)
    # Write final merged and summary
    save_json(final_json, merged)
    write_summary_csv(summary_csv, merged)

    print(f"Saved results: {json_path}\nSaved CSV: {csv_path}\nLog: {log_path}\nCOT md dir: {cot_dir}")
    return 0


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(path: str, results: List[Dict[str, Any]]) -> None:
    # unified CSV: core + all objective metrics + circulation metrics + perspective metrics
    metric_keys = ["niqe", "brisque", "inception_score", "piqe"]
    circulation_keys = ["circulation_efficiency", "circulation_convenience", 
                       "circulation_dynamics", "overall_circulation_score"]
    perspective_keys = ["perspective_score"]
    fields = [
        "row_id", "category", "prompt", "question", "image_url", "answer",
        "cot_score",
    ] + metric_keys + circulation_keys + perspective_keys
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k) for k in [
                "row_id", "category", "prompt", "question", "image_url", "answer", "cot_score"
            ]}
            metrics = r.get("objective_metrics") or {}
            for k in metric_keys + circulation_keys:
                row[k] = metrics.get(k)
            writer.writerow(row)


def save_metrics_csv(*args, **kwargs):
    # removed: single unified CSV only
    return None


def write_summary_csv(path: str, records: List[Dict[str, Any]]) -> None:
    import math
    nums = {
        'cot_score': [], 'niqe': [], 'brisque': [], 'inception_score': [], 'piqe': [],
        'circulation_efficiency': [], 'circulation_convenience': [], 
        'circulation_dynamics': [], 'overall_circulation_score': [],
        'perspective_score': []
    }
    for r in records:
        if isinstance(r.get('cot_score'), (int, float)):
            nums['cot_score'].append(float(r['cot_score']))
        metrics = r.get('objective_metrics') or {}
        for k in ['niqe', 'brisque', 'inception_score', 'piqe',
                  'circulation_efficiency', 'circulation_convenience', 
                  'circulation_dynamics', 'overall_circulation_score',
                  'perspective_score']:
            v = metrics.get(k)
            if isinstance(v, (int, float)) and not math.isnan(float(v)):
                nums[k].append(float(v))
    means = {k: (sum(vs)/len(vs) if vs else None) for k, vs in nums.items()}
    fields = list(means.keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(means)


def write_log(log_path: str, message: str) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message)


def save_cot_markdown(cot_dir: str, rec: Dict[str, Any], image_url: str, answer: str, cot: int, cot_text: str) -> None:
    rid = rec.get("row_id")
    category = rec.get("category")
    prompt = rec.get("prompt")
    question = rec.get("question")
    md_path = os.path.join(cot_dir, f"cot_row_{rid}.md")
    content = (
        f"# COT 评审 - 行 {rid} ({category})\n\n"
        f"## 原始提示词\n\n{prompt}\n\n"
        f"## 问题\n\n{question}\n\n"
        f"## 生成图像\n\n{image_url}\n\n"
        f"## 图像分析答案\n\n{answer}\n\n"
        f"## 最终评分\n\n{cot}\n\n"
        f"## 评审COT全文\n\n{cot_text}\n"
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)


def parse_args():
    p = argparse.ArgumentParser(description="Minimal Orchestrator for Architecture Image Evaluation")
    p.add_argument("--csv", default="Q&A.csv", help="Path to Q&A.csv")
    p.add_argument("--start-row", type=int, default=1, help="Start row (1-based)")
    p.add_argument("--end-row", type=int, default=10**9, help="End row (inclusive)")
    p.add_argument("--out", default="results", help="Output directory")
    p.add_argument("--model", default="dall-e-3", help="Image generation model (e.g., dall-e-3, gpt-image-1, mj_imagine, text-embedding-3-large)")
    p.add_argument("--images-dir", default=None, help="Import images dir (numeric filenames)")
    p.add_argument("--repack-in", default=None, help="Repack existing results directory")
    p.add_argument("--objective-json", default=None, help="Optional JSON with objective metrics mapped by row_id")
    p.add_argument("--circulation-only", action="store_true", help="Only recompute circulation analysis without generating new images")
    p.add_argument("--perspective-only", action="store_true", help="Only recompute perspective analysis without generating new images")
    p.add_argument("--delay", type=float, default=1.0, help="Delay seconds between API calls")
    p.add_argument("--save-every", type=int, default=5, help="Save partial results every N rows")
    p.add_argument("--device", default="cuda", help="Device for PIQE backend (pyiqa), e.g., 'cuda' or 'cpu'")
    # API robustness
    p.add_argument("--image-timeout", type=int, default=300, help="Timeout (sec) for image generation API")
    p.add_argument("--chat-timeout", type=int, default=120, help="Timeout (sec) for chat (analysis/eval) API")
    p.add_argument("--api-retries", type=int, default=5, help="Number of retries on API failure/timeouts")
    p.add_argument("--api-backoff", type=float, default=2.0, help="Backoff seconds multiplier between retries")
    return p.parse_args()


if __name__ == "__main__":
    exit(run(parse_args()))



# Modal deployment: Qwen3.8-27B (AWQ INT4) served by vLLM as an OpenAI-compatible API.
# Deploy:  modal deploy app.py        (credentials via MODAL_TOKEN_ID / MODAL_TOKEN_SECRET env)
# Test:    modal run app.py           (ephemeral)
# Swap model later: change MODEL_ID (+ GPU_CONFIG) below and redeploy. Nothing else changes.
#
# Default config rationale:
# - cyankiwi/Qwen3.8-27B-AWQ-INT4: community AWQ build of Qwen/Qwen3.8-27B, ~900k HF downloads,
#   compressed-tensors format, vLLM-compatible. Weights ~16 GB -> fits an L4 (24 GB) with KV room.
# - L4 at ~$0.80/h -> the $30/mo Starter credit lasts ~37 container-hours; scale-to-zero when idle.
# Full-quality alternative: MODEL_ID = "Qwen/Qwen3.8-27B-FP8" (official), GPU_CONFIG = "L40S" (~15 h/mo).

import subprocess

import modal

MODEL_ID = "cyankiwi/Qwen3.8-27B-AWQ-INT4"
SERVED_NAME = "qwen3.8-27b"
GPU_CONFIG = "L4"
MAX_MODEL_LEN = 32768
SCALEDOWN_WINDOW = 300  # seconds idle before the container stops billing

app = modal.App("qwen-llm")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

hf_cache = modal.Volume.from_name("qwen-hf-cache", create_if_missing=True)
CACHE_DIR = "/root/.cache/huggingface"


@app.function(image=image, volumes={CACHE_DIR: hf_cache}, timeout=60 * 30)
def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID)
    hf_cache.commit()
    print("downloaded:", MODEL_ID)


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={CACHE_DIR: hf_cache},
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=60 * 60,
)
@modal.web_server(port=8000, startup_timeout=60 * 10)
def serve():
    cmd = [
        "vllm", "serve", MODEL_ID,
        "--served-model-name", SERVED_NAME,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.92",
        "--max-num-seqs", "8",
    ]
    subprocess.Popen(cmd)

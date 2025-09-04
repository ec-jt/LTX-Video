import os
import asyncio
import tempfile
import shutil
import threading
from pathlib import Path
from typing import Optional, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import torch
from huggingface_hub import hf_hub_download

from ltx_video.inference import (
    InferenceConfig,
    infer,
    PreloadedPipeline,
    create_ltx_video_pipeline,
    create_latent_upsampler,
    load_pipeline_config,
    get_device,
)

app = FastAPI(title="LTX-Video Frame Streaming API (Warm & Robust)", version="1.4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# One generation at a time (raise cautiously if you have headroom)
app.state.gen_sema = asyncio.Semaphore(1)

BOUNDARY = "frame"  # multipart/x-mixed-replace boundary

# --------------------------- Pipeline cache -----------------------------------

class PipelineManager:
    def __init__(self, max_cache: int = 2):
        self.max_cache = max_cache
        self._lock = threading.RLock()
        self._cache: dict[str, PreloadedPipeline] = {}
        self._lru: list[str] = []

    def _resolve_model_file(self, filename: str) -> str:
        if os.path.isabs(filename) and os.path.isfile(filename):
            return filename
        if os.path.isfile(filename):
            return filename
        return hf_hub_download(
            repo_id="Lightricks/LTX-Video", filename=filename, repo_type="model"
        )

    def get(self, pipeline_config_path: str) -> PreloadedPipeline:
        key = os.path.abspath(pipeline_config_path)

        with self._lock:
            if key in self._cache:
                self._lru.remove(key)
                self._lru.append(key)
                return self._cache[key]

        cfg_dict = load_pipeline_config(key)
        device = get_device()

        ckpt = self._resolve_model_file(cfg_dict["checkpoint_path"])
        upsampler_path = cfg_dict.get("spatial_upscaler_model_path")
        if upsampler_path:
            upsampler_path = self._resolve_model_file(upsampler_path)

        preload_enhancers = cfg_dict.get("prompt_enhancement_words_threshold", 0) > 0

        pipeline = create_ltx_video_pipeline(
            ckpt_path=ckpt,
            precision=cfg_dict["precision"],
            text_encoder_model_name_or_path=cfg_dict["text_encoder_model_name_or_path"],
            sampler=cfg_dict.get("sampler", None),
            device=device,
            enhance_prompt=preload_enhancers,
            prompt_enhancer_image_caption_model_name_or_path=cfg_dict.get(
                "prompt_enhancer_image_caption_model_name_or_path"
            ),
            prompt_enhancer_llm_model_name_or_path=cfg_dict.get(
                "prompt_enhancer_llm_model_name_or_path"
            ),
        )

        if cfg_dict.get("pipeline_type", None) == "multi-scale":
            if not upsampler_path:
                raise ValueError("spatial_upscaler_model_path missing for multi-scale.")
            latent_upsampler = create_latent_upsampler(upsampler_path, device)
            from ltx_video.pipelines.pipeline_ltx_video import LTXMultiScalePipeline
            pipeline = LTXMultiScalePipeline(pipeline, latent_upsampler=latent_upsampler)

        bundle = PreloadedPipeline(pipeline=pipeline, config=cfg_dict)

        with self._lock:
            self._cache[key] = bundle
            if key in self._lru:
                self._lru.remove(key)
            self._lru.append(key)

            if len(self._lru) > self.max_cache:
                evict_key = self._lru.pop(0)
                ev = self._cache.pop(evict_key, None)
                if ev is not None:
                    p = ev.pipeline
                    try:
                        vp = getattr(p, "video_pipeline", p)
                        for name in [
                            "transformer",
                            "vae",
                            "text_encoder",
                            "prompt_enhancer_image_caption_model",
                            "prompt_enhancer_llm_model",
                        ]:
                            m = getattr(vp, name, None)
                            if m is not None:
                                try:
                                    m.to("cpu")
                                except Exception:
                                    pass
                        del ev
                    finally:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

        return bundle

PIPELINES = PipelineManager(max_cache=2)

# --------------------------- upload utils -------------------------------------

CONTENT_TYPE_EXT = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/x-matroska": ".mkv",
    "video/x-msvideo": ".avi",
    "video/avi": ".avi",
}

def sniff_ext_from_magic(buf: bytes) -> Optional[str]:
    if buf.startswith(b"\xff\xd8\xff"): return ".jpg"
    if buf.startswith(b"\x89PNG\r\n\x1a\n"): return ".png"
    if len(buf) >= 12 and buf[0:4] == b"RIFF" and buf[8:12] == b"WEBP": return ".webp"
    if b"ftyp" in buf[:256]: return ".mp4"
    if buf.startswith(b"\x1A\x45\xDF\xA3"): return ".mkv"
    if len(buf) >= 12 and buf[0:4] == b"RIFF" and buf[8:12] == b"AVI ": return ".avi"
    return None

async def save_upload_to_temp(media: UploadFile) -> Optional[str]:
    head = await media.read(64 * 1024)
    if not head:
        return None
    suffix = os.path.splitext(media.filename or "")[1].lower()
    if not suffix:
        if media.content_type and media.content_type in CONTENT_TYPE_EXT:
            suffix = CONTENT_TYPE_EXT[media.content_type]
        else:
            sniffed = sniff_ext_from_magic(head)
            if sniffed:
                suffix = sniffed
    if not suffix:
        raise HTTPException(status_code=400, detail="Uploaded media not recognized.")
    tmp_dir = tempfile.mkdtemp(prefix="ltxv_upload_")
    path = Path(tmp_dir) / f"input{suffix}"
    with open(path, "wb") as out:
        out.write(head)
        while True:
            chunk = await media.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    try:
        await media.seek(0)
    except Exception:
        pass
    return str(path)

def mjpeg_part(jpeg: bytes, idx: int, total: int) -> bytes:
    headers = (
        f"--{BOUNDARY}\r\n"
        f"Content-Type: image/jpeg\r\n"
        f"X-Frame-Index: {idx}\r\n"
        f"X-Frame-Total: {total}\r\n"
        f"\r\n"
    ).encode("ascii")
    return headers + jpeg + b"\r\n"

async def stream_frames(queue: "asyncio.Queue[Optional[Tuple[int,int,bytes]]]",
                        fps: int, realtime: bool, client_gone: asyncio.Event):
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            idx, total, jpeg = item
            yield mjpeg_part(jpeg, idx, total)
            if realtime and fps > 0:
                await asyncio.sleep(1.0 / float(fps))
        yield f"--{BOUNDARY}--\r\n".encode("ascii")
    except asyncio.CancelledError:
        # tell producer to stop caring
        client_gone.set()
        return

# --------------------------- endpoints ----------------------------------------

@app.get("/v1/health")
def health():
    return {"ok": True, "device": get_device()}

@app.post("/v1/frames", response_class=StreamingResponse)
async def generate_frames(
    prompt: str = Form(...),
    media: Optional[UploadFile] = File(default=None, description="optional image/video"),
    height: int = Form(704),
    width: int = Form(1216),
    num_frames: int = Form(121),
    frame_rate: int = Form(30),
    seed: int = Form(171198),
    negative_prompt: str = Form("worst quality, inconsistent motion, blurry, jittery, distorted"),
    pipeline_config: str = Form("configs/ltxv-2b-0.9.8-distilled.yaml"),
    offload_to_cpu: bool = Form(False),
    image_cond_noise_scale: float = Form(0.15),
    realtime: bool = Form(True),
    jpeg_quality: int = Form(85),
    buffer_all: bool = Form(True),   # buffer so we don't drop frames
):
    """
    Streams FINAL frames as MJPEG using a warm pipeline and robust queueing.
    """

    # optional media
    tmp_dir_root = None
    input_media_path = None
    if media is not None:
        input_media_path = await save_upload_to_temp(media)
        if input_media_path:
            tmp_dir_root = str(Path(input_media_path).parent)

    # Queue: unbounded when buffer_all=True (prevents deadlock on client disconnect)
    queue_max = 0 if buffer_all else 1
    q: "asyncio.Queue[Optional[Tuple[int,int,bytes]]]" = asyncio.Queue(maxsize=queue_max)
    loop = asyncio.get_running_loop()
    client_gone = asyncio.Event()

    def _safe_put(item):
        # Runs in event loop thread; never blocks
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            # should not happen when maxsize=0, but safe if user disables buffer_all
            pass

    async def producer():
        # keep GPU usage serialized
        async with app.state.gen_sema:
            try:
                bundle = PIPELINES.get(pipeline_config)

                cfg = InferenceConfig(
                    prompt=prompt,
                    pipeline_config=pipeline_config,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=frame_rate,
                    offload_to_cpu=offload_to_cpu,
                    negative_prompt=negative_prompt,
                    input_media_path=input_media_path,
                    image_cond_noise_scale=image_cond_noise_scale,
                )

                # non-blocking, safe enqueue from worker thread
                def on_final_frame(frame_idx: int, total_frames: int, jpeg_bytes: bytes):
                    if client_gone.is_set():
                        return  # stop enqueuing if client left
                    loop.call_soon_threadsafe(_safe_put, (frame_idx, total_frames, jpeg_bytes))

                # run inference without blocking the event loop
                await loop.run_in_executor(
                    None,
                    lambda: infer(
                        cfg,
                        on_final_frame=on_final_frame,
                        final_frame_jpeg_quality=jpeg_quality,
                        preloaded=bundle,
                    ),
                )

            except Exception:
                # On error, signal end (best effort)
                loop.call_soon_threadsafe(_safe_put, None)
                raise
            finally:
                # Release semaphore (by leaving context) *then* signal end best-effort.
                loop.call_soon_threadsafe(_safe_put, None)
                if tmp_dir_root and os.path.isdir(tmp_dir_root):
                    shutil.rmtree(tmp_dir_root, ignore_errors=True)

    asyncio.create_task(producer())

    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Connection": "keep-alive",
    }
    media_type = f"multipart/x-mixed-replace; boundary={BOUNDARY}"
    return StreamingResponse(
        stream_frames(q, frame_rate, realtime, client_gone),
        media_type=media_type,
        headers=headers,
    )

import os
import io
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# IMPORTANT: this must point to your patched infer that emits on_final_frame(...)
from inference import InferenceConfig, infer

app = FastAPI(title="LTX-Video Frame Streaming API", version="1.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# One generation at a time unless you add proper queuing
app.state.gen_sema = asyncio.Semaphore(1)

BOUNDARY = "frame"  # multipart/x-mixed-replace boundary


# --------------------------- utils -------------------------------------------

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
    # images
    if buf.startswith(b"\xff\xd8\xff"):  # JPEG
        return ".jpg"
    if buf.startswith(b"\x89PNG\r\n\x1a\n"):  # PNG
        return ".png"
    if len(buf) >= 12 and buf[0:4] == b"RIFF" and buf[8:12] == b"WEBP":
        return ".webp"
    # videos
    if b"ftyp" in buf[:256]:  # MP4/MOV brand box near start
        # crude heuristic; default to mp4
        return ".mp4"
    if buf.startswith(b"\x1A\x45\xDF\xA3"):  # EBML (MKV/WebM)
        return ".mkv"
    if len(buf) >= 12 and buf[0:4] == b"RIFF" and buf[8:12] == b"AVI ":
        return ".avi"
    return None

async def save_upload_to_temp(media: UploadFile) -> Optional[str]:
    """
    Save uploaded file with a sensible extension.
    Returns the file path, or None if the upload is empty.
    Raises 400 if type is unrecognized.
    """
    # Read a small head chunk for sniffing, but we must still stream the rest to disk
    head = await media.read(64 * 1024)
    if not head:
        # empty file input => treat as no media
        return None

    # Derive extension from filename, content-type, or magic
    suffix = os.path.splitext(media.filename or "")[1].lower()
    if not suffix:
        if media.content_type and media.content_type in CONTENT_TYPE_EXT:
            suffix = CONTENT_TYPE_EXT[media.content_type]
        else:
            sniffed = sniff_ext_from_magic(head)
            if sniffed:
                suffix = sniffed

    if not suffix:
        raise HTTPException(status_code=400, detail="Uploaded media not recognized as image or video.")

    tmp_dir = tempfile.mkdtemp(prefix="ltxv_upload_")
    path = Path(tmp_dir) / f"input{suffix}"

    # Write head then the rest of the body
    with open(path, "wb") as out:
        out.write(head)
        # stream the remaining bytes
        while True:
            chunk = await media.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)

    # Rewind underlying file to leave UploadFile in a sane state (not strictly required)
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
                        fps: int, realtime: bool):
    """Async generator yielding MJPEG parts. Ends on sentinel None."""
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
        return


# --------------------------- endpoints ---------------------------------------

@app.get("/v1/health")
def health():
    return {"ok": True}

@app.post("/v1/frames", response_class=StreamingResponse)
async def generate_frames(
    # core inputs
    prompt: str = Form(...),
    media: Optional[UploadFile] = File(default=None, description="optional image/video"),
    # video params
    height: int = Form(704),
    width: int = Form(1216),
    num_frames: int = Form(121),
    frame_rate: int = Form(30),
    seed: int = Form(171198),
    negative_prompt: str = Form("worst quality, inconsistent motion, blurry, jittery, distorted"),
    pipeline_config: str = Form("configs/ltxv-13b-0.9.8-distilled.yaml"),
    offload_to_cpu: bool = Form(False),
    image_cond_noise_scale: float = Form(0.15),
    # streaming controls
    realtime: bool = Form(True),     # pace output at 1/fps
    jpeg_quality: int = Form(85),    # quality for streamed frames
    buffer_all: bool = Form(True),   # if True, buffer so no frames are skipped
):
    """
    Streams FINAL frames as MJPEG at the requested FPS.
    - If buffer_all=True, we buffer up to `num_frames` frames so none are skipped.
    - The MP4 is written to disk inside `infer`.
    """

    # Save uploaded media if present
    tmp_dir_root = None
    input_media_path = None
    if media is not None:
        input_media_path = await save_upload_to_temp(media)
        if input_media_path:
            tmp_dir_root = str(Path(input_media_path).parent)

    # Buffer size: allow the full clip to queue (no drops)
    queue_size = max(1, int(num_frames)) if buffer_all else 1
    q: "asyncio.Queue[Optional[Tuple[int,int,bytes]]]" = asyncio.Queue(maxsize=queue_size)
    loop = asyncio.get_running_loop()

    async def producer():
        async with app.state.gen_sema:
            try:
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
                    input_media_path=input_media_path,  # None if no/empty upload
                    image_cond_noise_scale=image_cond_noise_scale,
                )

                # Block until frames are enqueued (backpressure when queue is full).
                def on_final_frame(frame_idx: int, total_frames: int, jpeg_bytes: bytes):
                    fut = asyncio.run_coroutine_threadsafe(q.put((frame_idx, total_frames, jpeg_bytes)), loop)
                    try:
                        fut.result()  # block this worker thread until queued
                    except Exception:
                        # If client disconnected and generator is gone, we might get CancelledError
                        pass

                await loop.run_in_executor(
                    None,
                    lambda: infer(cfg, on_final_frame=on_final_frame, final_frame_jpeg_quality=jpeg_quality),
                )

            finally:
                # Signal end-of-stream (also blocking to ensure delivery order)
                try:
                    asyncio.run_coroutine_threadsafe(q.put(None), loop).result()
                except Exception:
                    pass
                # Clean up temp files
                if tmp_dir_root and os.path.isdir(tmp_dir_root):
                    shutil.rmtree(tmp_dir_root, ignore_errors=True)

    asyncio.create_task(producer())

    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Connection": "keep-alive",
    }
    media_type = f"multipart/x-mixed-replace; boundary={BOUNDARY}"
    return StreamingResponse(stream_frames(q, frame_rate, realtime), media_type=media_type, headers=headers)

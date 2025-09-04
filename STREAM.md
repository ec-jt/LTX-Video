#API server
pip install fastapi uvicorn imageio imageio-ffmpeg pillow python-multipart
LTXV_PRELOAD=configs/ltxv-2b-0.9.8-distilled.yaml UVICORN_WORKERS=1 uvicorn server:app --host 0.0.0.0 --port 8000

#API request
curl -N -X POST "http://localhost:8000/v1/frames" \
  -H "Content-Type: multipart/form-data" \
  -F "prompt=A cinematic rain-soaked street at night, neon reflections" \
  -F "height=320" -F "width=480" -F "num_frames=121" -F "frame_rate=30" \
  -F "pipeline_config=configs/ltxv-2b-0.9.8-distilled.yaml" \
  --output - | hexdump -C | head

#Browser streaming
python -m http.server 8080 &
http://localhost:8080

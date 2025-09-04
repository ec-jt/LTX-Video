#Setup 
git clone https://github.com/ec-jt/LTX-Video.git
cd LTX-Video/
python3 -m venv venv
source venv/bin/activate
pip install packaging wheel ninja setuptools
pip install -e .\[inference\]
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
#git clone https://github.com/KONAKONA666/q8_kernels
#pip install --no-build-isolation git+https://github.com/Lightricks/LTX-Video-Q8-Kernels.git

#API server
pip install fastapi uvicorn imageio imageio-ffmpeg pillow python-multipart
LTXV_PRELOAD=configs/ltxv-2b-0.9.8-distilled.yaml UVICORN_WORKERS=1 uvicorn server:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/v1/health

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

#Edit index.html hostname if remote to view stream

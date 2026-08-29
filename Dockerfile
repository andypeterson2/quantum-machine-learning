FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu torchvision==0.17.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Quantum extra (full-quantum deploy): pennylane lights up the iris QVC; qiskit +
# qiskit-aer the MNIST quantum models (the code gates each model on its lib being
# importable). PINNED, and numpy re-pinned to 1.26.4: pennylane>=0.45 requires
# numpy>=2, which breaks torch 2.2.2's numpy bridge ("Numpy is not available").
# Keep this set in lockstep with the torch/numpy pins.
RUN pip install --no-cache-dir \
    "numpy==1.26.4" "pennylane==0.44.1" "qiskit==2.4.2" "qiskit-aer==0.17.2"

COPY classifiers/ classifiers/
COPY .cert[s]/ .certs/

ENV DEV_CERT_DIR=/app/.certs
# Never run the Werkzeug debugger/reloader in the image (interactive-debugger RCE).
ENV CLASSIFIERS_DEBUG=0

# Production WSGI server (gunicorn). ONE worker — the model registry is per-process
# in-memory state; --threads gives concurrency for SSE/parallel requests within it.
# --timeout 0 because SSE/training stream unbounded. Bind Railway's injected $PORT
# (default 8080). `sh -c exec` so gunicorn becomes PID 1 and gets SIGTERM on restart.
EXPOSE 8080
CMD ["sh", "-c", "exec gunicorn --worker-class gthread --workers 1 --threads 8 --timeout 0 --bind 0.0.0.0:${PORT:-8080} classifiers.wsgi:app"]

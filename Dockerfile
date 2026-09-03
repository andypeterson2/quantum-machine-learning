FROM python:3.12-slim

WORKDIR /app

# CPU-only torch first — its own layer, the biggest download. The linux image
# is free of the Intel-macOS ceiling that pins local dev to torch 2.2 (PyTorch
# shipped its last x86_64-macOS wheels at 2.2.2), so production runs the
# current stack: latest torch, numpy 2, current pennylane/qiskit.
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY classifiers/ classifiers/

# The package plus the quantum extra, resolved from pyproject's ranges — the
# environment markers keep the numpy<2 / pennylane<0.45 pins Intel-Mac-only,
# so this resolves the modern stack here. Installing the package also lets
# /health report the real version via importlib.metadata.
RUN pip install --no-cache-dir .[quantum]

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

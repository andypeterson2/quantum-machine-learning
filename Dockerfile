FROM python:3.12-slim

WORKDIR /app

# CPU-only torch first — its own layer, the biggest download. The linux image is
# free of the Intel-macOS ceiling that pins local dev to torch 2.2 (PyTorch
# shipped its last x86_64-macOS wheels at 2.2.2), so production runs the current
# stack: torch 2.14, numpy 2, current pennylane/qiskit. Both files are locks, so
# two builds of the same commit install the same versions.
COPY requirements/linux/ requirements/linux/
RUN pip install --no-cache-dir -r requirements/linux/torch.txt

COPY pyproject.toml README.md ./
COPY classifiers/ classifiers/

# The rest of the stack from the lock, then the package itself with --no-deps so
# the lock stays authoritative over pyproject's ranges. Installing the package is
# what lets /health report the real version via importlib.metadata.
RUN pip install --no-cache-dir -r requirements/linux/requirements.txt
RUN pip install --no-cache-dir --no-deps .

# Never run the Werkzeug debugger/reloader in the image (interactive-debugger RCE).
ENV CLASSIFIERS_DEBUG=0

# Production WSGI server (gunicorn). ONE worker — the model registry is per-process
# in-memory state; --threads gives concurrency for SSE/parallel requests within it.
# --timeout 120: SSE streams are lifetime-bounded server-side now (see
# routes/sse.py), and gthread's timeout watches the worker process, not
# individual streaming threads — a hung worker gets restarted instead of
# wedging forever behind --timeout 0. Bind Railway's injected $PORT
# (default 8080). `sh -c exec` so gunicorn becomes PID 1 and gets SIGTERM on restart.
EXPOSE 8080
CMD ["sh", "-c", "exec gunicorn --worker-class gthread --workers 1 --threads 8 --timeout 120 --bind 0.0.0.0:${PORT:-8080} classifiers.wsgi:app"]

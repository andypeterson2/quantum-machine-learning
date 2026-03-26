FROM python:3.12-slim

WORKDIR /app

COPY packages/quantum-protein-kernel/requirements.txt .
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu torchvision==0.17.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY packages/quantum-protein-kernel/classifiers/ classifiers/
COPY packages/ui-kit/ classifiers/static/ui-kit/

CMD ["python", "-m", "classifiers"]

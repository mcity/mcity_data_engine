FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
RUN apt-get update && \
    apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    libcurl4

WORKDIR /launch

COPY --link requirements.txt ./
RUN pip install -r requirements.txt

COPY --link main.py configs/ ./
ENTRYPOINT ["python", "main.py"]
# Pinned by digest, not just by tag: a retagged base would silently change the CUDA
# runtime and the torch build underneath a task whose thresholds were calibrated
# against this exact one.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

WORKDIR /app

# Only the public half of the data. The generator that could re-derive the held-out
# labels from the seed is deliberately not in this image, and neither are the labels.
COPY task_inputs/ /app/task_inputs/

RUN mkdir -p /app/results

# GeoCrop Engine container (CPU), CPython 3.12 with the pinned dependency set constraints/py312.txt.
# ESA SNAP is not included; run Sentinel-1 preprocessing on the host or in a separate SNAP image and mount the
# scene folders. Runs as an unprivileged user. Select optional extras with --build-arg EXTRAS=... (e.g.
# all,stac,web,deep,ftw for the PyTorch models); every extra resolves against constraints/py312.txt. On x86_64,
# add --build-arg PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu to avoid the CUDA build of torch.
FROM python:3.12-slim-bookworm
LABEL org.opencontainers.image.title="GeoCrop Engine" \
      org.opencontainers.image.source="https://github.com/ZeljkoDzafic/geocrop-engine" \
      org.opencontainers.image.licenses="Apache-2.0"
ARG EXTRAS=all,stac,web
ARG PIP_EXTRA_INDEX_URL=
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1 \
    GEOCROP_CACHE=/home/geocrop/.cache/geocrop
# libgomp1: OpenMP runtime of LightGBM/scikit-learn wheels; libexpat1: needed by the GDAL bundled in the rasterio
# wheel (not in the slim base). GDAL/PROJ/GEOS themselves come inside the rasterio, pyogrio,
# pyproj and shapely wheels. `_entrypoint.sh` keeps the entry point of the former micromamba image, so
# `docker run --entrypoint /usr/local/bin/_entrypoint.sh <image> python ...` keeps working.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libexpat1 && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 geocrop \
    && printf '#!/bin/sh\nexec "$@"\n' > /usr/local/bin/_entrypoint.sh && chmod 755 /usr/local/bin/_entrypoint.sh
COPY constraints/py312.txt /opt/geocrop/constraints/py312.txt
COPY pyproject.toml README.md LICENSE NOTICE /opt/geocrop/
COPY src /opt/geocrop/src
RUN python -m pip install -c /opt/geocrop/constraints/py312.txt "/opt/geocrop[${EXTRAS}]" \
    && geocrop --version && mkdir -p /work && chown geocrop:geocrop /work
USER geocrop
WORKDIR /work
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "geocrop"]
CMD ["--help"]

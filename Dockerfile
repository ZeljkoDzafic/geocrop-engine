# GeoCrop Engine container (CPU). ESA SNAP is not included; run Sentinel-1 preprocessing on the host
# or in a separate SNAP image and mount the scene folders. Runs as the unprivileged micromamba user.
FROM mambaorg/micromamba:1.5-jammy
LABEL org.opencontainers.image.title="GeoCrop Engine" \
      org.opencontainers.image.source="https://github.com/ZeljkoDzafic/geocrop-engine" \
      org.opencontainers.image.licenses="Apache-2.0"
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
# environment.yml ends with a pip section installing the checkout ("-e ."), which is not present yet in
# this layer; drop it here and install the engine below once the sources are copied.
RUN sed -i '/^  - pip:/,$d' /tmp/environment.yml && \
    micromamba install -y -n base -f /tmp/environment.yml && micromamba clean -a -y
COPY --chown=$MAMBA_USER:$MAMBA_USER . /opt/geocrop
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN pip install --no-cache-dir -e "/opt/geocrop[all,stac,web]" && geocrop --version
WORKDIR /work
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "geocrop"]
CMD ["--help"]

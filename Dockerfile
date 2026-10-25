# GeoCrop Engine container (CPU). ESA SNAP is not included; run Sentinel-1 preprocessing on the host
# or in a separate SNAP image and mount the scene folders.
FROM mambaorg/micromamba:1.5-jammy
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && micromamba clean -a -y
COPY --chown=$MAMBA_USER:$MAMBA_USER . /opt/geocrop
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN pip install --no-cache-dir -e "/opt/geocrop[all,stac,web]" && geocrop --version
WORKDIR /work
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "geocrop"]
CMD ["--help"]

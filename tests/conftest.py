import os
import tempfile

import pytest

from geocrop.demo import generate


@pytest.fixture(scope="session")
def demo(tmp_path_factory):
    """Synthetic composites, reference polygons, parcels and field points (shared by tests)."""
    return generate(tmp_path_factory.mktemp("demo"))


@pytest.fixture(scope="session")
def stack(demo, tmp_path_factory):
    from geocrop.features import build_stack
    return build_stack(demo["composites"], tmp_path_factory.mktemp("features") / "stack.tif", block_size=64)


@pytest.fixture(scope="session")
def samples(stack, demo):
    from geocrop.samples import extract
    return extract(stack, demo["training"], max_per_polygon=40)


# Benchmark runs are recorded in a registry (geocrop.registry); keep test runs out of the working directory.
os.environ.setdefault("GEOCROP_REGISTRY", tempfile.mkdtemp(prefix="geocrop-registry-"))

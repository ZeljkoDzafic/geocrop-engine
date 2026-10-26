# GeoCrop plugin example

A complete, installable plugin that adds three things to GeoCrop Engine without changing its code:

| Entry-point group      | Name               | Object                                   |
|------------------------|--------------------|------------------------------------------|
| `geocrop.estimators`   | `nearest_centroid` | factory `(params, seed) -> estimator`    |
| `geocrop.indices`      | `NDTI`             | `geocrop.indices.Index`                  |
| `geocrop.segmenters`   | `grid`             | class with `name` and `segment(...)`     |

```bash
pip install -e examples/plugin_example
python -m geocrop.plugins          # lists built-ins and the three plugins
```

Then use them like built-ins:

```yaml
features:
  s2_indices: [NDVI, NDTI]
model:
  algorithm: nearest_centroid
```

```python
from geocrop import benchmark
benchmark.segmentation(raster, reference, "out", methods={"grid_16": {"segmenter": "grid", "cell_px": 16}})
```

Contracts are documented in `geocrop.plugins` and docs/adr/0002-plugin-entry-points.md. A plugin that fails to
import produces a `geocrop.plugins.PluginWarning` and is skipped.

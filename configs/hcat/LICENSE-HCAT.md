# Licence and attribution of the HCAT mapping tables

The files in this folder (`rs_bih_hcat_harmonisation.csv`, `hcat_mapping_rs_bih.csv`) are **data, not code**.
They contain codes and names of the **Hierarchical Crop and Agriculture Taxonomy (HCAT, version 3)** published by
the EuroCrops project and are therefore an adaptation of that taxonomy.

* Source: Schneider, M., Chan, A., Körner, M. (2024). *EuroCrops* [dataset], file `HCAT3.csv`. Zenodo.
  <https://doi.org/10.5281/zenodo.14094196> (also <https://github.com/maja601/EuroCrops>), retrieved 2026-10-26.
* Reference: Schneider, M., Schelte, T., Schmitz, F., Körner, M. (2023). EuroCrops: The Largest Harmonized Open
  Crop Dataset Across the European Union. *Scientific Data* 10, 612. <https://doi.org/10.1038/s41597-023-02517-0>
* Licence of the source: Creative Commons Attribution-ShareAlike 4.0 International (CC-BY-SA-4.0).

**These two tables are licensed under CC-BY-SA-4.0** (<https://creativecommons.org/licenses/by-sa/4.0/>),
not under the Apache-2.0 licence of the GeoCrop source code. If you redistribute them, or tables or labels
derived from them, keep this attribution and share them under the same licence. Changes made here: selection of
HCAT codes and assignment to the GeoCrop / Republic of Srpska class names (`maize`, `soybean`, `sunflower`,
`winter_crops`, `grassland`, `vegetables`, `tobacco`, `forest`, `water`, `built_up`).

No EuroCrops parcel data are included in this repository. `geocrop benchmark-public` downloads them at run time
from the official Zenodo record into a local cache; parcel labels derived from them must not be redistributed
without the ShareAlike terms.

## Mapping decisions (review per study)

* `winter_crops` = autumn-sown cereals only. Winter oilseed rape (`3301060401`) is **not** included; add a row if
  the study's reference data count it as a winter crop.
* `grassland` includes temporary grass on arable land (`3301090100`); clover and lucerne are not mapped.
* `vegetables` lists explicit vegetable groups; potatoes, sugar beet and oil pumpkins are not mapped.
* `water` and `built_up` have no HCAT equivalent (HCAT is a crop taxonomy). The export table assigns them
  `3399000000 not_known_and_other` only so that a STAC export of a full legend is possible.
* In `hcat_mapping_rs_bih.csv` (one representative code per class, used by `geocrop export --hcat`)
  `winter_crops` is published as the parent `3301010000 cereal`, which is broader than the class.

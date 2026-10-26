# VM runbook (Ubuntu 26.04 or 24.04, CPU only)

Recommended for the northern Republic of Srpska (~10,000 km², one season): 16 vCPU, 64 GB RAM, 2 TB local
NVMe (`/scratch`), 2–3 TB archive (`/data`), outbound HTTPS. A pilot on one or two Sentinel-2 tiles works with
8 vCPU, 32 GB RAM and 500 GB.

## 1. Setup
```bash
sudo useradd -r -m geocrop                  # the run user (systemd unit below runs as it)
sudo mkdir -p /srv/geocrop && sudo chown geocrop:geocrop /srv/geocrop
sudo -u geocrop -H git clone https://github.com/ZeljkoDzafic/geocrop-engine.git /srv/geocrop/geocrop-engine
cd /srv/geocrop/geocrop-engine
sudo bash deploy/setup_ubuntu.sh            # packages, /data/geocrop and /scratch/geocrop (root-owned, mode 750)
sudo chown -R geocrop:geocrop /data/geocrop /scratch/geocrop
sudo -u geocrop -H bash deploy/setup_ubuntu.sh --user   # Miniforge, conda env "geocrop" (Python 3.11), engine, tests
sudo -u geocrop -H env SNAP_VERSION=11.0 bash deploy/setup_ubuntu.sh --snap   # optional, Sentinel-1 (verify installer name)
```
Run the `--user` and `--snap` steps as the account that will run the pipeline (here `geocrop`): Miniforge and
SNAP are installed into that account's home, and the systemd unit sources `~/miniforge3` of user `geocrop`.
The conda environment brings its own Python, so the distribution's default Python (newer than 3.12 on Ubuntu
26.04) does not matter.
Credentials (never in git): `/etc/geocrop/credentials.env` with `CDSE_USERNAME=` and `CDSE_PASSWORD=`, mode 600.

## 2. Pilot (one or two tiles)
Copy the study configuration and point its grid and paths at the VM layout, for example
`cp configs/rs_north_2023.yaml configs/rs_north_pilot.yaml` and set `grid.reference:
/data/geocrop/reference/grid_pilot.tif`, `paths.s2_scenes: /data/geocrop/interim/s2`, `paths.s1_scenes:
/data/geocrop/interim/s1`, the reference layers under `/data/geocrop/reference/` and `paths.output:
/data/geocrop/outputs/rs_north_pilot` (relative paths resolve against the working directory
`/srv/geocrop/geocrop-engine`).
```bash
sudo -iu geocrop                                         # work as the run user
source ~/miniforge3/etc/profile.d/conda.sh && conda activate geocrop
cd /srv/geocrop/geocrop-engine
geocrop grid --aoi /data/geocrop/reference/pilot_aoi.gpkg --crs EPSG:32634 --out /data/geocrop/reference/grid_pilot.tif
geocrop stac-import --reference /data/geocrop/reference/grid_pilot.tif --start 2023-03-15 --end 2023-10-31 \
    --out /data/geocrop/interim/s2 --min-clear 0.3          # windowed reads, no full downloads
geocrop catalog --aoi-wkt-file aoi.wkt --start 2023-03-15 --end 2023-10-31 --collection SENTINEL-1 --out manifests/s1.csv
geocrop download --manifest manifests/s1.csv --out /data/geocrop/raw/s1 --limit 4
geocrop snap-graph --out configs/s1_sigma0.xml --crs EPSG:32634
geocrop snap --gpt ~/esa-snap/bin/gpt --graph configs/s1_sigma0.xml --input <GRD.zip> --output /scratch/geocrop/<id>.tif
geocrop s1-import --geotiff /scratch/geocrop/<id>.tif --product-name <GRD name> --orbit ASCENDING --out /data/geocrop/interim/s1/<id>
geocrop preflight --config configs/rs_north_pilot.yaml --inventory manifests/s1.csv \
    --out /data/geocrop/outputs/preflight_pilot.json --markdown /data/geocrop/outputs/preflight_pilot.md
geocrop run --config configs/rs_north_pilot.yaml        # only when preflight is not BLOCKED
```
Read every preflight warning before the pilot ([docs/preflight.md](../docs/preflight.md)): missing windows,
cloud-starved windows, classes that cannot be tested in every fold, CRS problems and the disk/RAM estimate.
Measure from `<output>/stages.json` and `run.log` (seconds and peak RSS per stage) and `du -sh /data/geocrop/*`,
then extrapolate by area before scaling to the whole AOI.

**Before scaling** (e.g. 25 municipalities): switch the config to the full grid and inventories, rerun
`geocrop preflight --config ... --strict` and start the long run only on `ready` (or on warnings you have
accepted in writing). Compare the pilot's measured seconds and sizes with the preflight estimate and adjust
`resources.compression_ratio` if needed; the disk guard stops `composite`, `features` and `predict` early when an
output volume lacks the estimated space.

## 3. Long runs
```bash
sudo cp deploy/geocrop-run@.service /etc/systemd/system/   # user geocrop was created in step 1
sudo systemctl daemon-reload && sudo systemctl start geocrop-run@rs_north_2023   # runs configs/rs_north_2023.yaml
journalctl -u geocrop-run@rs_north_2023 -f
```
Runs are resumable: completed stages are skipped after a restart; a lock prevents concurrent runs on one output.

## 4. Container (optional)
```bash
docker build -t geocrop .
docker run --rm --user "$(id -u geocrop):$(id -g geocrop)" -v /data/geocrop:/work \
    -v /srv/geocrop/geocrop-engine/configs:/configs:ro geocrop run --config /configs/study.yaml
```
The container's working directory is `/work`, so relative paths in the configuration resolve under the mounted
`/data/geocrop`; `--user` lets the container write to the directories owned by `geocrop`. SNAP is not in the
image; run Sentinel-1 preprocessing on the host and mount the scene folders.

## 5. Housekeeping
* Keep 20–25 % free space on `/scratch`; SNAP temporary files go to `/scratch/geocrop/snap-tmp`.
* `geocrop clean --config ... --dry-run` lists temporary files and quarantined partial outputs of failed stages
  (`<output>/.quarantine`); without `--dry-run` it removes them. `--all-intermediates` also removes composites,
  feature stacks and samples (regenerated on the next run). Scene folders, reference layers and configs are never removed.
* Archive `run_manifest.json`, `stages.json`, configs and inventories with every result.
* LPIS and field data stay on `/data/geocrop/reference` (restricted permissions); nothing is pushed to git.

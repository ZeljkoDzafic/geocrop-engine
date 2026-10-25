# VM runbook (Ubuntu 24.04, CPU only)

Recommended for the northern Republic of Srpska (~10,000 km², one season): 16 vCPU, 64 GB RAM, 2 TB local
NVMe (`/scratch`), 2–3 TB archive (`/data`), outbound HTTPS. A pilot on one or two Sentinel-2 tiles works with
8 vCPU, 32 GB RAM and 500 GB.

## 1. Setup
```bash
git clone https://github.com/ZeljkoDzafic/geocrop-engine.git /srv/geocrop/geocrop-engine
cd /srv/geocrop/geocrop-engine
sudo bash deploy/setup_ubuntu.sh            # packages, /data/geocrop and /scratch/geocrop
bash deploy/setup_ubuntu.sh --user          # Miniforge, conda env "geocrop", engine, test suite
SNAP_VERSION=11.0 bash deploy/setup_ubuntu.sh --snap   # optional, Sentinel-1 (verify installer name)
```
Credentials (never in git): `/etc/geocrop/credentials.env` with `CDSE_USERNAME=` and `CDSE_PASSWORD=`, mode 600.

## 2. Pilot (one or two tiles)
```bash
conda activate geocrop
geocrop grid --aoi /data/geocrop/reference/pilot_aoi.gpkg --crs EPSG:32634 --out /data/geocrop/reference/grid_pilot.tif
geocrop stac-import --reference /data/geocrop/reference/grid_pilot.tif --start 2023-03-15 --end 2023-10-31 \
    --out /data/geocrop/interim/s2 --min-clear 0.3          # windowed reads, no full downloads
geocrop catalog --aoi-wkt-file aoi.wkt --start 2023-03-15 --end 2023-10-31 --collection SENTINEL-1 --out manifests/s1.csv
geocrop download --manifest manifests/s1.csv --out /data/geocrop/raw/s1 --limit 4
geocrop snap-graph --out configs/s1_sigma0.xml --crs EPSG:32634
geocrop snap --gpt ~/esa-snap/bin/gpt --graph configs/s1_sigma0.xml --input <GRD.zip> --output /scratch/geocrop/<id>.tif
geocrop s1-import --geotiff /scratch/geocrop/<id>.tif --product-name <GRD name> --orbit ASCENDING --out /data/geocrop/interim/s1/<id>
geocrop run --config configs/rs_north_2023.yaml
```
Measure from `<output>/stages.json` and `run.log` (seconds and peak RSS per stage) and `du -sh /data/geocrop/*`,
then extrapolate by area before scaling to the whole AOI.

## 3. Long runs
```bash
sudo useradd -r -m geocrop && sudo cp deploy/geocrop-run@.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl start geocrop-run@rs_north_2023
journalctl -u geocrop-run@rs_north_2023 -f
```
Runs are resumable: completed stages are skipped after a restart; a lock prevents concurrent runs on one output.

## 4. Container (optional)
```bash
docker build -t geocrop .
docker run --rm -v /data/geocrop:/work geocrop run --config /work/configs/study.yaml
```
SNAP is not in the image; run Sentinel-1 preprocessing on the host and mount the scene folders.

## 5. Housekeeping
* Keep 20–25 % free space on `/scratch`; SNAP temporary files go to `/scratch/geocrop/snap-tmp`.
* Archive `run_manifest.json`, `stages.json`, configs and inventories with every result.
* LPIS and field data stay on `/data/geocrop/reference` (restricted permissions); nothing is pushed to git.

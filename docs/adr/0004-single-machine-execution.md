# 0004. Single-machine execution

* Status: Accepted
* Date: 2026-10-26

## Context
Typical studies cover one region and season: tens of Sentinel-2 tiles, tens of thousands of training pixels,
up to millions of parcels. Users are researchers and paying agencies who must run, audit and archive the
analysis themselves, often on one workstation or VM and sometimes without internet access after download.
Microservices, message queues or Kubernetes would add deployment, security and failure modes that these users
cannot operate, and would make a run harder to reproduce from a manifest.

## Decision
* GeoCrop runs as one Python package on one machine: the CLI (and the optional local web workspace, which
  launches the same CLI) executes stages in-process.
* Scalability comes from bounded memory rather than distribution: raster stages work block-wise, samples are
  read per polygon window, stages are resumable with content fingerprints (`cache`, `pipeline`).
* Parallelism is optional and local: estimators use `n_jobs` (joblib/threads), and a stage may use
  `multiprocessing` internally when it is embarrassingly parallel. Results must not depend on the worker count.
* Larger areas are split by the user into independent runs (per tile/region) sharing configuration; because
  stages communicate through files (ADR 0003), each run can also be executed on a batch scheduler or cloud VM
  without code changes.

## Consequences
* Installation is `pip install` (or the Docker image / VM script); there is no service to keep alive.
* National-scale, multi-year processing needs external orchestration of independent runs; GeoCrop does not
  provide it.
* Revisit if a supported use case needs interactive multi-user processing at scale; the layered core (ADR 0001)
  would then be wrapped by a new interface layer rather than rewritten.

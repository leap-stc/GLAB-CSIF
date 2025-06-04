#!/usr/bin/env python3

import os
import glob
import logging

from dask.distributed import Client, LocalCluster
import gcsfs
import xarray as xr
from pathlib import Path
from dask.diagnostics import ProgressBar

# ─── Clean Up Previous Log Files ───────────────────────────────────────────────
for log_file in Path(".").glob("*.log"):
    try:
        log_file.unlink()
        print(f"🗑️  Deleted old log file: {log_file}")
    except Exception as e:
        print(f"⚠️  Could not delete {log_file}: {e}")

# ─── Remove old log files ──────────────────────────────────────────────────────
for logfile in glob.glob("*.log"):
    try:
        os.remove(logfile)
    except OSError:
        pass

# ─── Logging Configuration ─────────────────────────────────────────────────────
logging.basicConfig(
    filename="recipe_CSIF.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def get_chunk_scheme(total_time, lat_size, lon_size, dtype_bytes, target_MB=125):
    """Compute chunks so that each chunk ≲ target_MB MB."""
    time_chunk = total_time
    allowed = target_MB * 1024**2
    max_spatial = allowed / (time_chunk * dtype_bytes)

    ratio = lat_size / lon_size
    lat_chunk = int((max_spatial * ratio) ** 0.5)
    lon_chunk = int((max_spatial / ratio) ** 0.5)

    lat_chunk = max(1, min(lat_size, lat_chunk))
    lon_chunk = max(1, min(lon_size, lon_chunk))

    return {"time": time_chunk, "lat": lat_chunk, "lon": lon_chunk}


def main():
    logger.info("🚀 CSIF recipe starts")

    # ─── start Dask cluster ────────────────────────────────────────────────────
    # cluster = LocalCluster(n_workers=2, threads_per_worker=2, memory_limit='20GB')
    cluster = LocalCluster(
        n_workers=8,  # ← Change from 4 to 8 workers
        threads_per_worker=1,  # ← Keep 1 thread per worker (best for I/O tasks like Zarr writing)
        memory_limit="7GB",  # ← Each worker uses max 5 GB (because 8×5 = 40GB total; fits your system)
        processes=True,  # ← Default, but good to be explicit (one process per worker)
        dashboard_address=None,
    )  # ← If you want a dashboard, can change to ':8787'
    # 72 files of 1.6GB are read and 54 chunks are written 80% of memory is reached, each batch takes about 2 minutes to be written
    client = Client(cluster)

    # ─── set up GCS paths ───────────────────────────────────────────────────────
    fs = gcsfs.GCSFileSystem()
    base_path = "leap-scratch/mitraa90/GLAB-CSIF/"
    zarr_store = "gs://leap-persistent/data-library/GLAB-CSIF/GLAB-CSIF.zarr"
    fs.mkdirs(os.path.dirname(zarr_store), exist_ok=True)

    # ─── find all .nc files ─────────────────────────────────────────────────────
    gcs_paths = fs.glob(f"{base_path}**/*.nc")
    print(f"Found {len(gcs_paths)} NetCDF files.", flush=True)
    if not gcs_paths:
        return

    # ─── peek metadata from a single file ──────────────────────────────────────
    print("Reading lat/lon sizes + dtype from one file…", flush=True)
    with fs.open(gcs_paths[0], "rb") as f:
        ds0 = xr.open_dataset(f, engine="h5netcdf", chunks={"time": 1})
    lat_size = ds0.sizes["lat"]
    lon_size = ds0.sizes["lon"]
    var = next(iter(ds0.data_vars))
    dtype_bytes = ds0[var].dtype.itemsize
    ds0.close()
    print(f"  → lat={lat_size}, lon={lon_size}, bytes/pt={dtype_bytes}", flush=True)

    # ─── set batch (time) size and build chunks ────────────────────────────────
    batch_size = 72
    chunks = {
        "time": batch_size,
        "lat": 600,
        "lon": 800,
    }  # get_chunk_scheme(total_time, lat_size, lon_size, dtype_bytes, target_MB=125)
    print("Using chunk scheme:", chunks, flush=True)

    # ─── process in batches ────────────────────────────────────────────────────
    batches = [
        gcs_paths[i : i + batch_size] for i in range(0, len(gcs_paths), batch_size)
    ]
    if len(gcs_paths) % batch_size != 0:
        raise ValueError(
            f"❌ Number of files ({len(gcs_paths)}) is not evenly divisible by batch_size ({batch_size}). "
            f"Please choose a batch_size that divides evenly into {len(gcs_paths)} otherwise, append commend fails at the last batch."
        )

    for i, batch in enumerate(batches, 1):
        print(f"\n📂 Batch {i}/{len(batches)}: opening {len(batch)} files…", flush=True)
        ds = xr.open_mfdataset(
            [fs.open(p, "rb") for p in batch],
            engine="h5netcdf",
            chunks={"time": 1},  # small chunks while loading (safe)
            combine="by_coords",
            parallel=True,
        )
        print(" DS opened and is to be chunked")
        ds = ds.chunk(chunks)
        vname = "lcspp_clear_daily"  # or any main variable you are chunking

        print(
            f"  → DS is chunked: time={ds.sizes['time']} (chunk {ds[vname].data.chunks[0]}), "
            f"lat={ds.sizes['lat']} (chunk {ds[vname].data.chunks[1]}), "
            f"lon={ds.sizes['lon']} (chunk {ds[vname].data.chunks[2]})",
            flush=True,
        )

        mode = "w" if i == 1 else "a"
        consolidated = True if i == 1 else False
        print(f"  → Writing batch to Zarr (mode={mode})…", flush=True)
        append_dim = "time" if i > 1 else None

        with ProgressBar():
            ds.to_zarr(
                zarr_store,
                mode=mode,
                append_dim=append_dim,
                consolidated=consolidated,
                compute=True,
            )
        print(f"  ✅ Batch {i} written.", flush=True)

        ds.close()

    client.close()
    cluster.close()
    logger.info("🚀 CSIF recipe ends")


if __name__ == "__main__":
    main()

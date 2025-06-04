#!/usr/bin/env python3
import logging
import requests
import tarfile
import io
import numpy as np
import gcsfs
from dask.distributed import Client, LocalCluster

# ─── Logging Configuration ─────────────────────────────────────────────────────

logging.basicConfig(
    filename="download_data_CSIF2023.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)
def main():
    cluster = LocalCluster(n_workers=2, threads_per_worker=2)
    client = Client(cluster)

    fs = gcsfs.GCSFileSystem()
    base_path = "leap-scratch/mitraa90/GLAB-CSIF/"
    fs.mkdirs(base_path, exist_ok=True)

    years = np.arange(2001,2025)

    for year in years:
        print(year)
        try:
            url =f"https://zenodo.org/records/14568491/files/LCSPP_AVHRR_v3.2_{year}.tar.gz?download=1"
            print(f"⬇️  Downloading and extracting {year}...")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                tar_bytes = io.BytesIO(r.content)
            with tarfile.open(fileobj=tar_bytes, mode="r:gz") as tar:
                for member in tar.getmembers():
                    if member.isfile() and member.name.endswith(".nc"):
                        dest = base_path + member.name
                        if fs.exists(dest):
                            print(f"✅ {member.name} already exists in GCS, skipping")
                            continue
                        print(f"→ Extracting {member.name} to {dest}")
                        fileobj = tar.extractfile(member)
                        with fs.open(dest, "wb") as f:
                            f.write(fileobj.read())
                        print(f"✅ {year} written to GCS")
                    else:
                        print(f"⚠️ Skipping unexpected member: {member.name}")
        except Exception as e:
            print(f"❌ Unexpected error while processing {year}: {e}") 

    logger.info("Finished download_data.py with decompression")  # ✅ Moved inside main

if __name__ == "__main__":
    logger.info("Starting download_data.py with decompression")
    main()
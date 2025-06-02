#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download (if needed), extract, and flatten the BDD100K **val** image archive.

After the script finishes, you will have:
./data/bdd100k/images/val/  (containing 20 k validation images)

If you later grab the *train* or *test* archives, just change the `subset`
variable near the top.
"""
import os
import shutil
import zipfile
import requests
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# UTILITIES                                                                   #
# --------------------------------------------------------------------------- #
def file_is_complete(local_path: str, remote_url: str) -> bool:
    if not os.path.exists(local_path):
        return False
    try:
        r = requests.head(remote_url, timeout=10)
        r.raise_for_status()
        return os.path.getsize(local_path) == int(r.headers.get("content-length", 0))
    except Exception:
        # If we cannot verify, assume the file is OK.
        return True


def download(url: str, dst: str) -> None:
    """Stream-download *url* into *dst* with a tqdm progress bar."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    head = requests.head(url, timeout=10)
    head.raise_for_status()
    total = int(head.headers.get("content-length", 0))

    with requests.get(url, stream=True, timeout=30) as r, \
         open(dst, "wb") as f,                                    \
         tqdm(total=total, unit="B", unit_scale=True,
              desc=os.path.basename(dst)) as bar:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:                                   # skip keep-alive
                f.write(chunk)
                bar.update(len(chunk))


def extract(zip_file: str, to_dir: str) -> str:
    """
    Extract ZIP to *to_dir* and return the **deepest path that contains the
    actual images**.  For the BDD100K archives this is always
    .../images/bdd100k/images/100k/<subset>
    """
    print(f"\nExtracting {os.path.basename(zip_file)} ...")
    if os.path.exists(to_dir):
        shutil.rmtree(to_dir)
    os.makedirs(to_dir, exist_ok=True)

    with zipfile.ZipFile(zip_file) as z, \
         tqdm(total=len(z.namelist()), unit="files") as bar:
        for member in z.namelist():
            z.extract(member, to_dir)
            bar.update(1)

    # autodetect the “deepest” folder that actually holds image files
    for root, dirs, files in os.walk(to_dir):
        if files and all(f.lower().endswith(".jpg") for f in files):
            return root                 # ← e.g. .../100k/val
    raise RuntimeError("Could not locate extracted images!")


def move_into_place(src_dir: str, dst_dir: str) -> None:
    """Move *src_dir* wholesale to *dst_dir*, replacing existing dir if needed."""
    if os.path.exists(dst_dir):
        print(f"Overwriting existing {dst_dir}")
        shutil.rmtree(dst_dir)
    os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
    shutil.move(src_dir, dst_dir)


# --------------------------------------------------------------------------- #
# MAIN                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:

    # --------------------------------------------------------------------------- #
    # CONFIGURATION                                                               #
    # --------------------------------------------------------------------------- #
    subsets      = ["train", "val"]

    for subset in subsets:
        url         = f"https://dl.cv.ethz.ch/bdd100k/data/100k_images_{subset}.zip"
        root_dir    = "./data/bdd100k"            # everything lives under here
        zip_path    = os.path.join(root_dir, os.path.basename(url))
        final_dir   = os.path.join(root_dir, "images", subset)        # <- target
        scratch_dir = os.path.join(root_dir, "_extract_tmp")          # <- scratch

        # 1. Download (if necessary)
        if not file_is_complete(zip_path, url):
            print(f"Downloading {url} …")
            download(url, zip_path)
        else:
            print(f"Found complete archive: {zip_path}")

        # 2. Extract to scratch dir
        extracted_leaf = extract(zip_path, scratch_dir)

        # 3. Move images/<subset> to the final flat location
        move_into_place(extracted_leaf, final_dir)

        # 4. Clean up
        if os.path.exists(scratch_dir):
            shutil.rmtree(scratch_dir)
        print(f"\nFinished!  Images are now in: {final_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user – exiting.")
import argparse, shutil, zipfile, requests
from pathlib import Path
from tqdm.auto import tqdm

BASE_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
ZIP_FILES = [
    "VOCtrainval_06-Nov-2007.zip",   # 2007 train/val
    "VOCtest_06-Nov-2007.zip",       # 2007 test
    "VOCtrainval_11-May-2012.zip",   # 2012 train/val
]
SPLITS = [
    ("2007", "train"), ("2007", "val"), ("2007", "test"),
    ("2012", "train"), ("2012", "val"),
]

# ZIP_FILES = [
#     "VOCtest_06-Nov-2007.zip",   # 2007 test  (✔ 그대로)
# ]

# # `train`·`val`·2012 항목을 모두 지우고 2007-test 하나만!
# SPLITS = [
#     ("2007", "test"),
# ]

def download(url: str, dst: Path, chunk=1 << 20):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"[스킵] {dst.name} (이미 존재)")
        return
    print(f"[다운로드] {dst.name}")
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with dst.open("wb") as f, tqdm(total=total, unit="B",
                                       unit_scale=True, desc=dst.name) as bar:
            for c in r.iter_content(chunk):
                if c:
                    f.write(c); bar.update(len(c))

def extract(zip_path: Path, root: Path, force: bool):
    done_flag = zip_path.with_suffix(".done")
    if done_flag.exists() and not force:
        print(f"[스킵] {zip_path.name} (이미 해제)")
        return
    print(f"[압축 해제] {zip_path.name}")
    with zipfile.ZipFile(zip_path) as z:
        for m in tqdm(z.infolist(), desc=zip_path.name):
            z.extract(m, root)
    done_flag.touch()

def move_images(root: Path):
    voc_root = root / "VOCdevkit"
    img_root = root / "images"
    for year, split in SPLITS:
        dst = img_root / f"{split}{year}"; dst.mkdir(parents=True, exist_ok=True)
        ids = (voc_root / f"VOC{year}/ImageSets/Main/{split}.txt"
               ).read_text().strip().split()
        src_dir = voc_root / f"VOC{year}/JPEGImages"
        for i in tqdm(ids, desc=f"{split}{year}", ncols=80):
            shutil.move(str(src_dir / f"{i}.jpg"), dst / f"{i}.jpg")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/voc"))
    ap.add_argument("--force_extract", action="store_true",
                    help="기존 해제 마커 무시하고 다시 풂")
    ap.add_argument("--delete_zips", action="store_true")
    args = ap.parse_args()
    root = args.root.expanduser(); root.mkdir(parents=True, exist_ok=True)

    for z in ZIP_FILES:
        download(BASE_URL + z, root / z)
        extract(root / z, root, args.force_extract)

    move_images(root)

    if args.delete_zips:
        for z in ZIP_FILES: (root / z).unlink(missing_ok=True)

    print(f"\n이미지 준비 완료 → {root/'images'}")

if __name__ == "__main__":
    main()
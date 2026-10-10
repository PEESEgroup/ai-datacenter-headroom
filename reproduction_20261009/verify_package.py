"""Verify packaged inputs/code and restore compressed large fixtures."""
from pathlib import Path
import gzip
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify():
    for entry in json.loads((ROOT / "PACKAGE_MANIFEST.json").read_text())["files"]:
        path = ROOT / entry["path"]
        if not path.is_file() or sha(path) != entry["sha256"]:
            raise ValueError("Package bytes differ: " + entry["path"])
    restored = []
    for entry in json.loads((ROOT / "COMPRESSED_FILES.json").read_text()):
        target = ROOT / entry["target"]
        if not target.exists() or sha(target) != entry["uncompressed_sha256"]:
            target.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(ROOT / entry["stored"], "rb") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
        if sha(target) != entry["uncompressed_sha256"]:
            raise ValueError("Restored bytes differ: " + entry["target"])
        restored.append(entry["target"])
    return {"passed": True, "restored_large_fixtures": restored}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))

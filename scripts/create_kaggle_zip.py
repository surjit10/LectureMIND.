#!/usr/bin/env python3
import os
import zipfile
from pathlib import Path

WORKSPACE_ROOT = Path("/home/surjit/Desktop/lecuremid")
OUTPUT_ZIP_DIST = WORKSPACE_ROOT / "dist" / "lecturemind-code-kaggle.zip"
OUTPUT_ZIP_ROOT = WORKSPACE_ROOT / "lecturemind-code-kaggle.zip"

ROOT_FILES = [
    "README.md",
    "config.py",
    "requirements.txt",
    "run_kaggle.py",
    "run_kaggle_reranker_training.py",
]

INCLUDE_DIRS = [
    "cloud",
    "schemas",
]

EXCLUDE_PATTERNS = [
    "__pycache__",
    ".pytest_cache",
    ".pyc",
    ".pyo",
    ".DS_Store",
    "cloud/tests",
]

def should_exclude(rel_path: str) -> bool:
    normalized = rel_path.replace("\\", "/")
    for pattern in EXCLUDE_PATTERNS:
        if pattern in normalized:
            return True
    return False

def build_zip(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    added_files = []

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        # Add root files
        for filename in ROOT_FILES:
            file_path = WORKSPACE_ROOT / filename
            if file_path.is_file():
                zf.write(file_path, arcname=filename)
                added_files.append(filename)
            else:
                print(f"Warning: Root file missing: {filename}")

        # Add include directories
        for dirname in INCLUDE_DIRS:
            dir_path = WORKSPACE_ROOT / dirname
            for root, dirs, files in os.walk(dir_path):
                # Prune excluded dirs in-place
                dirs[:] = [d for d in dirs if not should_exclude(os.path.join(root, d))]
                
                for file in sorted(files):
                    full_path = Path(root) / file
                    rel_path = full_path.relative_to(WORKSPACE_ROOT).as_posix()
                    if not should_exclude(rel_path):
                        zf.write(full_path, arcname=rel_path)
                        added_files.append(rel_path)

    print(f"Created zip at: {output_path}")
    print(f"Total files included: {len(added_files)}")
    print(f"Archive size: {output_path.stat().st_size / 1024:.2f} KB")
    return added_files

if __name__ == "__main__":
    added = build_zip(OUTPUT_ZIP_DIST)
    # Also write to workspace root for quick access
    import shutil
    shutil.copy2(OUTPUT_ZIP_DIST, OUTPUT_ZIP_ROOT)
    print(f"Copied to workspace root: {OUTPUT_ZIP_ROOT}")

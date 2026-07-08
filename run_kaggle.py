import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import re

def generate_lecture_id(video_path, existing_ids):
    filename = os.path.basename(video_path)
    # Remove file extension
    name, _ = os.path.splitext(filename)
    # Replace anything that isn't a letter or number with an underscore
    sanitized = re.sub(r'[^a-zA-Z0-9]', '_', name)
    # Collapse multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading and trailing underscores
    base_id = sanitized.strip('_')
    
    # Ensure it's safe and fallback if empty
    if not base_id:
        base_id = "lecture"
        
    final_id = base_id
    counter = 2
    while final_id in existing_ids:
        final_id = f"{base_id}_{counter}"
        counter += 1
        
    existing_ids.add(final_id)
    return final_id

# ==================================================
# EDIT THESE PATHS
# ==================================================

REPO_PATH = str(Path(__file__).resolve().parent)

VIDEO_DIR = "/kaggle/input/datasets/jit007/v7-lectureminddataset-testing-01-07-2026"

# ==================================================

SUPPORTED_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".webm"
}


def install_requirements():

    if not os.path.exists("requirements.txt"):
        print("requirements.txt not found")
        return

    print("\nInstalling requirements...")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            "requirements.txt",
        ],
        check=False,
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",

            "sentence-transformers==3.1.1",
        ],
        check=False,
    )


def discover_videos(video_dir):

    videos = []

    for root, _, files in os.walk(video_dir):

        for file in files:

            if Path(file).suffix.lower() in SUPPORTED_EXTENSIONS:

                videos.append(
                    os.path.join(root, file)
                )

    return sorted(videos)


def process_video(video_path, lecture_id):
    print("\n")
    print("=" * 80)
    print(f"PROCESSING {lecture_id}")
    print("=" * 80)
    print("VIDEO:", video_path)

    start = time.time()

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "cloud.orchestration.run_ingestion_pipeline",
        "--lecture-id",
        lecture_id,
        "--video-path",
        video_path,
    ]

    try:
        process = subprocess.run(cmd, capture_output=True, text=True)
        sys.stdout.write(process.stdout)
        sys.stderr.write(process.stderr)
        
        runtime = round((time.time() - start) / 60, 2)
        rc = process.returncode
        
        has_traceback = "Traceback (most recent call last):" in process.stderr
        is_stderr_empty = len(process.stderr.strip()) == 0
        
        if rc == 0:
            status = "SUCCESS"
        elif rc == 1:
            status = "FAILED (Python Exception)"
        elif rc == -6 or rc == 134:
            status = "FAILED (SIGABRT)"
        elif rc == -9 or rc == 137:
            status = "FAILED (SIGKILL / OOM Killer)"
        elif rc == -11 or rc == 139:
            status = "FAILED (SIGSEGV)"
        elif rc < 0:
            status = f"FAILED (Signal {-rc})"
        else:
            status = f"FAILED (Exit Code {rc})"
            
        if has_traceback and "Exception" not in status:
            status += " w/ Traceback"

        print("\n--- DIAGNOSTICS ---")
        print(f"Return Code: {rc}")
        print(f"Duration: {runtime} min")
        print(f"Stderr Empty: {is_stderr_empty}")
        print(f"Has Traceback: {has_traceback}")
        print("-------------------\n")

    except Exception as e:
        runtime = round((time.time() - start) / 60, 2)
        status = f"ERROR: {str(e)}"

    output_folder = f"/kaggle/working/cloud_runtime/lectures/{lecture_id}/"
    zip_path = f"/kaggle/working/lecturemind/transfer/{lecture_id}_knowledge_package.zip"
    zip_exists = os.path.exists(zip_path)

    print("\nSTATUS :", status)
    print("RUNTIME:", runtime, "minutes")
    print("OUTPUT :", output_folder)
    if zip_exists:
        print("ZIP    :", zip_path)
    else:
        print("ZIP    : NOT FOUND")

    return {
        "lecture_id": lecture_id,
        "video": os.path.basename(video_path),
        "status": status,
        "runtime": runtime,
        "zip": zip_exists,
    }


def main():

    print("=" * 80)
    print("LECTUREMIND V2 - KAGGLE RUNNER")
    print("=" * 80)

    print("\nPython:", sys.version)

    try:
        subprocess.run(["nvidia-smi"])
    except Exception:
        print("GPU information unavailable")

    dest_path = "/kaggle/working/lecturemind"

    print("\nCopying repository")
    print("SOURCE :", REPO_PATH)
    print("DEST   :", dest_path)

    shutil.copytree(
        REPO_PATH,
        dest_path,
        dirs_exist_ok=True
    )

    os.chdir(dest_path)

    print("\nCurrent directory:")
    print(os.getcwd())

    install_requirements()

    # ==================================================
    # OUTPUT LOCATIONS
    # ==================================================

    os.environ["LECTURE_OUTPUT_DIR"] = (
        "/kaggle/working/cloud_runtime/lectures/{lecture_id}/"
    )

    os.environ["FRAME_DIR"] = (
        "/kaggle/working/cloud_runtime/lectures/{lecture_id}/frames/"
    )

    os.environ["LOG_DIR"] = (
        "/kaggle/working/cloud_runtime/lectures/{lecture_id}/logs/"
    )

    os.makedirs(
        "/kaggle/working/cloud_runtime/lectures",
        exist_ok=True
    )

    # ==================================================

    print("\n=== MODEL VALIDATION ===")
    from config import CloudSettings
    from pathlib import Path
    
    settings = CloudSettings()
    
    required_models = {
        "WHISPER_MODEL_PATH": settings.WHISPER_MODEL_PATH,
        "BGE_MODEL_PATH": settings.BGE_MODEL_PATH,
        "QWEN_VL_MODEL_PATH": settings.QWEN_VL_MODEL_PATH,
        "QWEN_TEXT_MODEL_PATH": settings.QWEN_TEXT_MODEL_PATH,
    }
    
    for name, path in required_models.items():
        print(path)
        exists = Path(path).exists()
        print(exists)
        if not exists:
            raise FileNotFoundError(f"Missing required model: {name} at {path}")

    # ==================================================

    # === DEPENDENCY COMPATIBILITY GUARD ===
    print("\n=== DEPENDENCY GUARD ===")
    import transformers
    tf_version = transformers.__version__
    print(f"transformers version: {tf_version}")
    if not tf_version.startswith("4.46"):
        raise RuntimeError(
            f"Transformers version {tf_version} is not compatible. "
            f"Expected 4.46.x. Do NOT upgrade transformers."
        )

    # Verify vLLM is NOT installed (prevents dependency conflicts).
    try:
        import vllm  # noqa: F401
        print("WARNING: vLLM is installed. This may cause dependency conflicts.")
        print("         Consider uninstalling: pip uninstall vllm -y")
    except ImportError:
        print("vLLM: not installed (OK)")

    print("Dependency guard passed.\n")

    # ==================================================

    videos = discover_videos(VIDEO_DIR)

    if not videos:

        print("\nNo videos found.")
        return

    print("\nDetected Videos")
    print("-" * 80)

    for idx, video in enumerate(videos, start=1):

        print(
            f"{idx:02d}. "
            f"{os.path.basename(video)}"
        )

    print("-" * 80)

    results = []
    total_start = time.time()

    existing_ids = set()
    for video_path in videos:
        lecture_id = generate_lecture_id(video_path, existing_ids)
        result = process_video(video_path, lecture_id)
        results.append(result)

    total_runtime = round(
        (time.time() - total_start) / 60,
        2
    )

    print("\n")
    print("=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    for row in results:

        print(
            f"{row['lecture_id']:10} | "
            f"{row['status']:10} | "
            f"{row['runtime']:6} min | "
            f"ZIP={'YES' if row['zip'] else 'NO'} | "
            f"{row['video']}"
        )

    print("\nTotal Runtime:", total_runtime, "minutes")

    print("\nOutputs:")
    print(
        "/kaggle/working/cloud_runtime/lectures/"
    )


if __name__ == "__main__":
    main()
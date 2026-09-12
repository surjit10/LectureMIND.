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

# ==================================================
# KAGGLE OUTPUT LOCATIONS
# ==================================================
# After a run, /kaggle/working is consolidated to contain ONLY transfer/*.zip
# so the Kaggle output download is small and contains just the knowledge packages.

WORKING_DIR = "/kaggle/working"


def cleanup_lecture_intermediates(working_dir, lecture_id):
    """Delete one lecture's intermediate output directory.

    Safe to call as soon as that lecture's knowledge_package.zip has been
    exported: the zip is self-contained, so frames/logs/transcripts/embeddings
    are dead weight. Called after each video to keep disk usage bounded when
    multiple videos run in a single session.
    """
    if not lecture_id:
        return
    lecture_dir = os.path.join(working_dir, "cloud_runtime", "lectures", lecture_id)
    if os.path.isdir(lecture_dir):
        shutil.rmtree(lecture_dir, ignore_errors=True)
        print(f"  [cleanup] removed intermediates: {lecture_dir}")


def consolidate_outputs(working_dir):
    """Move every exported zip into <working_dir>/transfer/ and delete the rest.

    Final layout on Kaggle:
        /kaggle/working/
          transfer/
            <lecture>_knowledge_package.zip
            ...

    Deletes the copied repository and the per-lecture intermediate tree, so the
    downloaded Kaggle output contains only the useful knowledge packages.
    """
    transfer_src = os.path.join(working_dir, "lecturemind", "transfer")
    transfer_dst = os.path.join(working_dir, "transfer")
    os.makedirs(transfer_dst, exist_ok=True)

    moved = []
    if os.path.isdir(transfer_src):
        moved = [
            name
            for name in sorted(os.listdir(transfer_src))
            if name.endswith("_knowledge_package.zip")
        ]
        if moved:
            for name in moved:
                shutil.move(
                    os.path.join(transfer_src, name),
                    os.path.join(transfer_dst, name),
                )
            print(f"[consolidate] moved {len(moved)} knowledge package(s) to {transfer_dst}")
        else:
            print("[consolidate] no knowledge packages found in transfer/")
    else:
        print("[consolidate] no transfer/ dir found under the repo copy")

    for junk in (
        os.path.join(working_dir, "lecturemind"),
        os.path.join(working_dir, "cloud_runtime"),
    ):
        if os.path.isdir(junk):
            shutil.rmtree(junk, ignore_errors=True)
            print(f"[consolidate] removed: {junk}")


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

    # Propagate USE_TF/USE_JAX into the child process explicitly so that
    # even if the parent env is mutated later these flags are always present.
    child_env = os.environ.copy()
    child_env.setdefault("USE_TF", "0")
    child_env.setdefault("USE_JAX", "0")

    try:
        process = subprocess.run(cmd, capture_output=True, text=True, env=child_env)
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
    # FRAMEWORK GUARD (TF/JAX off — protobuf conflict)
    # ==================================================
    # Kaggle ships TensorFlow, whose generated protobuf code needs a newer
    # protobuf (google.protobuf.runtime_version), but we pin protobuf==3.20.3
    # for PaddlePaddle 2.6.x compatibility (Paddle aborts at import with newer
    # protobuf). transformers probes for TF with importlib.util.find_spec()
    # and, when it finds it, imports it while loading Qwen2-VL — that import
    # crashes with "cannot import name 'runtime_version' from 'google.protobuf'".
    #
    # Two layers of defense (the pipeline is 100% PyTorch; TF is never used):
    #   1. USE_TF=0 / USE_JAX=0 — transformers 4.46.x honours these env vars
    #      and skips the TF/JAX backends entirely.
    #   2. sys.modules["tensorflow"] = None — makes find_spec("tensorflow")
    #      return None, so any code that probes for TF sees it as unavailable.
    #      This is deterministic and independent of env-var support.
    #
    # NOTE: sys.modules is per-process. The pipeline subprocess re-applies the
    # same guard itself (see run_ingestion_pipeline.py); only the USE_TF/USE_JAX
    # env vars propagate into it.
    os.environ["USE_TF"] = "0"
    os.environ["USE_JAX"] = "0"
    sys.modules.setdefault("tensorflow", None)
    print("[env] TF/JAX disabled (USE_TF=0 + find_spec guard) to avoid protobuf conflict")

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
        # Nothing was processed, but the repo copy and empty runtime tree are
        # still in /kaggle/working — clean them so the output stays small.
        consolidate_outputs(WORKING_DIR)
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
        # Free disk before the next video: the zip is already exported, so this
        # lecture's intermediates (frames/logs/transcripts) are dead weight.
        # Keep them for failed lectures so the session logs remain inspectable.
        if result["zip"]:
            cleanup_lecture_intermediates(WORKING_DIR, lecture_id)

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

    # ==================================================
    # CONSOLIDATE OUTPUTS
    # ==================================================
    print("\n=== CONSOLIDATING OUTPUTS ===")
    # Step out of the repo copy before it is deleted, so the process is never
    # rooted inside a directory tree that consolidation removes.
    os.chdir(WORKING_DIR)
    consolidate_outputs(WORKING_DIR)

    print("\nOutputs:")
    print(os.path.join(WORKING_DIR, "transfer"))


if __name__ == "__main__":
    main()
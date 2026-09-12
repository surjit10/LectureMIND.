import os
import shutil
import subprocess
import sys
from pathlib import Path

# ==================================================
# KAGGLE NOTEBOOK B — GLOBAL RERANKER TRAINING
# ==================================================
# This is a COMPLETELY INDEPENDENT Kaggle workflow from the lecture processing
# notebook (run_kaggle.py). It never touches videos.
#
# Input:   previously exported lecture knowledge packages (zips) mounted as a
#          Kaggle dataset under /kaggle/input (or RERANKER_TRAINING_PACKAGES_ROOT).
# Output:  /kaggle/working/reranker_models/
#            v1/, v2/, ..., best/, latest/, index.json
#            global_reranker_v{N}.zip   ← download this, upload via
#              POST /api/reranker/upload to hot-reload the local server.
#
# The ONLY artifact shared with the lecture pipeline is triplets.json.

REPO_PATH = str(Path(__file__).resolve().parent)

# Kaggle mounts every attached dataset under /kaggle/input/<slug>. The
# "datasets" entry holds the big model datasets (jit007) — never scanned.
DEFAULT_PACKAGES_ROOT = "/kaggle/input"
MODELS_ROOT = "/kaggle/working/reranker_models/"
SCRATCH_ROOT = "/kaggle/working/reranker_training/scratch/"
WORKING_DIR = "/kaggle/working"


def resolve_packages_root() -> str:
    """Use RERANKER_TRAINING_PACKAGES_ROOT if set, else scan /kaggle/input."""
    env_root = os.environ.get("RERANKER_TRAINING_PACKAGES_ROOT", "").strip()
    return env_root or DEFAULT_PACKAGES_ROOT


def install_requirements() -> None:
    # Reuse Pipeline A's pip install logic — no duplicated business logic.
    from run_kaggle import install_requirements

    install_requirements()


def check_gpu() -> None:
    """GPU-only training. Abort with a clear message when CUDA is missing."""
    print("\n=== GPU CHECK ===")
    try:
        subprocess.run(["nvidia-smi"])
    except Exception:
        print("nvidia-smi unavailable")
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU required for reranker training but "
                "torch.cuda.is_available() is False. Attach a GPU to this "
                "notebook (GPU T4 x2 recommended) and re-run."
            )
        print(f"torch {torch.__version__} — CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        raise RuntimeError("torch not installed. Install requirements first.")


def consolidate_outputs() -> None:
    """
    Keep only the versioned model store + exported zips in /kaggle/working.

    Removes the repo copy and the training scratch tree so the downloaded
    Kaggle output is small and contains just reranker_models/.
    """
    models = Path(MODELS_ROOT)
    if models.is_dir():
        print(f"[consolidate] keeping reranker_models/ -> {models}")
    else:
        print("[consolidate] WARNING: reranker_models/ not found")

    for junk in (
        os.path.join(WORKING_DIR, "lecturemind"),
        os.path.join(WORKING_DIR, "reranker_training"),
    ):
        if os.path.isdir(junk):
            shutil.rmtree(junk, ignore_errors=True)
            print(f"[consolidate] removed: {junk}")


def main() -> None:
    print("=" * 80)
    print("LECTUREMIND - KAGGLE NOTEBOOK B: GLOBAL RERANKER TRAINING")
    print("=" * 80)
    print("\nPython:", sys.version)

    dest_path = "/kaggle/working/lecturemind"
    print("\nCopying repository")
    print("SOURCE :", REPO_PATH)
    print("DEST   :", dest_path)
    shutil.copytree(REPO_PATH, dest_path, dirs_exist_ok=True)
    os.chdir(dest_path)

    install_requirements()
    check_gpu()

    # ==================================================
    # FRAMEWORK GUARD (TF/JAX off — protobuf conflict)
    # ==================================================
    # Same as run_kaggle.py: Kaggle ships TensorFlow, whose generated protobuf
    # code needs a newer protobuf (google.protobuf.runtime_version), but we pin
    # protobuf==3.20.3 for PaddlePaddle 2.6.x compatibility. transformers
    # probes for TF via find_spec() and crashes when it imports it. The
    # training subprocess re-applies the sys.modules guard itself
    # (cloud/reranker_training/pipeline.py); the env vars below propagate into
    # it and are also honored by transformers 4.46.x directly.
    os.environ["USE_TF"] = "0"
    os.environ["USE_JAX"] = "0"
    sys.modules.setdefault("tensorflow", None)
    print("[env] TF/JAX disabled (USE_TF=0 + find_spec guard) to avoid protobuf conflict")

    packages_root = resolve_packages_root()
    print("\nPackages root:", packages_root)

    # Environment contract honored by cloud/reranker_training/pipeline.py's
    # CLI defaults (explicit flags always win). The pipeline builds its own
    # scratch CloudSettings, so LECTURE_OUTPUT_DIR is intentionally NOT set.
    os.environ["RERANKER_TRAINING_MODELS_ROOT"] = MODELS_ROOT
    os.environ["RERANKER_TRAINING_SCRATCH_ROOT"] = SCRATCH_ROOT

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "cloud.reranker_training.pipeline",
        "--packages-root",
        packages_root,
        "--models-root",
        MODELS_ROOT,
        "--scratch-root",
        SCRATCH_ROOT,
    ] + sys.argv[1:]  # forward any extra args (epochs, batch-size, ...)

    print("\nRunning:", " ".join(cmd))
    process = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(process.stdout)
    sys.stderr.write(process.stderr)

    print("\n" + "=" * 80)
    if process.returncode == 0:
        print("PIPELINE B: SUCCESS")
    else:
        print(f"PIPELINE B: FAILED (exit {process.returncode})")

    print("=" * 80)

    os.chdir(WORKING_DIR)
    consolidate_outputs()

    print("\nOutputs:")
    print(os.path.join(WORKING_DIR, "reranker_models"))
    if process.returncode != 0:
        sys.exit(process.returncode)


if __name__ == "__main__":
    main()

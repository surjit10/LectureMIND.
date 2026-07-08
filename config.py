# config.py
# Centralized config for LECTUREMIND V2 using pydantic-settings.
#
# Three-class split enforcing cloud/local isolation:
#   - SharedSettings: imported by BOTH cloud and local code.
#   - CloudSettings:  imported ONLY by cloud/ modules.
#   - LocalSettings:  imported ONLY by local/ modules.
#
# EMBEDDING_DIMENSION must always equal 1024 (bge-large-en-v1.5).
# The most common AI-agent mistake is hardcoding 768 from older BGE models — this raises immediately.

from pydantic import field_validator
from pydantic_settings import BaseSettings


class SharedSettings(BaseSettings):
    """
    Settings shared across cloud and local environments.
    Imported by both cloud/ and local/ modules.
    """

    # Embedding dimension — MUST be 1024.
    # bge-large-en-v1.5 outputs 1024-dim vectors.
    # 768 is the old bge-base model and is NOT used in this project.
    EMBEDDING_DIMENSION: int = 1024

    @field_validator("EMBEDDING_DIMENSION")
    @classmethod
    def must_be_1024(cls, v: int) -> int:
        if v != 1024:
            raise ValueError(
                f"EMBEDDING_DIMENSION must be 1024 (bge-large-en-v1.5). Got {v}. "
                "768 is the old bge-base model and is not permitted."
            )
        return v

    model_config = {"env_file": ".env", "case_sensitive": False, "extra": "ignore"}


class CloudSettings(BaseSettings):
    """
    Cloud (Kaggle) environment settings.
    Imported ONLY by cloud/ modules. Never imported by local/ modules.
    """

    KAGGLE_OUTPUT_ROOT: str = "/kaggle/working"
    LECTURE_OUTPUT_DIR: str = "cloud_runtime/lectures/{lecture_id}/"
    FRAME_DIR: str = "cloud_runtime/lectures/{lecture_id}/frames/"
    LOG_DIR: str = "cloud_runtime/lectures/{lecture_id}/logs/"

    # PaddleOCR GPU control.
    # Default False because PaddlePaddle's C++ runtime may call abort()
    # (SIGABRT) when cuDNN libraries are missing, which kills the process
    # before Python exception handling can intercept it.
    # Set to True ONLY when cuDNN is verified present.
    OCR_USE_GPU: bool = False

    # Kafka (cloud-side, between A6 and consumers)
    KAFKA_BROKER: str = "localhost:9092"

    # Kaggle dataset root — all model datasets live under this prefix.
    KAGGLE_DATA_ROOT: str = "/kaggle/input/datasets/jit007"

    # Derived model roots.
    CORE_ROOT: str = "/kaggle/input/datasets/jit007/lecturemind-core-models"
    QWEN_VL_ROOT: str = "/kaggle/input/datasets/jit007/lecturemind-qwen2-vl-7b"
    QWEN_TEXT_ROOT: str = "/kaggle/input/datasets/jit007/lecturemind-qwen2-7b"

    # Resolved model paths.
    WHISPER_MODEL_PATH: str = "/kaggle/input/datasets/jit007/lecturemind-core-models/faster-whisper-large-v3"
    BGE_MODEL_PATH: str = "/kaggle/input/datasets/jit007/lecturemind-core-models/bge-large-en-v1.5"
    QWEN_VL_MODEL_PATH: str = "/kaggle/input/datasets/jit007/lecturemind-qwen2-vl-7b/qwen2-vl-7b-instruct"
    QWEN_TEXT_MODEL_PATH: str = "/kaggle/input/datasets/jit007/lecturemind-qwen2-7b/qwen2_5_7b"

    model_config = {"env_file": ".env", "case_sensitive": False, "extra": "ignore"}

    def lecture_dir(self, lecture_id: str) -> str:
        """Return the resolved output directory for a given lecture."""
        return self.LECTURE_OUTPUT_DIR.format(lecture_id=lecture_id)

    def frame_dir(self, lecture_id: str) -> str:
        """Return the resolved frame directory for a given lecture."""
        return self.FRAME_DIR.format(lecture_id=lecture_id)

    def log_dir(self, lecture_id: str) -> str:
        """Return the resolved log directory for a given lecture."""
        return self.LOG_DIR.format(lecture_id=lecture_id)


class LocalSettings(BaseSettings):
    """
    Local (laptop/server) environment settings.
    Imported ONLY by local/ modules. Never imported by cloud/ modules.
    """

    NEO4J_URI: str = "bolt://localhost:7687"
    QDRANT_URL: str = "http://localhost:6333"
    OLLAMA_MODEL: str = "qwen2.5:3b"
    LOCAL_MODEL_DIR: str = "local_runtime/models/"

    # Global reranker — loaded once at startup, shared for the application lifetime.
    # Fine-tuned global model goes here; fallback is RERANKER_MODEL_ID from HuggingFace.
    GLOBAL_RERANKER_DIR: str = "local_runtime/models/global_reranker/"
    RERANKER_MODEL_ID: str = "BAAI/bge-reranker-base"
    UPLOAD_TEMP_DIR: str = "local_runtime/temp/uploads/"
    MAX_MODEL_UPLOAD_MB: int = 2000
    ALLOWED_MODEL_FILES: list[str] = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "model.safetensors", "pytorch_model.bin", "vocab.txt"]
    AUTO_RELOAD_RERANKER: bool = True

    TRANSFER_DIR: str = "transfer/"
    LOCAL_LOG_DIR: str = "local_runtime/logs/"
    LOCAL_CACHE_DIR: str = "local_runtime/cache/"
    LOCAL_UPLOAD_DIR: str = "local_runtime/uploads/"

    ENABLE_AUTO_MODEL_RECOVERY: bool = True
    ENABLE_AUTO_OLLAMA_PULL: bool = True

    model_config = {"env_file": ".env", "case_sensitive": False, "extra": "ignore"}



# Module-level singletons — import the appropriate one per environment.
shared_settings = SharedSettings()
cloud_settings = CloudSettings()
local_settings = LocalSettings()

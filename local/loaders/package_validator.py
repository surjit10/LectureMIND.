import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REQUIRED_FILES = [
    "manifest.json",
    "segments.json",
    "entities.json",
    "relations.json",
    "embeddings.npy",
    "embedding_ids.json",
    "multimodal_chunks.json"
]

def validate_package(package_dir: str | Path) -> None:
    """
    Validates that the extracted knowledge package contains all required files.
    Raises ValueError if any are missing.
    """
    package_dir = Path(package_dir)
    missing_files = []
    
    for filename in REQUIRED_FILES:
        file_path = package_dir / filename
        if not file_path.exists() or not file_path.is_file():
            missing_files.append(filename)
            
    if missing_files:
        error_msg = f"Knowledge package validation failed. Missing required files: {', '.join(missing_files)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    logger.info("Knowledge package validation successful for %s", package_dir)

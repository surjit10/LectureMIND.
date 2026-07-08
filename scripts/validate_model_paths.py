import sys
import os

# Add parent directory to sys.path to allow importing config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import CloudSettings
from transformers import AutoConfig

def main():
    settings = CloudSettings()
    
    text_path = settings.QWEN_TEXT_MODEL_PATH
    vl_path = settings.QWEN_VL_MODEL_PATH
    
    print(f"TEXT MODEL PATH: {text_path}")
    print(f"VISION MODEL PATH: {vl_path}")
    
    if not os.path.exists(text_path):
        print(f"ERROR: TEXT MODEL PATH DOES NOT EXIST: {text_path}")
        sys.exit(1)
        
    if not os.path.exists(vl_path):
        print(f"ERROR: VISION MODEL PATH DOES NOT EXIST: {vl_path}")
        sys.exit(1)
        
    text_config = AutoConfig.from_pretrained(text_path, trust_remote_code=True)
    vl_config = AutoConfig.from_pretrained(vl_path, trust_remote_code=True)
    
    print(f"TEXT MODEL TYPE: {getattr(text_config, 'model_type', 'UNKNOWN')}")
    print(f"VISION MODEL TYPE: {getattr(vl_config, 'model_type', 'UNKNOWN')}")

if __name__ == "__main__":
    main()

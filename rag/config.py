import os
from dotenv import load_dotenv

# Load environment variables once, at import time
load_dotenv()

# Hard-keep same paths/values your code used
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
DATA_DIR = os.getenv("DATA_DIR", "data")

def require_env(var: str) -> str:
    val = os.getenv(var)
    if not val:
        raise RuntimeError(f"{var} not found. Create .env with {var}=...")
    return val

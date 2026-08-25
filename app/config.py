"""Config adapter for eval loop - exposes backend.config as app.config."""
from backend import config as backend_config

# Expose all backend config values
GENERATION_BACKEND = "api"
GENERATION_MODEL = "openai/gpt-oss-20b"
LOCAL_GENERATION_MODEL = "fastembed-paraphrase-multilingual-MiniLM-L12-v2"
LATENCY_BUDGET_MS = 200

# Pass through any other config values from backend.config
for attr in dir(backend_config):
    if not attr.startswith('_'):
        globals()[attr] = getattr(backend_config, attr)
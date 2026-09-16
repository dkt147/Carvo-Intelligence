"""Dummy LLM key so FastAPI startup validation (D9 / Q18) can run in tests.

Not a real secret. Production must set LLM_API_KEY in the environment.
"""

import os

os.environ.setdefault("LLM_API_KEY", "test-not-a-real-key")

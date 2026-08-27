import os
import sys
from pathlib import Path

# Force the mock LLM backend for the whole test session so CI never needs a
# real ANTHROPIC_API_KEY secret just to prove the pipeline runs end-to-end.
os.environ["USE_MOCK_LLM"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

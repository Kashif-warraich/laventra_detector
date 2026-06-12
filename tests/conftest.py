"""Pytest setup — adds the project root to sys.path so `import config` works."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

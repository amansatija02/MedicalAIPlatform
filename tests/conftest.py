import sys
from pathlib import Path

# lets "pytest" find config/ and src/ when it is run from any folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

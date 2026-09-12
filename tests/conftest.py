import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for relative in ("postprocess/src", "shared/src", "code"):
    sys.path.insert(0, str(ROOT / relative))

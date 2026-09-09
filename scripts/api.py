import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from infer.api import main as api_main


if __name__ == "__main__":
    api_main()
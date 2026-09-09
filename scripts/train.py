import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train.train import main as train_main


if __name__ == "__main__":
    train_main()
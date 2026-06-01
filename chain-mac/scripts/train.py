import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import Config
from src.trainer import Trainer


def main():
    cfg = Config()
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()

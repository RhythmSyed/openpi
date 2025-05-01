"""
Script for converting PushT data to LeRobot format.
"""

import shutil
from pathlib import Path
import pandas as pd

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro

REPO_NAME = "msr915/pusht_simulation"  # Change this to your desired name

def main(data_dir: str, *, push_to_hub: bool = False):
    # First let's examine the data structure
    data_path = Path(data_dir)
    first_chunk = next(data_path.glob("chunk-*/file-*.parquet"))
    df = pd.read_parquet(first_chunk)
    print("Available columns in the parquet file:")
    print(df.columns)
    print("\nFirst row sample:")
    print(df.iloc[0])

if __name__ == "__main__":
    tyro.cli(main)
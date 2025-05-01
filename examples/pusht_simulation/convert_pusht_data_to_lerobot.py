"""
Script for converting raw PushT zarr data to LeRobot format.
"""

import shutil
from pathlib import Path
import zarr
import numpy as np

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro

REPO_NAME = "msr915/pusht_sim_img_state_action"  # Change this to your desired name

def main(data_dir: str, *, push_to_hub: bool = False):
    # Open zarr dataset first to check data format
    zarr_path = Path(data_dir) / "pusht_cchi_v7_replay.zarr"
    zarr_dataset = zarr.open(str(zarr_path), mode='r')
    
    # Print data formats
    print("\n=== State Data Analysis ===")
    print("First state shape:", zarr_dataset['data/state'][0].shape)
    print("First state data:", zarr_dataset['data/state'][0])
    
    # Print state data for first few frames to see if it changes
    print("\nState data for first 5 frames:")
    for i in range(5):
        print(f"Frame {i}:", zarr_dataset['data/state'][i])
    
    # Check if state values are normalized
    print("\nState value ranges:")
    for i in range(zarr_dataset['data/state'][0].shape[0]):
        values = zarr_dataset['data/state'][:100, i]  # Check first 100 frames
        print(f"State dim {i}: min={values.min():.3f}, max={values.max():.3f}, mean={values.mean():.3f}")
    
    print("\n=== Action Data Analysis ===")
    print("First action shape:", zarr_dataset['data/action'][0].shape)
    print("First action data:", zarr_dataset['data/action'][0])
    
    # Get number of episodes and total samples
    episode_ends = zarr_dataset['meta/episode_ends'][:]
    num_episodes = len(episode_ends)
    total_samples = episode_ends[-1]
    print(f"\n=== Dataset Stats ===")
    print(f"Total episodes: {num_episodes}")
    print(f"Total samples: {total_samples}")
    
    # Let user decide whether to proceed with conversion
    input("\nPress Enter to continue with dataset conversion...")
    
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset with PushT features
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="pusht",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (96, 96, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (5,),
                "names": ["state_0", "state_1", "state_2", "state_3", "state_4"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["motor_0", "motor_1"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    
    # Process each episode
    for episode_idx in range(num_episodes):
        # Get episode start and end indices
        start_idx = 0 if episode_idx == 0 else episode_ends[episode_idx - 1]
        end_idx = episode_ends[episode_idx]
        
        # Process each frame in the episode
        for frame_idx in range(start_idx, end_idx):
            # Get data
            image = zarr_dataset['data/img'][frame_idx]
            state = np.array(zarr_dataset['data/state'][frame_idx], dtype=np.float32)
            action = np.array(zarr_dataset['data/action'][frame_idx], dtype=np.float32)
            
            # Ensure action is 2D array
            if len(action.shape) == 1:
                action = action.reshape(2)
            
            # Add frame to dataset
            dataset.add_frame(
                {
                    "image": image,
                    "state": state,
                    "actions": action,
                }
            )
        
        # Save episode
        dataset.save_episode(task=f"push_task_{episode_idx}")

    # Consolidate the dataset
    dataset.consolidate(run_compute_stats=False)

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["pusht", "robotics"],
            private=False,
            push_videos=True,
            license="mit",
        )

if __name__ == "__main__":
    tyro.cli(main)
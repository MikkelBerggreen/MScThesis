from metrics.noise_ceiling import NoiseCeiling
from time import time
from utils.config_loader import ConfigLoader
from pathlib import Path
import numpy as np

class Run:
    def __init__(self):
        self.config = ConfigLoader()

    def run(self):
        base_path = Path(__file__).resolve().parent
        noise_ceiling_path = (base_path / f"../data/datasets/one_person/noise_ceiling.npy").resolve()
        noise_floor_path = (base_path / f"../data/datasets/one_person/noise_floor.npy").resolve()

        nc = NoiseCeiling()
        
        # Compute noise floor and save to file
        # Check if file exists
        if not noise_floor_path.exists():
            noise_floor = nc.compute_robust_noise_floor()
            np.save(noise_floor_path, noise_floor)

        # Compute noise ceiling and save to file
        if not noise_ceiling_path.exists():
            noise_ceiling = nc.compute_robust_noise_ceiling()                
            np.save(noise_ceiling_path, noise_ceiling)

if __name__ == '__main__':
    start_time = time()
    print("\nStarting run...")
    r = Run()
    r.run()
    print("Total run time:", time() - start_time)
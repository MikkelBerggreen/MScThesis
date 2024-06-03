from core.trainer import Trainer
from time import time
from utils.config_loader import ConfigLoader

class Run:
    def __init__(self):
        self.config = ConfigLoader()

    def run(self):
        trainer = Trainer()
        trainer.train(show_progress_plots=False, use_wandb=True, sweep=False)

if __name__ == '__main__':
    start_time = time()
    print("\nStarting run...")
    r = Run()
    r.run()
    print("Total run time:", time() - start_time)
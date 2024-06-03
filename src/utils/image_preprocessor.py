import os
import torchvision.transforms as TV
from PIL import Image
from pathlib import Path
import torch
from utils.config_loader import ConfigLoader

# Class used to preprocess and save a directory of images
class ImagePreprocessor:
    def __init__(self, source_dir, target_dir):
        self.config = ConfigLoader()
        self.dimensions = self.config.transformations.image.dimensions
        
        self.source_dir = source_dir
        self.target_dir = target_dir
        base_path = Path(__file__).resolve().parent
        self.source_path = (base_path / f"../../{self.source_dir}").resolve()
        self.target_path = (base_path / f"../../{self.target_dir}").resolve()
        self.transform = TV.Compose([
            TV.Resize(256),
            TV.CenterCrop(self.dimensions),
            TV.ToTensor(),
            TV.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def preprocess_image(self, image_path, save_path):
        # Modify save path extension to '.pt' for torch tensor files
        save_path = save_path.replace('.jpg', '.pt').replace('.jpeg', '.pt').replace('.png', '.pt')
        
        # Check if the processed file already exists
        if not os.path.exists(save_path):
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            # Save the tensor directly
            torch.save(image, save_path)
            print(f"Processed and saved: {save_path}")
        else:
            print(f"File already processed, skipping: {save_path}")
    
    def preprocess_directory(self):
        for root, dirs, files in os.walk(self.source_path):
            for file in files:
                if file.lower().endswith(('png', 'jpg', 'jpeg')):
                    relative_path = os.path.relpath(root, self.source_path)
                    target_path = os.path.join(self.target_path, relative_path)
                    
                    # Create target directory if it doesn't exist
                    Path(target_path).mkdir(parents=True, exist_ok=True)
                    
                    image_path = os.path.join(root, file)
                    save_path = os.path.join(target_path, file)
                    self.preprocess_image(image_path, save_path)
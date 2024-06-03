import yaml
import os
from utils.constants import CONFIG_FILE

class ConfigLoader:
    def __init__(self):
        self.load_config(CONFIG_FILE)
        
    def load_config(self, config_path):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        final_path = os.path.join(dir_path, config_path)
        with open(final_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        self.set_attributes(config_dict)
    
    def set_attributes(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):  # If the value is a dict, create a nested ConfigLoader
                setattr(self, key, ConfigLoader.from_dict(value))
            else:
                setattr(self, key, value)

    @classmethod
    def from_dict(cls, config_dict):
        config = cls.__new__(cls)  # Create a new instance without calling __init__
        cls.set_attributes(config, config_dict)
        return config

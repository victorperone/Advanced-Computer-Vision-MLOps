import yaml
import os

def load_config(profile="laptop"):
    """Loads the config.yaml file and selects the hardware profile."""
    # Find the config file relative to this script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config.yaml")
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        
    # Merge the dataset config with the specific hardware profile
    active_config = config['dataset']
    active_config.update(config[profile])

    # Save the profile name so train.py can access it!
    active_config['profile_name'] = profile
    
    # Add absolute paths for saving models
    active_config['models_dir'] = os.path.join(base_dir, 'models')
    active_config['logs_dir'] = os.path.join(base_dir, 'logs')
    if not os.path.exists(active_config['models_dir']):
        os.makedirs(active_config['models_dir'])
    
    if not os.path.exists(active_config['logs_dir']):
        os.makedirs(active_config['logs_dir'])
        
    return active_config
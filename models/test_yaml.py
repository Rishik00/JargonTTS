import yaml
import argparse

def load_yaml(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)

    # Ensure batch_size exists in the YAML file before accessing
    return data.get("batch_size", "Key 'batch_size' not found in YAML file")

# 

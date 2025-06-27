import os
import yaml

project_dir = os.path.abspath(os.path.dirname(__file__))

train_path = os.path.join(project_dir, 'dataset/train')
val_path = os.path.join(project_dir, 'dataset/val')

data_yaml = {
    'train': train_path,
    'val': val_path,
    'nc': 1,
    'names': ['dot']
}

output_file = os.path.join(project_dir, 'custom.yaml')
with open(output_file, 'w') as f:
    yaml.dump(data_yaml, f)

print(f"Archivo generado: {output_file}")

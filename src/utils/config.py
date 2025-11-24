
import yaml, json, os
def load_yaml(path): 
    with open(path, "r") as f: return yaml.safe_load(f)
def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2)

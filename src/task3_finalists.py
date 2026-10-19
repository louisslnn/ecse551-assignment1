"""Display corrected settings. Never rewrite the historical frozen configuration."""
import json
from task3_protocol import load_protocol

if __name__ == "__main__":
    print(json.dumps(load_protocol()[0], indent=2))

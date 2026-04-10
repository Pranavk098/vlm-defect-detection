import json
import os
import uuid
from pathlib import Path

# Configuration
DATASET_ROOT = Path("mvtec_anomaly_detection")
OUTPUT_FILE = "mvtec_train.json"

def create_dataset():
    if not DATASET_ROOT.exists():
        print(f"Error: Dataset root {DATASET_ROOT} not found.")
        return

    dataset = []
    
    # Iterate over all categories (bottle, carpet, etc.)
    for category_path in DATASET_ROOT.iterdir():
        if not category_path.is_dir():
            continue
            
        print(f"Processing category: {category_path.name}")
        
        # 1. Process Training Data (Usually contains only 'good' samples)
        train_path = category_path / "train" / "good"
        if train_path.exists():
            for img_path in train_path.glob("*.png"):
                relative_path = f"{category_path.name}/train/good/{img_path.name}"
                
                entry = {
                    "id": str(uuid.uuid4()),
                    "image": relative_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\nIs there any anomaly in this image? Answer 'Yes' or 'No'."
                        },
                        {
                            "from": "gpt",
                            "value": "No."
                        }
                    ]
                }
                dataset.append(entry)

        # 2. Process Test Data (Contains 'good' and defect types)
        test_path = category_path / "test"
        if test_path.exists():
            for defect_type_path in test_path.iterdir():
                if not defect_type_path.is_dir():
                    continue
                
                defect_name = defect_type_path.name
                is_anomaly = defect_name != "good"
                
                for img_path in defect_type_path.glob("*.png"):
                    relative_path = f"{category_path.name}/test/{defect_name}/{img_path.name}"
                    
                    if is_anomaly:
                        gpt_response = f"Yes, there is a {defect_name} anomaly."
                    else:
                        gpt_response = "No."

                    entry = {
                        "id": str(uuid.uuid4()),
                        "image": relative_path,
                        "conversations": [
                            {
                                "from": "human",
                                "value": "<image>\nIs there any anomaly in this image? Describe it."
                            },
                            {
                                "from": "gpt",
                                "value": gpt_response
                            }
                        ]
                    }
                    dataset.append(entry)

    print(f"Generated {len(dataset)} samples.")
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    create_dataset()

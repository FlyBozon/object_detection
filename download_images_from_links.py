#code to download images for lab testing set
# git commit -m "added script to run everything for dataset from terminal at once, modified the way of creating dataset"

import os
import requests

output_folder = "test_set"

def download_images_from_file(filepath):
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return

    os.makedirs(output_folder, exist_ok=True)

    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    current_category = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.endswith(':'):
            current_category = line[:-1]
        elif current_category:
            try:
                response = requests.get(line, timeout=10)
                response.raise_for_status()
                ext = line.split('.')[-1].split('?')[0][:5]
                filename = f"{current_category}_{abs(hash(line))}.{ext}"
                filepath_out = os.path.join(output_folder, filename)
                with open(filepath_out, 'wb') as f:
                    f.write(response.content)
                print(f"✔ Downloaded: {filepath_out}")
            except Exception as e:
                print(f"✖ Failed: {line}\n  Error: {e}")

if __name__ == "__main__":
    download_images_from_file("obrazy_linki.txt")

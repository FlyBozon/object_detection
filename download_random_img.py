import os
import requests

def download_random_images(count=50, width=256, height=256, save_dir='random_images'):
    os.makedirs(save_dir, exist_ok=True)

    current_image = 1

    for _ in range(count):
        url = f'https://picsum.photos/{width}/{height}'

        while True:
            file_path = os.path.join(save_dir, f'random_{current_image}.jpg')
            if not os.path.exists(file_path):
                break
            current_image += 1

        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f'Downloaded: {file_path}')
            current_image += 1 
        except Exception as e:
            print(f'Error downloading image {current_image}: {e}')

download_random_images(count=50, width=1024, height=1024)

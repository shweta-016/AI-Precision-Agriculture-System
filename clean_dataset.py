import os
from PIL import Image

dataset_path = "C:/Users/Admin/OneDrive/Desktop/agricultur/plant_disease_dataset"

bad_images = 0

for root, dirs, files in os.walk(dataset_path):

    for file in files:

        path = os.path.join(root, file)

        try:
            img = Image.open(path)
            img.verify()

        except:
            print("Removing corrupted file:", path)
            os.remove(path)
            bad_images += 1

print("\nTotal corrupted images removed:", bad_images)
import os

dataset = "C:/Users/Admin/OneDrive/Desktop/agricultur/plant_disease_dataset"

valid_extensions = [".jpg", ".jpeg", ".png"]

removed = 0

for root, dirs, files in os.walk(dataset):

    for file in files:

        if not file.lower().endswith(tuple(valid_extensions)):

            path = os.path.join(root, file)

            print("Removing:", path)

            os.remove(path)

            removed += 1

print("\nRemoved files:", removed)
import os
import shutil
import argparse

import pandas as pd


def sort_val_images(val_solution_path, val_image_dir):
    df = pd.read_csv(val_solution_path)
    labels = df["PredictionString"].tolist()
    labels = [x.split()[0] for x in labels]
    names = df["ImageId"].tolist()

    label_set = set()
    for x in labels:
        label_set.add(x)

    for x in label_set:
        class_dir = os.path.join(val_image_dir, x)
        os.makedirs(class_dir, exist_ok=True)

    for name, label in zip(names, labels):
        img_path = os.path.join(val_image_dir, f"{name}.JPEG")
        class_dir = os.path.join(val_image_dir, label)

        if os.path.exists(img_path):
            shutil.move(img_path, class_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_solution_path", type=str)
    parser.add_argument("--val_image_dir", type=str)

    args = parser.parse_args()
    sort_val_images(args.val_solution_path, args.val_image_dir)


if __name__ == "__main__":
    main()

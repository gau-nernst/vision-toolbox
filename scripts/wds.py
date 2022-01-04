import random
import math
import os
import argparse

import webdataset as wds
from torchvision.datasets import ImageFolder

def image_folder_to_webdataset(data_dir, save_dir, name, shuffle=False, shard_size=1e9, max_shards=1e5):
    os.makedirs(save_dir, exist_ok=True)

    ds = ImageFolder(data_dir)
    ds_len = len(ds)
    idx_str_len = int(math.log10(ds_len)) + 1
    
    indices = list(range(ds_len))
    if shuffle:
        random.shuffle(indices)

    pattern = os.path.join(save_dir, f"{name}-%06d.tar")
    with wds.ShardWriter(pattern, maxsize=shard_size, maxcount=max_shards) as sink:
        for i, idx in enumerate(indices):
            img_path, label = ds.imgs[idx]

            with open(img_path, "rb") as f:
                img = f.read()
            
            img_name = os.path.basename(img_path)
            img_ext = os.path.splitext(img_name)[-1][1:]
            
            sink.write({
                "__key__": f"{i:0{idx_str_len}d}",
                img_ext: img,
                "cls": label 
            })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--shuffle", type=bool, default=False)

    args = parser.parse_args()
    image_folder_to_webdataset(args.data_dir, args.save_dir, args.name, shuffle=args.shuffle)

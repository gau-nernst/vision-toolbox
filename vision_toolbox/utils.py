import os

import torch


def torch_hub_download(url: str) -> str:
    save_path = os.path.join(torch.hub.get_dir(), os.path.basename(url))
    if not os.path.exists(save_path):
        torch.hub.download_url_to_file(url, save_path)
    return save_path

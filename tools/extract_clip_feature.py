import torch
from PIL import Image
import json
import numpy as np
import os
from tqdm import tqdm
import argparse
from tools.CLIP import clip


def get_embeddings(args):
    ann = json.load(open(args.annotation, 'r'))
    images = ann['images']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)
    for sample in tqdm(images):
        img_path = os.path.join(args.root_dir, sample['filename'])
        img_id = sample['id']
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features = image_features.squeeze().cpu().numpy()
        np.savez_compressed(os.path.join(args.save_path, f"{img_id}.npz"), feature=image_features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", default=r'data/mimic/mimic.json', type=str,
        help="path of json file: data.json")
    parser.add_argument("--save_path", default=r'mimic_clip16_att_512', type=str)
    args = parser.parse_args()
    get_embeddings(args)

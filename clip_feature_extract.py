import os
import gc
import json
import pickle
import argparse
import numpy as np
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
from PIL import Image
from tqdm import tqdm

DEFAULT = "$NO_USE$"
CLIP_MODEL_PATH = ""    # the path of CLIP model

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str)
parser.add_argument("--img_dir", type=str, default=DEFAULT)
parser.add_argument("--txt", action="store_true")
args = parser.parse_args()
data = args.data
data_dir = os.path.join("data", data, "candidates")


model = CLIPModel.from_pretrained(CLIP_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL_PATH)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH)


if args.txt:
    print(f"### Etract text features for {data}")
    with open(os.path.join(data_dir, "name_dict"), "r", encoding="utf-8") as fr:
        name_dict = json.load(fr)
        id2name = name_dict["ent"]
        eids, names = np.array(list(id2name.keys())), np.array(list(id2name.values()))
        bs = 2048
        idxs = list(range(len(eids)))
        batch_idxs = [idxs[i*bs:(i+1)*bs] for i in range((len(idxs)//bs) + 1)]
    vec = {}
    for batch in tqdm(batch_idxs, desc="TXT"):
        batch_eids, batch_names = eids[batch], list(names[batch])
        inputs = tokenizer(batch_names, padding=True, return_tensors="pt")
        features = model.get_text_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        features = features.cpu().detach().numpy()
        for i in range(len(batch_eids)):
            eid, feat = batch_eids[i], features[i]
            vec[int(eid)] = feat
    print(f"### {data} : TXT Features Extracted.")
    with open(os.path.join(data_dir, f"{data}_id_txt_feature_dict.pkl"), "wb") as fw:
        pickle.dump(vec, fw)
    gc.collect()
    print(f"### {data} : TXT Features Saved.")

if args.img_dir != DEFAULT:
    print(f"### Extract image features for {data}")
    with open(os.path.join(data_dir, "image_path"), "r", encoding="utf-8") as fr:
        image_path_dict = json.load(fr)
    vec = {}
    for eid, value in tqdm(list(image_path_dict.items())):
        img_path = os.path.join(args.img_dir, value["root"], str(eid), value["file"][0])
        img = Image.open(img_path)

        inputs = processor(images=img, return_tensors="pt")
        features = model.get_image_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)

        vec[int(eid)] = features.cpu().detach().numpy().reshape(-1)
        gc.collect()

    with open(os.path.join(data_dir, f"{data}_id_img_feature_dict.pkl"), "wb") as fw:
        pickle.dump(vec, fw)
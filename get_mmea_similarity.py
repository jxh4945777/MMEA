import os
import json
import pickle
import argparse

import numpy as np
from CSLS_ import *


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str)
args = parser.parse_args()
data = args.data
data_dir = os.path.join("data", data)
cand_dir = os.path.join(data_dir, "candidates")


with open(os.path.join(cand_dir, "name_dict"), "r", encoding="utf-8") as fr:
    name_dict = json.load(fr)
    id2name = name_dict["ent"]
ent_num = max([int(eid) for eid in id2name.keys()]) + 1

### load candidate entities
cand_files = []
for _, _, files in os.walk(cand_dir):
    for file in files:
        if file.startswith("cand"):
            cand_files.append(file)

pairs = set()
for cand_file in cand_files:
    with open(os.path.join(cand_dir, cand_file), "r", encoding="utf-8") as fr:
        candidates = json.load(fr)
        for e1, value in candidates.items():
            e1 = int(e1)
            for e2 in value["candidates"]:
                pair = f"{e1}-{e2}" if e1 < e2 else f"{e2}-{e1}"
                pairs.add(pair)

pairs = sorted(list(pairs))
l_ent, r_ent = set(), set()
for pair in pairs:
    e1, e2 = [int(e) for e in pair.split("-")]
    l_ent.add(e1)
    r_ent.add(e2)
l_ent, r_ent = list(l_ent), list(r_ent)
l_e2i, r_e2i = {e:i for i, e in enumerate(l_ent)}, {e:i for i, e in enumerate(r_ent)}

### load image and text features
def load_id_features(ent_num, feat_path):
    feat_dict = pickle.load(open(feat_path, "rb"))
    feat_np = np.array(list(feat_dict.values()))
    mean, std = np.mean(feat_np, axis=0), np.std(feat_np, axis=0)
    feat_embed = np.array([feat_dict[i] if i in feat_dict else np.random.normal(mean, std, mean.shape[0]) for i in range(ent_num)])
    feat_embed = feat_embed / np.linalg.norm(feat_embed, axis=-1, keepdims=True)
    return feat_embed, len(feat_dict)


img_feats, len_img_dict = load_id_features(ent_num, os.path.join(cand_dir, f"{data}_id_img_feature_dict.pkl"))
print(f"### {len_img_dict/ent_num:.2%} entities have images.")
txt_feats, len_txt_dict = load_id_features(ent_num, os.path.join(cand_dir, f"{data}_id_txt_feature_dict.pkl"))
print(f"### {len_txt_dict/ent_num:.2%} entities have name texts.")

### image-image similarity
embed1, embed2 = img_feats[l_ent], img_feats[r_ent]
i2i_sim = np.matmul(embed1, embed2.T)

### text-image similarity
embed1, embed2 = txt_feats[l_ent], img_feats[r_ent]
t2i_sim = np.matmul(embed1, embed2.T)

### image-text similarity
embed1, embed2 = img_feats[l_ent], txt_feats[r_ent]
i2t_sim = np.matmul(embed1, embed2.T)

### get multi modal similarity = max(i2i, t2i, i2t)
mmea_sim = np.max([i2i_sim, t2i_sim, i2t_sim], axis=0)

### save image-image and multi-modal similarity
pair_mmea_sims = {}
for pair in pairs:
    e1, e2 = [int(e) for e in pair.split("-")]
    id1, id2 = l_e2i[e1], r_e2i[e2]
    pair_mmea_sims[pair] = float(mmea_sim[id1][id2])
    
with open(os.path.join(cand_dir, "pair_mmea_sims"), "w", encoding="utf-8") as fw:
    json.dump(pair_mmea_sims, fw, ensure_ascii=False, indent=4)
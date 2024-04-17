import os
import json
import random
import pickle
import argparse

import numpy as np
from CSLS_ import *


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str)
parser.add_argument("--alpha", type=float, default=0.3)
parser.add_argument("--noise_ratio", type=float, default=0.0)
parser.add_argument("--no_img", action="store_true")
parser.add_argument("--test", action="store_true")
args = parser.parse_args()
data = args.data
no_img = args.no_img
noise = args.noise_ratio
data_dir = os.path.join("data", data)
cand_dir = os.path.join(data_dir, "candidates")
print(f"Data={data} , {f'Alpha={args.alpha} , ' if not args.test else ''}Noise Ratio={noise}")


with open(os.path.join(cand_dir, "name_dict"), "r", encoding="utf-8") as fr:
    name_dict = json.load(fr)
    id2name = name_dict["ent"]
ent_num = max([int(eid) for eid in id2name.keys()]) + 1

def load_id_features(ent_num, feat_path, noise_ratio=0.0):
    feat_dict = pickle.load(open(feat_path, "rb"))
    feat_np = np.array(list(feat_dict.values()))
    mean, std = np.mean(feat_np, axis=0), np.std(feat_np, axis=0)
    feat_embed = np.array([feat_dict[i] if i in feat_dict else np.random.normal(mean, std, mean.shape[0]) for i in range(ent_num)])
    feat_embed = feat_embed / np.linalg.norm(feat_embed, axis=-1, keepdims=True)
    ### add noise to embeddings
    if noise_ratio > 0:
        dim = len(feat_embed[0])
        sample_list = [i for i in range(dim)]
        bs = 1024
        for i in range((len(feat_embed)//bs) + 1):
            mask_id = random.sample(sample_list, int(dim * noise_ratio))
            feat_embed[i*bs:(i+1)*bs, mask_id] = 0
    return feat_embed, len(feat_dict)

### load ref_pairs
ref_pairs = []
with open(os.path.join(data_dir, "ref_pairs"), "r", encoding="utf-8") as fr:
    for line in fr.readlines():
        e1, e2 = [int(i) for i in line.split("\t")]
        ref_pairs.append([e1, e2])
ref_pairs = np.array(ref_pairs)
l_ent, r_ent = ref_pairs.T[0], ref_pairs.T[1]
print("### Read reference alignments, done.")

if not no_img:  
    ### load image and text features
    img_feats, len_img_dict = load_id_features(ent_num, os.path.join(cand_dir, f"{data}_id_img_feature_dict.pkl"), noise)
    print(f"### {len_img_dict/ent_num:.2%} entities have images.")
    txt_feats, len_txt_dict = load_id_features(ent_num, os.path.join(cand_dir, f"{data}_id_txt_feature_dict.pkl"), noise)
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

    ## get multi modal similarity = max(i2i, t2i, i2t)
    mmea_sim = np.max([i2i_sim, t2i_sim, i2t_sim], axis=0)

### get SimpleHHEA similarity
ent_feats, _ = load_id_features(ent_num, os.path.join(cand_dir, f"{data}_id_ent_feature_dict.pkl"), noise)
embed1, embed2 = ent_feats[l_ent], ent_feats[r_ent]
ent_sim = np.matmul(embed1, embed2.T)

### get final similarity
if args.test and not args.no_img:
    for i in range(11):
        alpha = i*0.1
        sim = (1 - alpha) * ent_sim + alpha * mmea_sim
        print(f"\nAlpha = {alpha:.1f}")
        eval_alignment_by_sim_mat(sim, [1, 5, 10], 16, 10, True)
        
else:
    if not args.no_img:
        alpha = args.alpha
        sim = (1 - alpha) * ent_sim + alpha * mmea_sim
        file_name = f"origin_cand_alpha_{int(alpha*10):02d}_noise_{int(noise*10):02d}"
    else:
        sim = ent_sim
        file_name = f"origin_cand_no_img_noise_{int(noise*10):02d}"
    print("### Evaluation result:")
    eval_alignment_by_sim_mat(sim, [1, 5, 10], 16, 10, True)

    ### generate candidates
    candidates = generate_candidates_by_sim_mat(sim, l_ent, r_ent, cand_num=20, csls=10, num_thread=16)
    print("### Generate candidates, done.\n")
    with open(os.path.join(cand_dir, file_name), "w", encoding="utf-8") as fw:
        json.dump(candidates, fw, ensure_ascii=False, indent=4)
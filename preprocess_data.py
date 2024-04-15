import os
import random
import argparse
import numpy as np
from tqdm import tqdm

from utils import *


# function for load data
def load_alignments(path):
    alignments = []
    with open(path, "r", encoding="utf-8") as fr:
        for line in tqdm(fr.readlines(), desc="Load Alignments"):
            if line:
                e1, e2 = [int(e) for e in line.strip().split("\t")]
                alignments.append(e1, e2)
    return np.array(alignments)

### generate name dict 
def load_name_dict(path):
    name_dict = {}
    with open(path, "r", encoding="utf-8") as fr:
        for line in tqdm(fr.readlines(), desc="Load Name"):
            if line:
                items = line.strip().split("\t")
                if items[0].isdigit():
                    idx, name = int(items[0]), items[1]
                else:
                    idx, name = int(items[1]), items[0]
                name_dict[idx] = name.split("/")[-1]
    return name_dict

def load_all_name_dict(data_dir):
    ### load ent name dict
    ent_name_dict = None
    if os.path.exists(os.path.join(data_dir, "ent_ids_1")):
        ent_name_dict_1 = load_name_dict(os.path.join(data_dir, "ent_ids_1"))
        ent_name_dict_2 = load_name_dict(os.path.join(data_dir, "ent_ids_2"))
        ent_name_dict = merge_dict(ent_name_dict_1, ent_name_dict_2)
    ent_ids = [int(idx) for idx in ent_name_dict.keys()]
    min_eid, max_eid = min(ent_ids), max(ent_ids)
    error_ent_ids = []
    if len(ent_ids) != max_eid - min_eid + 1:
        for eid in range(min_eid, max_eid+1):
            if eid not in ent_ids:
                error_ent_ids.append(eid)
    ### load rel name dict
    rel_name_dict = None
    if os.path.exists(os.path.join(data_dir, "rel_ids_1")):
        rel_name_dict_1 = load_name_dict(os.path.join(data_dir, "rel_ids_1"))
        rel_name_dict_2 = load_name_dict(os.path.join(data_dir, "rel_ids_2"))
        rel_name_dict = merge_dict(rel_name_dict_1, rel_name_dict_2)
    rel_ids = [int(idx) for idx in rel_name_dict.keys()]
    min_rid, max_rid = min(rel_ids), max(rel_ids)
    error_rel_ids = []
    if len(rel_ids) != max_rid - min_rid + 1:
        for rid in range(min_rid, max_rid+1):
            if rid not in rel_ids:
                error_rel_ids.append(rid)
    ### load time name dict
    time_name_dict = None
    if os.path.exists(os.path.join(data_dir, "time_id")):
        time_name_dict = {}
        with open(os.path.join(data_dir, "time_id"), "r", encoding="utf-8") as fr:
            for line in tqdm(fr.readlines(), desc="Load Time ID"):
                if line:
                    idx, time = line.strip().split("\t")
                    idx = int(idx)
                    if time == '' or time == '-400000':
                        time = '~'
                    time_name_dict[idx] = time
    else:
        time_size = 0
        for i in range(2):
            with open(os.path.join(data_dir, f"triples_{i+1}"), "r", encoding="utf-8") as fr:
                for line in fr.readlines():
                    if line:
                        items = line.strip().split("\t")
                        if len(items) == 4:
                            tau = items[3]
                            time_size = max(tau + 1, time_size)
                        elif len(items) == 5:
                            ts, te = items[3:]
                            time_size = max(ts + 1, te + 1, time_size)
                        else:
                            break
        if time_size > 0:
            time_name_dict = {i:i for i in range(time_size)}
    
    return {"ent": ent_name_dict, "rel": rel_name_dict, "time": time_name_dict}, error_ent_ids, error_rel_ids


### generate neighbors
def load_neighbors(path, error_ent_ids, error_rel_ids, neighbor_num=25):
    neighbors = {}
    with open(path, "r", encoding="utf-8") as fr:
        for line in tqdm(fr.readlines(), desc="Load Neighbors"):
            if line:
                items = [int(item) for item in line.strip().split("\t")]
                h, r, t = items[:3]
                if h in error_ent_ids or t in error_ent_ids or r in error_rel_ids:
                    continue
                if len(items) == 3:
                    neighbor = (h, r, t)
                elif len(items) == 4:
                    tau = items[3]
                    neighbor = (h, r, t, tau, tau)
                elif len(items) == 5:
                    ts, te = items[3:]
                    neighbor = (h, r, t, ts, te)
                else:
                    raise Exception("Knowledge Tuples's length is not 3, 4, or 5.")
                if h not in neighbors:
                    neighbors[h] = []
                if t not in neighbors:
                    neighbors[t] = []
                neighbors[h].append(neighbor)
                neighbors[t].append(neighbor)
    for eid, neigh in neighbors.items():
        if len(neigh) > neighbor_num:
            random.shuffle(neigh)
            neighbors[eid] = neigh[:neighbor_num]
    return neighbors





if __name__ == "__main__":
    ### arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="icews_wiki")
    parser.add_argument("--neighbor_num", type=int, default=25)
    args = parser.parse_args()
    data_dir = os.path.join("data", args.data)
    
    ### generate name dict
    name_dict, error_ent_ids, error_rel_ids = load_all_name_dict(data_dir)
    ### generate neighbors
    neighbors1 = load_neighbors(os.path.join(data_dir, "triples_1"), error_ent_ids, error_rel_ids, neighbor_num=args.neighbor_num)
    neighbors2 = load_neighbors(os.path.join(data_dir, "triples_2"), error_ent_ids, error_rel_ids, neighbor_num=args.neighbor_num)
    neighbors = merge_dict(neighbors1, neighbors2)

    ### save dict
    cand_dir = os.path.join(data_dir, "candidates")
    if not os.path.exists(cand_dir):
        os.makedirs(cand_dir)
    dump_json(os.path.join(cand_dir, "name_dict"), name_dict)
    dump_json(os.path.join(cand_dir, "neighbors"), neighbors)
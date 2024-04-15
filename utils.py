import os
import json


### json file IO
def load_json(path):
    with open(path, "r", encoding="utf-8") as fr:
        datas = load_json(fr)
    return datas

def dump_json(path, obj):
    with open(path, "w", encoding="utf-8") as fw:
        json.dump(obj, fw, ensure_ascii=False, indent=4)


### tool functions
def merge_dict(dic1, dic2):
    return {**dic1, **dic2}

def transform_idx_to_int(dic:dict):
    return {int(idx):data for idx, data in dic.items()}

def transform_time(seconds:int):
    h, m, s = 0, 0, 0
    h = seconds // 3600
    seconds -= h * 3600
    m = seconds // 60
    seconds -= m * 60
    s = seconds
    return h, m, s

def count_ranks(ranks):
    ###############
    ### Count the distribution of numbers of 0-1, 1-10, 10-20, >20 ranks
    ###############
    count = [0, 0, 0, 0]
    span = ['[ 0,  1)', '[ 1, 10)', '[10, 20)', '[20, --)']
    for r in ranks:
        if r == 0:
            count[0] += 1
        else:
            if r < 10:
                count[1] += 1
            else:
                if r < 20:
                    count[2] += 1
                else:
                    count[3] += 1
    total = len(ranks)
    print('Count of Ranks: ')
    for i in range(len(count)):
        print(f'  {span[i]} : {count[i]} , {count[i] / total:.2%}')


### evaluate
def evaluate_alignment(ranks, hit_k=[1, 5, 10]):
    ###############
    ### Input: ranks of all entities
    ### Output: Hits@K, MRR
    ###############
    hits = [0] * len(hit_k)
    mrr = 0
    for r in ranks:
        mrr += 1 / (r + 1)
        for j in range(len(hit_k)):
            if r < hit_k[j]:
                hits[j] += 1
    total_num = len(ranks)
    mrr /= total_num
    hits = [round(hits[i] / total_num, 4) for i in range(len(hit_k))]
    
    return hits, mrr


### generate entity neighbors information
class NeighborGenerator(object):
    def __init__(self, data, data_dir='../data', cand_file='cand', use_time=True, use_desc=True, use_name=True, use_img=True):
        self.use_time = use_time
        self.use_desc = use_desc
        self.use_name = use_name
        self.use_img  = use_img
        self.path = os.path.join(data_dir, data, 'candidates')
        self.ref, self.rank, self.cand, self.cand_score = self.load_candidates(cand_file=cand_file)
        self.neighbors = self.load_neighbors()
        if use_time:
            self.ent_name, self.rel_name, self.time_dict = self.load_name_dict()
        else:
            self.ent_name, self.rel_name = self.load_name_dict()
        if use_desc:
            self.description = self.load_description()
        if use_img:
            self.pair_mmea_sim = self.load_pair_mmea_sim()

        self.entities = sorted([int(e) for e in self.cand.keys()])

    # initialize, load data
    def load_candidates(self, cand_file='cand'):
        with open(os.path.join(self.path, cand_file), 'r', encoding='utf-8') as fr:
            origin_cand = json.load(fr)
        ref, rank, cand, cand_score = {}, {}, {}, {}
        for eid, data in origin_cand.items():
            eid = int(eid)
            ref[eid] = data['ref']
            rank[eid] = data['ground_rank']         # ranks from method based on embeddings
            cand[eid] = data['candidates']          # {ent_id: [cand_id_1, cand_id_2, ..., cand_id_20], ...}
            cand_score[eid] = data['cand_sims']     # similarity scores from method based on embeddings
        return ref, rank, cand, cand_score
    def load_neighbors(self):
        with open(os.path.join(self.path, 'neighbors'), 'r', encoding='utf-8') as fr:
            neighbors = json.load(fr)           # {ent_id: [ [h1, r1, t1, ts1, te1], [h2, r2, t2, ts2, te2], ... ], ...}
        return transform_idx_to_int(neighbors)
    def load_name_dict(self):
        with open(os.path.join(self.path, 'name_dict'), 'r', encoding='utf-8') as fr:
            name_dict = json.load(fr)
        ent_name = transform_idx_to_int(name_dict['ent'])
        rel_name = transform_idx_to_int(name_dict['rel'])
        if self.use_time:
            time_dict = transform_idx_to_int(name_dict['time'])
            return ent_name, rel_name, time_dict    # {id : name, ...}
        else:
            return ent_name, rel_name
    def load_description(self):
        with open(os.path.join(self.path, "description"), 'r', encoding='utf-8') as fr:
            origin_desc = json.load(fr)
        desc = {int(eid):d["desc"] for eid, d in origin_desc.items()}   # {ent_id: desc, ...}
        return desc
    def load_pair_mmea_sim(self):
        with open(os.path.join(self.path, "pair_mmea_sims"), "r", encoding="utf-8") as fr:
            pair_mmea_sim = json.load(fr)
        return pair_mmea_sim

    # API
    def get_all_entities(self):
        all_ent = set()
        for eid, cand in self.cand.items():
            all_ent.update([eid] + cand)
        return sorted(list(all_ent))
    def get_entities(self):
        return self.entities
    def get_ref_ent(self, ent_id:int):
        return self.ref[ent_id]
    def get_base_rank(self, ent_id:int):
        return self.rank[ent_id]
    def get_neighbors(self, ent_id:int, neigh_num=0):
        ###############
        ### Output: {'ent_id': ent_id, 'name': 'XXXX', 'neighbors': {'(h1, r1, XXXX, temp1_s, temp1_e)', ..., '(XXXX, r2, t2, temp2_s, temp2_e)'}
        ###############
        if ent_id in self.neighbors:
            neigh = self.neighbors[ent_id]
            if len(neigh) > 0:
                neigh_num = len(neigh) if neigh_num == 0 or neigh_num > len(neigh) else neigh_num
                neigh = neigh[:neigh_num]
                new_neigh = []
                if len(neigh[0]) == 3:
                    for h, r, t in neigh:
                        if self.use_name:
                            new_neigh.append((self.ent_name[h], self.rel_name[r], self.ent_name[t]))
                        else:
                            # replace entity name with entity id, when forbidding using name
                            new_neigh.append((f"'{h}'", self.rel_name[r], f"'{t}'"))
                else:
                    for h, r, t, ts, te in neigh:
                        if self.use_name:
                            new_neigh.append((self.ent_name[h], self.rel_name[r], self.ent_name[t], self.time_dict[ts], self.time_dict[te]))
                        else:
                            new_neigh.append((f"'{h}'", self.rel_name[r], f"'{t}'", self.time_dict[ts], self.time_dict[te]))
                neigh = new_neigh
        else:
            neigh = []

        return_dict = {"ent_id": ent_id, "name": self.ent_name[ent_id], "neighbors": neigh}
        if self.use_desc:
            desc = self.description[ent_id] if ent_id in self.description else self.ent_name[ent_id]
            if not self.use_img:
                desc = desc.split(". ")[-1]
            return_dict["desc"] = desc
        return return_dict

    def get_candidates(self, ent_id:int, neigh_num=0):
        ###############
        ### Output: [{'ent_id':cand_ent_1_id, 'name': 'XXXX', 'score': XXXX, 'neighbors': {...}}}, {'ent_id': cand_ent_2_id, ...}, ..., {'ent_id': cand_ent_20_id, ...}]
        ###############
        cand = []
        for score, cand_id in zip(self.cand_score[ent_id], self.cand[ent_id]):
            cand_ent = self.get_neighbors(cand_id, neigh_num)
            cand_ent["hhea_sim"] = round(score, 3)
            if self.use_img:
                pair = f"{ent_id}-{cand_id}" if ent_id < cand_id else f"{cand_id}-{ent_id}"
                cand_ent["mmea_sim"] = self.pair_mmea_sim[pair] if pair in self.pair_mmea_sim else 0.0
            cand.append(cand_ent)
        return cand
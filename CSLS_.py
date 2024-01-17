import gc
import multiprocessing
import numpy as np
import time


g = 1000000000

def div_list(ls, n):
    ls_len = len(ls)
    if n <= 0 or 0 == ls_len:
        return [ls]
    if n > ls_len:
        return [ls]
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = ls_len // n
        k = ls_len % n
        ls_num = [j] * n
        for i in range(k):
            ls_num[i] += 1
        ls_return = []
        ind = 0
        for i in range(n):
            ls_return.append(ls[ind : ind+ls_num[i]])
            ind += ls_num[i]
        return ls_return

def cal_rank_by_sim_mat(task, sim, top_k, accurate):
    mean = 0
    mrr = 0
    num = [0 for k in top_k]
    prec_set = set()
    for i in range(len(task)):
        ref = task[i]
        if accurate:
            rank = (-sim[i, :]).argsort()
        else:
            rank = np.argpartition(-sim[i, :], np.array(top_k) - 1)
        prec_set.add((ref, rank[0]))
        assert ref in rank
        rank_index = np.where(rank == ref)[0][0]
        mean += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                num[j] += 1
    return mean, mrr, num, prec_set


def eval_alignment_by_sim_mat(sim_mat, top_k, num_thread, csls=10, accurate=True):
    t = time.time()
    sim_mat = sim_handler_by_sim_mat(sim_mat, csls, num_thread)
    ref_num = sim_mat.shape[0]
    t_num = [0 for _ in top_k]
    t_mean, t_mrr, t_prec_set = 0, 0, set()
    tasks = div_list(np.array(range(ref_num)), num_thread)
    pool = multiprocessing.Pool(processes=len(tasks))
    reses = []
    for task in tasks:
        reses.append(pool.apply_async(cal_rank_by_sim_mat, (task, sim_mat[task, :], top_k, accurate)))
    pool.close()
    pool.join()

    for res in reses:
        mean, mrr, num, prec_set = res.get()
        t_mean += mean
        t_mrr += mrr
        t_num += np.array(num)
        t_prec_set |= prec_set
    assert len(t_prec_set) == ref_num
    acc = np.array(t_num) / ref_num * 100
    for i in range(len(acc)):
        acc[i] = round(acc[i], 2)
    t_mean /= ref_num
    t_mrr /= ref_num
    if accurate:
        print(f"accurate results: hits@{top_k} = {acc}, mr = {t_mean:.3f}, mrr = {t_mrr:.3f}, time = {time.time()-t:.3f} s")
    else:
        print(f"hits@{top_k} = {acc}, time = {time.time()-t:.3f} s")
    del sim_mat
    gc.collect()
    return t_prec_set, acc, t_mrr


def cal_csls_sim(sim_mat, k):
    sorted_mat = -np.partition(-sim_mat, k + 1, axis=1)  # -np.sort(-sim_mat1)
    nearest_k = sorted_mat[:, 0:k]
    sim_values = np.mean(nearest_k, axis=1)
    return sim_values


def CSLS_sim(sim_mat1, k, nums_threads):
    tasks = div_list(np.array(range(sim_mat1.shape[0])), nums_threads)
    pool = multiprocessing.Pool(processes=len(tasks))
    reses = list()
    for task in tasks:
        reses.append(pool.apply_async(cal_csls_sim, (sim_mat1[task, :], k)))
    pool.close()
    pool.join()
    sim_values = None
    for res in reses:
        val = res.get()
        if sim_values is None:
            sim_values = val
        else:
            sim_values = np.append(sim_values, val)
    assert sim_values.shape[0] == sim_mat1.shape[0]
    return sim_values


def sim_handler_by_sim_mat(sim_mat, csls, num_thread):
    if csls <= 0:
        print("csls = 0")
        return sim_mat
    csls1 = CSLS_sim(sim_mat, csls, num_thread)
    csls2 = CSLS_sim(sim_mat.T, csls, num_thread)
    csls_sim_mat = 2 * sim_mat.T - csls1
    csls_sim_mat = csls_sim_mat.T - csls2
    del sim_mat
    gc.collect()
    return csls_sim_mat


def merge_dic(dic1, dic2):
    return {**dic1, **dic2}


def generate_candidates_by_sim_mat(sim_mat, ent1, ent2, cand_num, csls=10, num_thread=16):
    '''
    Output: {ent_id_1:{'ground_rank': rank, 'candidates': candidates}}
        - ground_rank : the rank of entity2 in entity_list2 which matches ent_id_1, according to reference pairs
        - candidates  : top K entities which matches ent_id_1, according to sim_mat = dot(embed1, embed2)
    '''
    ent_idx_1 = {e:i for i, e in enumerate(ent1)}
    ent_frags = div_list(np.array(ent1), num_thread)
    sim_mat = sim_handler_by_sim_mat(sim_mat, csls, num_thread)
    ref_num = sim_mat.shape[0]
    tasks = div_list(np.array(range(ref_num)), num_thread)

    pool = multiprocessing.Pool(processes=len(tasks))
    results = []
    for i, task in enumerate(tasks):
        results.append(pool.apply_async(find_candidates_by_sim_mat, args=(ent_frags[i], ent_idx_1, sim_mat[task, :], np.array(ent2), cand_num)))
    pool.close()
    pool.join()

    dic = {}
    for res in results:
        dic = merge_dic(dic, res.get())
    del results
    gc.collect()
    return dic


def find_candidates_by_sim_mat(frags, ent_idx, sim, entity_list2, k):
    dic = {}
    for i in range(sim.shape[0]):
        ref = ent_idx[frags[i]]
        rank = (-sim[i, :]).argsort()
        rank_index = np.where(rank == ref)[0][0]
        cand_index = np.argpartition(-sim[i, :], k)[:k]
        candidates = entity_list2[cand_index].tolist()
        cand_sims = [float(s) for s in sim[i, cand_index]]
        min_s, max_s = min(cand_sims), max(cand_sims)
        cand_sims = [(s - min_s) / (max_s - min_s) for s in cand_sims]
        
        sorted_cand = sorted([(candidates[j], cand_sims[j]) for j in range(k)], key=lambda x: x[1], reverse=True)
        candidates = [c_idx for c_idx, _ in sorted_cand]
        cand_sims  = [c_sim for _, c_sim in sorted_cand]

        dic[int(frags[i])] = {'ref': int(entity_list2[ref]), 'ground_rank':int(rank_index), 'candidates':candidates, 'cand_sims':cand_sims}
    return dic

import os
import base64
import json
import argparse
import openai
from tqdm import tqdm


MAX = 100000000
LOG_PRINT = False

### openai==1.7.0
API_KEY = ""        ### your openai key
engine = "gpt-4-vision-preview"
client = openai.OpenAI(api_key=API_KEY)


def try_get_response(prompt, max_tokens=400, max_try_num=1):
    try_num = 0
    flag = True
    response = None
    while flag:
        try:
            response = client.chat.completions.create(
                model=engine,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.2
            )
            flag = False
        except openai.OpenAIError as e:
            print(f"### TRY ERROR : {e}")
            try_num += 1
            if try_num >= max_try_num:
                break
    return response, (not flag)


def generate_prompt(entity):
    tuples = [f"[{', '.join(n)}]" for n in entity["neigh"]]
    prompt = f"Given following informations: 1.[Entity] {entity['name']}; 2.[Knowledge Tuples] = [{', '.join(tuples)}]. Please answer the question: \n"
    # description
    prompt += f"[Question]: What is {entity['name']}? Please give a two-sentence brief introduction. The first sentence is to simply describe what is {entity['name']}. The second sentence is to give additional description about {entity['name']} based on [Knowledge Tuples] and YOUR OWN KNOWLEDGE. Give [answer] strictly in format: [ENT] is ......\n[answer]: "
    return prompt


def read_entities(data_dir, neigh_num=25):
    neigh_num = neigh_num if neigh_num > 0 else MAX
    # name info and neighbors info
    with open(os.path.join(data_dir, "candidates", "error_eids"), "r", encoding="utf-8") as fr:
        ent_ids = list(set(json.load(fr)))
    with open(os.path.join(data_dir, "candidates", "name_dict"), "r", encoding="utf-8") as fr:
        name_dict = json.load(fr)
        ent_name_dict, rel_name_dict, time_dict = name_dict["ent"], name_dict["rel"], name_dict["time"]
    with open(os.path.join(data_dir, "candidates", "neighbors"), "r", encoding="utf-8") as fr:
        neighbors = json.load(fr)
    # get entities
    entities = []
    for eid in ent_ids:
        ent = {"id": eid, "name": ent_name_dict[eid]}
        neigh = []
        if eid in neighbors:
            if time_dict is not None:
                for n in neighbors[eid][:neigh_num]:
                    h, r, t, ts, te = [str(i) for i in n]
                    neigh.append((ent_name_dict[h], rel_name_dict[r], ent_name_dict[t], time_dict[ts], time_dict[te]))
            else:
                for n in neighbors[eid][:neigh_num]:
                    h, r, t = [str(i) for i in n]
                    neigh.append((ent_name_dict[h], rel_name_dict[r], ent_name_dict[t]))
            ent["neigh"] = neigh
            entities.append(ent)
    return entities


def process_response(res:str, ent_name:str):
    if res == "[ERROR]":
        return "[ERROR]"
    if "[ENT] is " in res:
        desc = res.replace("[ENT]", ent_name)
    else:
        desc = "[ERROR]"
    return desc


def get_entity_description(data_dir, neigh_num=25, max_tokens=400):
    entities = read_entities(data_dir, neigh_num)
    print(f"### Total Num of Error Entities: {len(entities)}")
    
    ent_desc_path = os.path.join(data_dir, "candidates", "description")
    with open(ent_desc_path, "r", encoding="utf-8") as fr:
        ent_desc = json.load(fr)    # {eid:{"name": XXXX, "desc": XXXX}}

    query_num = 0
    for ent in tqdm(entities, desc=f"Entities"):
        if ent["id"] in ent_desc:
            continue
        query_num += 1
        # prompt
        prompt = generate_prompt(ent)
        # query GPT4
        res, get_res = try_get_response(prompt, max_tokens)
        response = res.choices[0].message.content if get_res else "[ERROR]"
        desc = process_response(response, ent["name"])
        if get_res and desc != "[ERROR]":
            ent_desc[ent["id"]] = {"name": ent["name"], "desc": desc}
        
        # temporily save
        if query_num % 200 == 0:
            print(f"### Temporarily Save {query_num} / {len(entities)}")
            with open(ent_desc_path, "w", encoding="utf-8") as fw:
                json.dump(ent_desc, fw, ensure_ascii=False, indent=4)

        if LOG_PRINT:
            print("\n### INFO :")
            print(ent["id"], ent["name"])
            print("\n### PROMPT :")
            print(prompt)
            print("\n### RESPONSE :")
            print(response)
            print("\n### DESC :")
            print(desc)
            print("\n")

    with open(ent_desc_path, "w", encoding="utf-8") as fw:
        json.dump(ent_desc, fw, ensure_ascii=False, indent=4)

    return ent_desc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--neigh", type=int, default=25)
    parser.add_argument("--max_tokens", type=int, default=400)
    parser.add_argument("--log_print", action="store_true")
    args = parser.parse_args()
    data_dir = os.path.join("data", args.data)
    LOG_PRINT = args.log_print

    ### get description and additional tuples from images
    get_entity_description(args.data_dir, args.neigh, args.max_tokens)
    

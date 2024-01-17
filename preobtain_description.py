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


def image_to_url(image_path:str):
    image_format = "png" if image_path.endswith(".png") else "jpeg"
    with open(image_path, "rb") as fr:
        base64_image = base64.b64encode(fr.read()).decode("utf-8")
    return f"data:image/{image_format};base64,{base64_image}"


def try_get_response(prompt, images, max_tokens=400, max_try_num=1):
    try_num = 0
    flag = True
    response = None
    image_content = [{"type": "image_url", "image_url": {"url": iu}} for iu in images]
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": prompt}] + image_content
    }]
    while flag:
        try:
            response = client.chat.completions.create(
                model=engine,
                messages=messages,
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
    prompt = f"Given following informations: 1.[Entity] {entity['name']}; 2.[Knowledge Tuples] = [{', '.join(tuples)}]; 3.IMAGES related to [Entity]. Please answer the question: \n"
    prompt += f"[Question]: What is {entity['name']}? Please give a two-sentence brief introduction. The first sentence is to simply describe what is {entity['name']}, combining the identity features in IMAGES. The second sentence is to give additional description about {entity['name']} based on IMAGES, [Knowledge Tuples] and YOUR OWN KNOWLEDGE. Give [answer] strictly in format: [Entity] is ......\n[answer]: "
    return prompt


def read_entities(data_dir, img_dir, cand_file, neigh_num=25):
    neigh_num = neigh_num if neigh_num > 0 else MAX
    # name info and neighbors info
    with open(os.path.join(data_dir, "candidates", cand_file), "r", encoding="utf-8") as fr:
        cand = json.load(fr)
        ent_ids = set()
        for eid, value in cand.items():
            ent_ids.add(eid)
            for i in value["candidates"]:
                ent_ids.add(str(i))
        ent_ids = sorted(list(ent_ids))
    with open(os.path.join(data_dir, "candidates", "name_dict"), "r", encoding="utf-8") as fr:
        name_dict = json.load(fr)
        ent_name_dict, rel_name_dict, time_dict = name_dict["ent"], name_dict["rel"], name_dict["time"]
    with open(os.path.join(data_dir, "candidates", "neighbors"), "r", encoding="utf-8") as fr:
        neighbors = json.load(fr)
    # images
    with open(os.path.join(data_dir, "candidates", "image_path"), "r", encoding="utf-8") as fr:
        image_path = json.load(fr)
    # get entities
    entities, error_eids = [], set()
    for eid in ent_ids:
        if eid not in ent_name_dict or eid not in neighbors or eid not in image_path:
            error_eids.add(eid)
            continue
        img_path = []
        for f in image_path[eid]["file"]:
            img_path.append(os.path.join(img_dir, image_path[eid]["root"], str(eid), f))
        ent = {"id": eid, "name": ent_name_dict[eid], "img_path": img_path}
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
    return entities, error_eids


def process_response(res:str, ent_name:str):
    if res == "[ERROR]":
        return "[ERROR]"
    if "[Entity] is " in res:
        desc = res.replace("[Entity]", ent_name)
    else:
        desc = "[ERROR]"
    return desc


def get_entity_description(data_dir, img_dir, cand_file, img_num=1, neigh_num=10, max_tokens=400):
    img_num = img_num if img_num > 0 else MAX

    entities, error_eids = read_entities(data_dir, img_dir, cand_file, neigh_num)
    print(len(entities), len(error_eids))
    print(f"### Total Num of Entities: {len(entities)}")
    
    ent_desc = {}
    ent_desc_path = os.path.join(data_dir, "candidates", "description")
    error_eid_path = os.path.join(data_dir, "candidates", "error_eids")
    if os.path.exists(ent_desc_path):
        with open(ent_desc_path, "r", encoding="utf-8") as fr:
            ent_desc = json.load(fr)    # {eid:{"name": XXXX, "desc": XXXX}}
    if os.path.exists(error_eid_path):
        with open(error_eid_path, "r", encoding="utf-8") as fr:
            error_eids = error_eids.union(set(json.load(fr)))  # [eid, eid, ...]
    
    query_num = 0
    for ent in tqdm(entities, desc=f"Entities"):
        if ent["id"] in ent_desc or ent["id"] in error_eids:
            continue
        query_num += 1
        # images
        images = [image_to_url(img_path) for img_path in ent["img_path"][:img_num]]
        # prompt
        prompt = generate_prompt(ent)
        # query GPT-4-Vision
        res, get_res = try_get_response(prompt, images, max_tokens)
        response = res.choices[0].message.content if get_res else "[ERROR]"
        desc = process_response(response, ent["name"])
        if get_res and desc != "[ERROR]":
            ent_desc[ent["id"]] = {"name": ent["name"], "desc": desc}
        else:
            error_eids.add(ent["id"])
        
        # temporily save
        if query_num % 200 == 0:
            print(f"### Temporarily Save {query_num} / {len(entities)}")
            with open(ent_desc_path, "w", encoding="utf-8") as fw:
                json.dump(ent_desc, fw, ensure_ascii=False, indent=4)
            with open(error_eid_path, "w", encoding="utf-8") as fw:
                json.dump(list(error_eids), fw, ensure_ascii=False, indent=4)

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
    with open(error_eid_path, "w", encoding="utf-8") as fw:
        json.dump(list(error_eids), fw, ensure_ascii=False, indent=4)

    return ent_desc, list(error_eids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--img_dir", type=str)
    parser.add_argument("--cand_file", type=str, default="cand")
    parser.add_argument("--img", type=int, default=1)
    parser.add_argument("--neigh", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=400)
    parser.add_argument("--log_print", action="store_true")
    args = parser.parse_args()
    data_dir = os.path.join("data", args.data)
    LOG_PRINT = args.log_print

    ### get description and additional tuples from images
    get_entity_description(data_dir, args.img_dir, args.cand_file, args.img, args.neigh, args.max_tokens)

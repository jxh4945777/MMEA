# MM-ChatAlign

### Dataset

Both the DBP15K and FB15K-YAGO15K are come from [MCLEA](https://github.com/lzxlin/MCLEA) reposity. You can download them directly in their pages.



### How to Run

The model runs in 4 steps:

#### 1. Collect Candidates

Firstly, save the name dictionary, the knowledge tuples and the file path of images under `data/{DATASET}/candidates`. As for the detailed file format of data, please see `name_dict`, `neighbors` and `image_path` under [data/example/candidates](data/example/candidates).

Then, use EA methods based on emebddings to obtain the embeddings of entities. Save these embeddings as a dictionary like `{ent_id : embedding, ...}`, in which `ent_id` is entity ID in format `int` and `embedding` is entity embedding in format `numpy.ndarray`. Save the ID-to-embedding dictionary in file `{DATASET}_id_ent_feature_dict.pkl` under `data/{DATASET}/candidates`.

Next, use CLIP to obtain the image features and text (entity name) features of entities, use:

```
python clip_feature_extract.py --data DATASET --img_dir IMG_DIR --txt
```

Finally, collect candidates, use:

```
python get_candidates.py --data DATASET
```

The candidates will be saved in `data/{DATASET}/candidates/all_cand`. As for the detailed data format, please see `cand` under [data/example/candidates](data/example/candidates).

You can sample part of entities from `all_cand` and save them in `cand_XXXX`

#### 2. Pre-obtain the cross-modal similarity of entity pairs

For efficiency, we need to pre-obtain the cross-modal similarity of entity pairs, use:

```
python get_mmea_similarity.py --data DATASET
```

#### 3. Pre-obtain the entity descrptions

For efficiency, we need to pre-obtain the entity descriptions using GPT-4-Vision based on images, tuples in knowledge graph and the inherent knowldege from LLM itself.

To  pre-obtain the entity descriptions, use:

```bash
python preobtain_description.py \
	--data DATASET \
    --img_dir IMG_DIR \
	--cand_file cand \
	--img 1 \
	--neigh 10 \
	--max_tokens 400
```

If there exits some entities that cannot get description by GPT-4-Vision, please use following code to get their description:

```bash
python preobtain_error_ent_description.py \
	--data DATASET \
	--neigh 25 \
	--max_tokens 400
```

#### 4. Run MM-ChatAlign

To run MM-ChatAlign, use:

```bash
python main_MMChatAlign.py \
	--LLM llama	\
	--data DATASET \
	--cand_file cand \
	--result_name RESULT_NAME
	--neigh 5
```

use `--log_print` to output the prompt and response of LLM

use `--save_step X` to save result for each `X` entities

use `--new_result` to ignore the existed results



### Prompts

#### 1. Getting description

```
Given following informations: 1.[Entity] {{ Name }}; 2.[Knowledge Tuples] = {{ Tuples }}; 3.IMAGES related to [Entity]. Please answer the question: 

[Question]: What is {{ Name }}? Please give a two-sentence brief introduction. The first sentence is to simply describe what is {{ Name }}, combining the identity features in IMAGES. The second sentence is to give additional description about {{ Name }} based on IMAGES, [Knowledge Tuples] and YOUR OWN KNOWLEDGE. Give [answer] strictly in format: [Entity] is ......

[answer]:
```

#### 2. MMKG-Code translation

```
A Knowledge Graph Entity is defined as follows: 

class Entity: 
	def __init__(self, name, id, tuples=[], images=[]):
		self.entity_name = name
		self.entity_id = id
		self.tuples = tuples
		self.images = images
	def get_neighbors(self):
		neighbors = set()
		for head_entity, _, tail_entity, _, _ in self.tuples
			if head_entity == self.entity_name:
				neighbors.add(tail_entity)
			else:
				neighbors.add(head_entity)
		return list(neighbors)
	def get_relation_information(self):
		relation_info = []
		for _, relation, _, _, _ in self.tuples:
			relation_info.append(relation)
		return relation_info
	def get_time_information(self):
		time_info = []
		for _, _, _, start_time, ent_time in self.tuples:
			time_info.append((start_time, ent_time))
		return time_info
	def get_description(self, LLM):
		description = LLM(self.entity_name, self.tuples, self.images)
		return description

You are a helpful assistant, helping me align or match entities of knowledge graphs according to name information (self.entity_name), description information (get_description), structure information (self.tuples, get_neighbors(), get_relation_information()), time information (get_time_information()), IMAGES, YOUR OWN KNOWLEDGE.

Your reasoning process for entity alignment should strictly follow this case step by step:

{{ CASE }}

[Output Format]: [NAME SIMILARITY] = A out of 5, [PROBABILITY OF DESCRIPTION POINTING SAME ENTITY] = B out of 5, [STRUCTURE SIMILARITY] = C out of 5, [TIME SIMILARITY] = D out of 5, [IMAGE SIMILARITY] = E out of 5. NOTICE, A,B,C,D,E are in range [1, 2, 3, 4, 5], which respectively means [VERY LOW], [LOW], [MEDIUM], [HIGH], [VERY HIGH]. NOTICE, you MUST strictly output like [Output Format].
```

#### 3. Reasoning

```
Now given [Main Entity] l_e = Entity( {{ Name and Tuples }} ), and [Candidate Entity] r_e = Entity( {{ Name and Tuples }} ),

- Do [Main Entity] and [Candidate Entity] align or match? Think of the answer STEP BY STEP with name, description, structure, time, YOUR OWN KNOWLEDGE:

Step 1, think of [NAME SIMILARITY] = A out of 5, using self.entity_name. 

Step 2, think of [PROBABILITY OF DESCRIPTION POINTING SAME ENTITY] = B out of 5, using get_descripton() and YOUR OWN KNOWLEDGE.

Step 3, think of [STRUCTURE SIMILARITY] = C out of 5, using self.tuples, get_neighbors() and get_relation_information().

Step 4, think of [TIME SIMILARITY] = D out of 5, using get_time_information().

Step 5, think of [IMAGE SIMILARITY] = E out of 5, using self.images.
```

#### 4. Rethinking

````
Now given the following entity alignments: 
[Main Entity]: {{ Name }} -> {{ Aligned Candidate List }}

Please answer the question: Do these entity alignments are satisfactory enough ([YES] or [NO])?

Answer [YES] if they are relatively satisfactory, which means the alignment score of the top-ranked candidate meet the threshold, and is far higher than others; otherwise, answer [NO] which means we must search other candidate entities to match with [Main Entity].

NOTICE, Just answer [YES] or [NO]. Your reasoning process should follow [EXAMPLE]s:
{{ Examples }}

Just directly answer [YES] or [NO], don't give other text.
````

# MM-ChatAlign

### Dataset

Both the DBP15K and FB15K-YAGO15K are come from [MCLEA](https://github.com/lzxlin/MCLEA) reposity. You can download them directly in their pages.



### How to Run

The model runs in three steps:

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


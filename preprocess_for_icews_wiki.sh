data="icews_wiki"
neigh_num=25
image_dir="$data/images"

# generate name_dict and neighbors
python preprocess_data.py --data $data --neighbor_num $neigh_num
# get image and text features by CLIP
python clip_feature_extract.py --data $data --img_dir $image_dir --txt
# generate candidate entities
python get_candidates.py --data $data
# get cross-modal similarity of entity pairs
python get_mmea_similarity.py --data $data
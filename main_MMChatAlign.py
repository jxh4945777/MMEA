import re
import time
import openai
import argparse
from tqdm import tqdm
from utils import *


### openai settings
API_KEY = ""	# your api key
engine = ""
client = openai.OpenAI(api_key=API_KEY)

### hyper-parameters
history_len = 3		# length of chat history
threshold = 0.5
img_sim_w = 0.1

LOG_PRINT = False
USE_PREVIOUS_RESULT = False
save_step = 0
save_result_path = ""

USE_CODE = True
USE_DESC = True
USE_NAME = True
USE_STRUCT = True
USE_TIME = True
USE_IMG = True
USE_CLIP = True

info, fields, weights, format_text, output_format_list = [], [], [], [], []
system_prompt = ""
search_num = [1, 10, 20]


def try_get_response(prompt, messages, max_try_num=3):
	try_num = 0
	flag = True
	response = None
	if LOG_PRINT:
		print("### PROMPT:")
		print_m = messages[1:] if len(messages) > 1 else messages
		p = "\n".join(m["content"] for m in print_m + [{"role": "user", "content": prompt}])
		print(p)
	while flag:
		try:
			# query API
			response = client.chat.completions.create(
				model=engine,
				messages=messages + [{"role": "user", "content": prompt}],
				max_tokens=700,
				temperature=0.3
			)
			flag = False
		except openai.OpenAIError as e:
			if LOG_PRINT:
				print(f'### ERROR: {e}')
			try_num += 1
			if try_num >= max_try_num:
				break
	if LOG_PRINT:
		if not flag:
			print('### RESPONSE:')
			print(response["choices"][0]["message"]["content"])
		else:
			print('### RESPONSE: [ERROR]')
	return response, (not flag)


### prompt
def init_fields():
	info, fields, weights, format_text = [], [], [], []
	if USE_NAME:
		info.append("name")
		fields.append("NAME SIMILARITY")
		weights.append(0.5)
		format_text.append("[NAME SIMILARITY]")
	if USE_NAME and USE_DESC and USE_CODE:
		info.append("description")
		fields.append("PROBABILITY OF DESCRIPTION POINTING SAME ENTITY")
		weights.append(0.3)
		format_text.append("[PROBABILITY OF DESCRIPTION POINTING SAME ENTITY]")
	if USE_STRUCT:
		info.append("structure")
		fields.append("STRUCTURE SIMILARITY")
		weights.append(0.2)
		format_text.append("[STRUCTURE SIMILARITY]")
	if USE_TIME:
		info.append("time")
		fields.append("TIME SIMILARITY")
		weights.append(0.1)
		format_text.append("[TIME SIMILARITY]")
	if USE_DESC:
		info.append("YOUR OWN KNOWLEDGE")
	scale_weights = [w/sum(weights) for w in weights]
	output_format_list = []
	for i, f in enumerate(format_text):
		output_format_list.append(f"{f} = {chr(i+ord('A'))} out of 5")
	return info, fields, scale_weights, format_text, output_format_list


def init_prompt():
	###############
	### get system prompt
	### KG2Code prompt
	###############
	prompt = ""
	### KG2Code
	if USE_CODE:
		tuple_format = 'head_entity, relation, tail_entity'
		if USE_TIME:
			tuple_format += ', start_time, end_time'
		# entity class definition
		class_init_head = "Class Entity: def __init__(self, ent_id"
		if USE_NAME:
			class_init_head += ", name"
		if USE_NAME and USE_DESC:
			class_init_head += ", description"
		if USE_STRUCT:
			class_init_head += ", tuple=[]"
		class_init_head += "): "
		class_init_body = "self.entity_id = ent_id"
		if USE_NAME:
			class_init_body += "self.entity_name = name; "
		if USE_NAME and USE_DESC:
			class_init_body += "self.entity_description = description; "
		if USE_STRUCT:
			class_init_body += "self.tuples = tuples; "
		class_init = f"A Knowledge Graph Entity is defined as follows: " + class_init_head + class_init_body
		# function definition
		func_get_neighbors = f" def get_neighbors(self): neighbors = set(); for {tuple_format} in self.tuples: if head_entity == {'self.entity_name' if USE_NAME else 'self.entity_id'}: neighbors.add(tail_entity); else: neighbors.add(head_entity); return list(neighbors)" if USE_STRUCT else ""
		func_get_relation_information = f" def get_relation_information(self): relation_info = []; for {tuple_format} in self.tuples: relation_info.append(relation); return relation_info" if USE_STRUCT else ""
		func_get_time_information = f" def get_time_information(self): time_info = []; for {tuple_format} in self.tuples: time_info.append((start_time, end_time)); return time_info;" if USE_TIME else ""
		prompt = class_init + func_get_neighbors + func_get_relation_information + func_get_time_information + "\n "
	### basic role and task
	used_infomation = []
	if USE_NAME:
		used_infomation.append(f"name information{' (self.entity_name)' if USE_CODE else ''}")
	if USE_NAME and USE_DESC and USE_CODE:
		used_infomation.append("description information (self.entity_description)")
	if USE_STRUCT:
		used_infomation.append(f"structure information{' (self.tuples, get_neighbors(), get_relation_information())' if USE_CODE else ''}")
	if USE_TIME:
		used_infomation.append(f"time information{' (get_time_information())' if USE_CODE else ''}")
	if USE_DESC:
		used_infomation.append(f"YOUR OWN KNOWLEDGE")
	prompt += f"You are a helpful assistant, helping me align or match entities of knowledge graphs according to {', '.join(used_infomation)}.\n "
	### reasoning example
	example = "Your reasoning process for entity alignment should strictly follow this case step by step: "
	example_entity_1 = {
		"ent_id": "'8535'",
		"name": "'Fudan University'",
		"desc": "'Fudan University, Located in Shanghai, established in 1905, is a prestigious Chinese university known for its comprehensive academic programs and membership in the elite C9 League.'",
		"tuples": "[(Fudan University, Make Statement, China, 2005-11, 2005-11), (Vincent C. Siew, Express intent to meet or negotiate, Fudan University, 2001-05, 2001-05), (Fudan University, Make an appeal or request, Hong Kong, 2003-09, 2003-09)]"
	}
	example_entity_2 = {
		"ent_id": "'24431'",
		"name": "'Fudan_University'",
		"desc": "'Fudan_University in Shanghai, founded in 1905, is a top-ranked institution in China, renowned for its wide range of disciplines and part of the C9 League.'",
		"tuples": "[(Fudan_University, country, China, ~, ~), (Fudan_University, instance of, University, ~, ~), (Shoucheng_Zhang, educated at, Fudan_University, ~, ~)]"
	}
	if USE_CODE:
		example_def_1 = f"Entity({example_entity_1['ent_id']}{', ' + example_entity_1['name'] if USE_NAME else ''}{', ' + example_entity_1['desc'] if USE_NAME and USE_DESC else ''}{', ' + example_entity_1['tuples'] if USE_STRUCT else ''})"
		example_def_2 = f"Entity({example_entity_2['ent_id']}{', ' + example_entity_2['name'] if USE_NAME else ''}{', ' + example_entity_2['desc'] if USE_NAME and USE_DESC else ''}{', ' + example_entity_2['tuples'] if USE_STRUCT else ''})"
		example += f"Given [Main Entity] l_e = {example_def_1}, and [Candidate Entity] r_e = {example_def_2}. "
	else:
		example_def_1 = f"entity ID is {example_entity_1['ent_id']}{(', name is ' + example_entity_1['name']) if USE_NAME else ''}{(', related knowledge tuples are ' + example_entity_1['tuples'] if USE_STRUCT else '')}"
		example_def_2 = f"entity ID is {example_entity_2['ent_id']}{(', name is ' + example_entity_2['name']) if USE_NAME else ''}{(', related knowledge tuples are ' + example_entity_2['tuples'] if USE_STRUCT else '')}"
		example += f"Given [Main Entity], whose {example_def_1}; and [Candidate Entity], whose {example_def_2}. "
	example += " - Do [Main Entity] and [Candidate Entity] align or match? You need to think of the answer STEP BY STEP with {', '.join(info)}: "
	example_step_name = f"think of [NAME SIMILARITY] using name information{' (self.entity_name)' if USE_CODE else ''} : 'Fudan University' and 'Fudan_University' are almost the same from the string itself and its semantic information with only a slight difference, so [NAME SIMILARITY] = 5 out of 5, which means name similarity is [VERY HIGH]"
	example_step_description = "think of [PROBABILITY OF DESCRIPTION POINTING SAME ENTITY] using entity_description: the two description all point the same entity, Fudan University in Shanghai, a top-ranked university in China, so [PROBABILITY OF DESCRIPTION POINTING SAME ENTITY] = 5 out of 5, which means the probability is [VERY HIGH]"
	example_step_struct = f"think of [STRUCTURE SIMILARITY] using tuples information{' (self.tuples, get_neighbors() and get_relation_information())' if USE_CODE else ''} : you can find that China is the common neighbor of l_e and r_e, so [STRUCTURE SIMILARITY] = 3 out of 5, which means [STRUCTURE SIMILARITY] is [MEDIUM]"
	example_step_time = f"think of [TIME SIMILARITY] using temporal information{' (get_time_information())' if USE_CODE else ''} : you can find that r_e does not have specific time information, so just assume [TIME SIMILARITY] = 2 out of 5, which means [TIME SIMILARITY] is [LOW]"
	example_step = []
	if USE_NAME:
		example_step.append(example_step_name)
	if USE_NAME and USE_DESC and USE_CODE:
		example_step.append(example_step_description)
	if USE_STRUCT:
		example_step.append(example_step_struct)
	if USE_TIME:
		example_step.append(example_step_time)
	for i, step in enumerate(example_step):
		example += f"Step {i+1}, {step}. "
	prompt += example
	### output format
	output_format = f"\n [Output Format]: {', '.join(output_format_list)}. "
	output_format += f"NOTICE, {','.join([chr(i+ord('A')) for i in range(len(format_text))])} are in range [1, 2, 3, 4, 5], which respectively means [VERY LOW], [LOW], [MEDIUM], [HIGH], [VERY HIGH]. NOTICE, you MUST strictly output like [Output Format]."
	prompt += output_format

	return prompt


def generate_prompt(main_entity, cand_entity, cand_list, ranked_candidate_list=[]):
	###############
	### Reasoning Prompt
	### Input:
	### 	main/cand_entity: {'name': 'XXXX', 'neighbors': ['(h1, r1, XXXX, temp1_s, temp1_e)', ..., '(XXXX, r2, t2, temp2_s, temp2_e)']}
	### 	cand_list: [cand0_name, cand1_name, ..., cand20_name]
	###		ranked_candidate_list: [(cand0_name, score), (cand1_name, score), ...]
	###############
	if LOG_PRINT:
		print(f"### GENERATED PROMPT: {main_entity['name']} && {cand_entity['name']}")

	### basic entity information
	cand_ent_list = ', '.join([cand["name"] for cand in cand_list]) if USE_NAME else ', '.join([f"'{cand['ent_id']}'" for cand in cand_list])
	prompt = f"[Candidate Entities List] which may be aligned with [Main Entity] {main_entity['name'] if USE_NAME else main_entity['ent_id']} are shown in the following list: [{cand_ent_list}]. "
	ranked_cand = f"Among [Candidate Entities List], ranked entities are shown as follows in format (candidate, align score): [{', '.join([f'({cand}, {score:.3f})' for cand, score in ranked_candidate_list])}]. " if len(ranked_candidate_list) > 0 else ""
	prompt += ranked_cand

	main_ent_neighbors = ['(' + ', '.join(list(neigh)) + ')' for neigh in main_entity['neighbors']]
	cand_ent_neighbors = ['(' + ', '.join(list(neigh)) + ')' for neigh in cand_entity['neighbors']]
	entity_1 = {"ent_id": f"'{main_entity['ent_id']}'", "name": f"'{main_entity['name']}'", "desc": f"'{main_entity['desc']}'", "tuples": f"[{', '.join(main_ent_neighbors)}]"}
	entity_2 = {"ent_id": f"'{cand_entity['ent_id']}'", "name": f"'{cand_entity['name']}'", "desc": f"'{cand_entity['desc']}'", "tuples": f"[{', '.join(cand_ent_neighbors)}]"}
	if USE_CODE:
		main_ent = f"Entity({entity_1['ent_id']}{', ' + entity_1['name'] if USE_NAME else ''}{', ' + entity_1['desc'] if USE_NAME and USE_DESC else ''}{', ' + entity_1['tuples'] if USE_STRUCT else ''})"
		cand_ent = f"Entity({entity_2['ent_id']}{', ' + entity_2['name'] if USE_NAME else ''}{', ' + entity_2['desc'] if USE_NAME and USE_DESC else ''}{', ' + entity_2['tuples'] if USE_STRUCT else ''})"
		prompt += f"\nNow given [Main Entity] l_e = {main_ent}, and [Candidate Entity] r_e = {cand_ent}."
	else:
		main_ent = f"entity ID is {entity_1['ent_id']}{(', name is ' + entity_1['name']) if USE_NAME else ''}{(', related knowledge tuples are ' + entity_1['tuples'] if USE_STRUCT else '')}"
		cand_ent = f"entity ID is {entity_2['ent_id']}{(', name is ' + entity_2['name']) if USE_NAME else ''}{(', related knowledge tuples are ' + entity_2['tuples'] if USE_STRUCT else '')}"
		prompt += f"\nNow given [Main Entity], whose {main_ent}; and [Candidate Entity], whose {cand_ent}. "

	### reasoning step
	think_step = f"- Compared with other Candidate Entities, do [Main Entity] and [Candidate Entity] align or match? Think of the answer STEP BY STEP with {', '.join(info)}: "
	if USE_CODE:
		steps = []
		if USE_NAME:
			steps.append("using self.entity_name")
		if USE_NAME and USE_DESC and USE_CODE:
			steps.append("using self.entity_description")
		if USE_STRUCT:
			steps.append("using self.tuples, get_neighbors() and get_relation_information()")
		if USE_TIME:
			steps.append("using get_time_information()")
	for i, output_format in enumerate(output_format_list):
		step = f"think of {output_format}"
		if USE_CODE:
			step += f", {steps[i]}"
		think_step += f"Step {i+1}, {step}. "
	if USE_DESC:
		think_step += "NOTICE, the information provided above is not sufficient, so use YOUR OWN KNOWLEDGE to complete them.\n"
	prompt += think_step

	### output format
	prompt += f" Output answer strictly in format: {', '.join(output_format_list)}. "

	### simple prompt content, remove entity information
	simple_prompt = f"Do [Main Entity] {main_entity['name']} and [Candidate Entity] {cand_entity['name']} align or match?"
	simple_prompt += f"Think of {', '.join(format_text)}."

	return prompt, simple_prompt


def ask_for_accuracy(main_entity, candidate_pairs, cand_list):
	###############
	### Rethinking Process
	###############
	cand_ent_list = ', '.join([cand["name"] for cand in cand_list]) if USE_NAME else ', '.join([f"'{cand['ent_id']}'" for cand in cand_list])
	prompt = f"[Candidate Entities List] which may be aligned with [Main Entity] {main_entity['name'] if USE_NAME else main_entity['ent_id']} are shown in the following list: [{cand_ent_list}]. "
	### alignment results
	prompt += "Now given the following entity alignments: "
	align_cand_list = []
	for i, candidate in enumerate(candidate_pairs):
		cand_pair_text = f"Candidate {i} = ("
		if USE_NAME:
			cand_pair_text += f"name = {candidate[0]['name']}"
		else:
			cand_pair_text += f"ent_id = {candidate[0]['ent_id']}"
		cand_pair_text += f", align score={candidate[1]}, rank={candidate[2]})"
		align_cand_list.append(cand_pair_text)
	prompt += f"[Main Entity]: {main_entity['name'] if USE_NAME else main_entity['ent_id']} -> [{', '.join(align_cand_list)}]. "
	### rethinking
	prompt += "Compared with candidate entities in [Candidate Entities List], please answer the question: Do these entity alignments are satisfactory enough ([YES] or [NO])?\n"
	prompt += "Answer [YES] if they are relatively satisfactory, which means the alignment score of the top-ranked candidate meet the threshold, and is far higher than others; otherwise, answer [NO] which means we must search other candidate entities to match with [Main Entity]. "
	prompt += "NOTICE, Just answer [YES] or [NO]. Your reasoning process should follow [EXAMPLE]s:\n"
	### rethinking examples
	example1 = "1.[EXAMPLE]:\n [user]: Give the following entity alignments: pMain Entity]: Eric Cantor -> [Candidate 0 = (name=Eric_Cantor, align score=0.92, rank=0), Candidate 1 = (name=George_J._Mitchell, align score=0.4, rank=1), Candidate 2 = (name=John_Turner, align score=0.2, rank=2)].\n [reasoning process]: Given this result, you can find that the alignment score of the first candidate in list 'Eric_Cantor', 0.92, is high enough and is far higher than others, 0.4 and 0.2. So the alignment is relatively satisfactory.\n [assistant]: [YES].\n"
	example2 = "2.[EXAMPLE]:\n [user]: Give the following entity alignments: [Main Entity]: Fudan University -> [Candidate 0 = (name=Peking University, align score=0.6, rank=0), Candidate 1 = (name=Tsinghua University, align score=0.5, rank=1), Candidate 2 = (name=Renming University, align score=0.45, rank=2)].\n [reasoning process]: Given this result, you can find that there is another candidate with a score, 0.5 and 0.45, close to the top-ranked candidate's score 0.6, so the search must continue to ensure a more accurate alignment. \n [assistant]: [NO].\n"
	prompt += example1
	prompt += example2
	prompt += "Just directly answer [YES] or [NO], don't give other text. [assistant]:"
	
	### request LLM
	res, get_res = try_get_response(prompt, [])
	ans = False
	if get_res and res is not None:
		res_content = res["choices"][0]["message"]["content"]
		ans = "yes" in res_content.lower()
	return ans


### process response
def get_score(res:str, field:str="NAME SIMILARITY"):
	content = res.replace('\n', ' ').replace('\t', ' ')
	content = re.sub(r'\d{2}\d*', '', content)
	score = 1
	if field in content:
		score_find = re.findall(f"{field}\D*[=|be|:] \d+", content)
		if len(score_find) > 0:
			score_find = re.findall(f"\d+", score_find[-1])
			score = int(score_find[-1])
	if score < 1:
		if LOG_PRINT:
			print(f'#### SCORE ERROR : {score}')
		score = 1
	if score > 5:
		if LOG_PRINT:
			print(f'#### SCORE ERROR : {score}')
		score = 5
	return score


def eval_alignment_for_evaluate(main_entity, candidate_entities, ref_ent, base_rank):
	###############
	### Reasoning && Rethinking
	### Input:
	###		main_entity: {'ent_id': ent_id, 'name': 'XXXX', 'neighbors': ['(h1, r1, XXXX, temp1_s, temp1_e)', ..., '(XXXX, r2, t2, temp2_s, temp2_e)']}
	###		candidate_entities: [{'ent_id': cand_ent_1_id, *same as main_entity*}, ..., {'ent_id': cand_ent_20_id, ...}]
	###		ref_ent: entity in ref_pairs, which is truly aligned with main_entity
	###		base_rank: rank of ref_ent from method based on embeddings
	###############
	st = time.time()
	rank = base_rank
	iterations = 0
	tokens_count = [0]*22

	### accelerate evaluation process by using some rules
	if base_rank >= 20:
		return rank, 0, tokens_count, time.time() - st
	base_sims = [cand['hhea_sim'] for cand in candidate_entities]
	if base_sims[0] - base_sims[1] > threshold:
		return rank, 1, tokens_count, time.time() - st

	### reasoning by LLM
	if base_rank < 20:
		# systom prompt
		system_messages = [{'role': 'system', 'content': system_prompt}]
		# reasoning
		chat_history = []
		candidate_scores = {cand['ent_id']:[f"'{cand['ent_id']}'", cand['name'], 0.0] for cand in candidate_entities}
		ranked_candidate_list = []
		while iterations < 3:
			responses = []
			aligned_pairs = []
			if LOG_PRINT:
				print(f'### ITERATIONS {iterations + 1}')
			for candidate in list(reversed(candidate_entities[:search_num[iterations]])):
				prompt, simple_prompt = generate_prompt(main_entity, candidate, candidate_entities, ranked_candidate_list)
				messages = system_messages + chat_history
				response, get_response = try_get_response(prompt, messages)

				if get_response and response is not None:
					response_content = response["choices"][0]["message"]["content"]
					usage = response["usage"]
					tokens_count[int(usage["total_tokens"]/200)] += 1
					
					# extract similarity score and calculate the final score
					sims = [get_score(response_content, f) for f in fields]
					score_weights = weights
					score = 0.0
					for j in range(len(sims)):
						score += score_weights[j] * (sims[j] - 1) * 0.25
					if USE_IMG and USE_CLIP:
						img_sim = candidate["mmea_sim"]
						if img_sim > 0:
							score = (1 - img_sim_w) * score + img_sim_w * img_sim
					score = round(score, 5)
					
					# update chat history
					if len(chat_history) >= history_len * 2:
						chat_history = chat_history[2:]
					simple_think_step = []
					for i, s in enumerate(format_text):
						simple_think_step.append(f"{s} = {sims[i]} out of 5")
					simple_response = f"{', '.join(simple_think_step)}."
					chat_history = chat_history + [
						{"role": "user", "content": simple_prompt},
						{"role": "assistant", "content": simple_response}
					]
					if LOG_PRINT:
						sim_name = ["[VERY LOW]", "[LOW]", "[MEDIUM]", "[HIGH]", "[VERY HIGH]"]
						program_output = f"### PROGRAM: {main_entity['name']} && {candidate['name']}, "
						for i, f in enumerate(format_text):
							program_output += f"{f}: {sims[i]}-{sim_name[sims[i]-1]}, "
						if USE_IMG and USE_CLIP:
							program_output += f"[PROBABILITY OF IMAGES POINTING SAME ENTITY]: {img_sim:.4f}, "
						program_output += f"FINAL SCORE : {score:.3f} , has ranked entity num: {len(ranked_candidate_list)}"
						print(program_output)
					
					# update ranked_candidate_list
					candidate_scores[candidate['ent_id']][2] = score
					ranked_candidate_list = []
					for rank_cand in candidate_scores.values():
						if rank_cand[2] != 0.0:
							name = rank_cand[1] if USE_NAME else rank_cand[0]
							ranked_candidate_list.append((name, rank_cand[2]))
					ranked_candidate_list = sorted(ranked_candidate_list, key=lambda x: x[1], reverse=True)

					responses.append((candidate, score))

				if LOG_PRINT:
					print('###############################################')
				
			iterations += 1
			# rank entities according to final score
			aligned_pairs = []
			if len(responses) > 0:
				sorted_entities = sorted(responses, key=lambda x: x[1], reverse=True)
				# update rank of ref_ent
				for j, (cand, score) in enumerate(sorted_entities):
					aligned_pairs.append((cand, score, j))
					if cand['ent_id'] == ref_ent:
						rank = j
				if LOG_PRINT:
					print([(cand["ent_id"], cand["name"], score) for cand, score, _ in aligned_pairs])
				# rethinking by LLM; also accelerate this process by some rules
				if aligned_pairs[0][1] < 0.5 or (len(aligned_pairs) == 1 and aligned_pairs[0][1] < 0.7):	# score is too low, go to next iteration
					good_enough = False
				elif aligned_pairs[0][1] > 0.8:	# score is good enough, stop rethinking
					good_enough = True
				else:
					if len(aligned_pairs) > 1 and aligned_pairs[0][1] - aligned_pairs[1][1] > 0.5:
						# score of top-rank entity is far higher than others, which means good enough, stop rethinking
						good_enough = True
					else:
						# rethink by LLM
						good_enough = ask_for_accuracy(main_entity, aligned_pairs, candidate_entities)	
				if good_enough:
					break

	return rank, iterations, tokens_count, time.time() - st


def re_rank_by_LLM(ng: NeighborGenerator, main_entities, neigh_num=0):
	result = {}
	# load temporarily saved result
	if USE_PREVIOUS_RESULT and os.path.exists(save_result_path):
		with open(save_result_path, "r", encoding="utf-8") as fr:
			result = json.load(fr)
			result = {int(eid):value for eid, value in result.items()}
	# start
	ent_num = 0
	for ent_id in tqdm(main_entities, desc="Reason && Rethink"):
		if ent_id in result:
			continue
		# get entity information
		main_entity = ng.get_neighbors(ent_id, neigh_num)
		candidate_entities = ng.get_candidates(ent_id, neigh_num)
		ref_ent = ng.get_ref_ent(ent_id)
		base_rank = ng.get_base_rank(ent_id)
		# reasoning && rethinking
		llm_rank, iterations, tokens_count, time_cost = eval_alignment_for_evaluate(main_entity, candidate_entities, ref_ent, base_rank)
		result[ent_id] = {"base_rank": int(base_rank), "llm_rank": int(llm_rank), "iteration": int(iterations), "tokens": tokens_count, "time_cost": time_cost}
		# temporarily save result
		ent_num += 1
		if save_step > 0 and ent_num % save_step == 0:
			with open(save_result_path, "w", encoding="utf-8") as fw:
				json.dump(result, fw, ensure_ascii=False, indent=4)
	return result


def evaluate(data, data_dir='data', cand_file='cand', neigh_num=0, hit_k=[1, 5, 10]):
	ng = NeighborGenerator(data=data, data_dir=data_dir, cand_file=cand_file, use_time=USE_TIME, use_desc=True, use_name=USE_NAME, use_img=USE_IMG)
	main_entities = ng.get_entities()
	load_previous_time_cost = USE_PREVIOUS_RESULT and os.path.exists(save_result_path)

	main_st = time.time()
	base_ranks, ranks, iteration_statistic, tokens_count = [], [], [0]*(len(search_num) + 1), [0]*22
	result = re_rank_by_LLM(ng, main_entities, neigh_num)

	previous_time_cost = 0
	for k in result.keys():
		base_ranks.append(result[k]['base_rank'])
		ranks.append(result[k]['llm_rank'])
		iteration_statistic[int(result[k]['iteration'])] += 1
		for i, cnt in enumerate(result[k]['tokens']):
			tokens_count[i] += cnt
		previous_time_cost += result[k]['time_cost'] if "time_cost" in result[k] else 0

	# base rank
	count_ranks(base_ranks)
	hits, mrr = evaluate_alignment(base_ranks, hit_k=hit_k)
	print(f'Base Rank : Hits@{hit_k} = {hits} , MRR = {mrr:.3f}')
	# LLM rank
	hits, mrr = evaluate_alignment(ranks, hit_k=hit_k)
	print(f'LLM Rank  : Hits@{hit_k} = {hits} , MRR = {mrr:.3f}')
	# time cost
	time_cost = time.time() - main_st
	if load_previous_time_cost:
		time_cost += previous_time_cost
	h, m, s = transform_time(int(time_cost))
	print(f'Time Cost : {h}hour, {m:02d}min, {s:02d}sec')

	return result, iteration_statistic, tokens_count


def print_tokens_count(tokens_type, tokens_count):
	total = sum(tokens_count)
	print(f"{tokens_type} :  {total}")
	for i, cnt in enumerate(tokens_count):
		if cnt != 0:
			print(f"\t[{i*200:4d} , {(i+1)*200:4d})  :  {cnt} , {cnt/total:.2%}")





### parameters
parser = argparse.ArgumentParser()
# LLM setting
parser.add_argument('--LLM', type=str, choices=["llama", "gpt3.5", "gpt4"], default="llama")
# data setting
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--data', type=str, default='icews_wiki')
parser.add_argument('--cand_file', type=str, default='cand')
parser.add_argument('--result_name', type=str, default='base')
# MM-ChatAlign setting
parser.add_argument('--neigh', type=int, default=0)
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--img_sim_w', type=float, default=0.1)
# ablation study
parser.add_argument('--no_code', action='store_true')
parser.add_argument('--no_desc', action='store_true')
parser.add_argument('--no_name', action='store_true')
parser.add_argument('--no_struct', action='store_true')
parser.add_argument('--no_img', action="store_true")
parser.add_argument('--no_clip', action="store_true")
parser.add_argument('--no_time', action='store_true')
# others
parser.add_argument('--save_step', type=int, default=10)
parser.add_argument('--log_print', action='store_true')
parser.add_argument('--new_result', action='store_true')
args = parser.parse_args()

### setting
save_dir = os.path.join("result", args.data)
if not os.path.exists(save_dir):
	os.makedirs(save_dir)
save_result_path = os.path.join(save_dir, f"result_thr_{args.threshold}_{args.result_name}_{args.cand_file}.json")

### change these settings, according to your LLM
### openai == 0.28.1
if args.LLM == "llama":
	openai.api_base = f"http://localhost:8000/v1"
	engine = openai.Model.list()["data"][0]["id"]
elif args.LLM == "gpt3.5":
	engine = "gpt-3.5-turbo"
elif args.LLM == "gpt4":
	engine = "gpt-4-1106-preview"

### set hyper-parameters
save_step = args.save_step
LOG_PRINT = args.log_print
USE_PREVIOUS_RESULT = not args.new_result

threshold = args.threshold
img_sim_w = args.img_sim_w

USE_CODE = not args.no_code
USE_DESC = not args.no_desc
USE_NAME = not args.no_name
USE_STRUCT = not args.no_struct
USE_IMG = not args.no_img
USE_CLIP = not args.no_clip
USE_TIME = True and not args.no_time if args.data in ["icews_wiki", "icews_yago"] else False

### Intialize
info, fields, weights, format_text, output_format_list = init_fields()
system_prompt = init_prompt()

### ChatEA evaluation
result, iteration_statistic, tokens_count = evaluate(data=args.data, data_dir=args.data_dir, cand_file=args.cand_file, neigh_num=args.neigh, hit_k=[1, 5, 10])

### print tokens count
print(f'Iteration Num Count: {list(range(len(search_num) + 1))} = {iteration_statistic}')
print('\nTokens Count:')
print_tokens_count("Total", tokens_count)

### save result
with open(save_result_path, 'w', encoding='utf-8') as fw:
	json.dump(result, fw, ensure_ascii=False, indent=4)
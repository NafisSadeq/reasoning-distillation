import os
import json
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from scipy.stats import pearsonr

def save_score_dist(scores_chosen, scores_rejected, save_path, ymax, xmax):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    formatter = FuncFormatter(lambda x, _: f'{int(x/1000)}k')

    # Plot for chosen scores
    axes[0].hist(scores_rejected, bins=range(11), edgecolor='black')
    axes[0].set_xlabel('Rejected Score', fontsize=24)
    axes[0].set_ylabel('Frequency', fontsize=24)
    axes[0].set_xlim(0, xmax)
    axes[0].set_ylim(0, ymax)
    axes[0].yaxis.set_major_formatter(formatter)
    axes[0].tick_params(axis='both', labelsize=20)
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))


    # Plot for rejected scores
    axes[1].hist(scores_chosen, bins=range(11), edgecolor='black')
    axes[1].set_xlabel('Chosen Score', fontsize=24)
    axes[1].set_xlim(0, xmax)
    axes[1].tick_params(axis='both', labelsize=24)
    axes[1].yaxis.set_major_formatter(formatter)
    axes[1].tick_params(axis='both', labelsize=20)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()

def construct_dataset(dir_loc,rule_reward_file,all_hypo_file,apply_rule_file,score_diff,task_name):

    with open(dir_loc+"/"+rule_reward_file,'r') as file:
        rule_reward_orig = json.load(file)

    with open(dir_loc+"/"+all_hypo_file,'r') as file:
        all_hypo_list = json.load(file)

    scores_chosen = []
    scores_rejected = []
    sharegpt_pair_list = []
    kto_list = []
    apply_rule_list = []
    generate_rule_list = []
    high_quality_rules = set()
    unique_rules = set()

    all_score_list = []
    all_rule_len = []
    
    for rule_reward, all_hypo in zip(rule_reward_orig,all_hypo_list):
        
        prompt = rule_reward['chosen'][0]
        score_rule_list = []
        
        for hypo in all_hypo:
            if hypo['Rule'] not in unique_rules:
                unique_rules.add(hypo['Rule'])
                score_rule_list.append((hypo['Score'],hypo['Rule']))
                all_score_list.append(hypo['Score'])
                all_rule_len.append(len(hypo['Rule'].split()))
            else:
                continue
            
        score_rule_list.sort(reverse=True)
        kto_count = 0
        
        for score_rule_x in score_rule_list:

            isi = prompt["content"].find("Input:")
            instruction = prompt["content"][:isi]
            input_content = prompt["content"][isi:]

            generate_rule_list.append(
                {
                    "instruction": instruction,
                    "input": input_content,
                    "output": score_rule_x[1]
                }
            )

            if(2*kto_count<len(score_rule_list)):
                kto_label = True
            else:
                kto_label = False

            kto_list.append(
                {
                "messages": [
                  {
                    "content": prompt["content"],
                    "role": "user"
                  },
                  {
                    "content": score_rule_x[1],
                    "role": "assistant"
                  }
                ],
                "label": kto_label
                }
            )
            kto_count+=1
            
            for score_rule_y in score_rule_list:
                
                if(score_rule_x[0]>(score_rule_y[0]+score_diff)):
    
                    sharegpt_pair = {}
                    chosen_rule = score_rule_x[1]
                    chosen_score = score_rule_x[0]
                    rejected_rule = score_rule_y[1]
                    rejected_score = score_rule_y[0]
                
                    scores_chosen.append(chosen_score)
                    scores_rejected.append(rejected_score)
    
                    sharegpt_pair["conversations"] = [
                        {
                            "from": "human",
                            "value": prompt["content"]
                        }
                    ]
    
                    sharegpt_pair["chosen"] = {
                        "from": "gpt",
                        "value": chosen_rule
                    }
    
                    sharegpt_pair["rejected"] = {
                        "from": "gpt",
                        "value": rejected_rule
                    }
    
                    sharegpt_pair_list.append(sharegpt_pair)
                    high_quality_rules.add(chosen_rule)

    hist1, _ = np.histogram(scores_chosen, bins=range(11))
    hist2, _ = np.histogram(scores_rejected, bins=range(11))
    max_freq = max(hist1.max(), hist2.max())
    max_example = min(max(scores_chosen + scores_rejected + [5]) + 1,12)

    max_score = max(all_score_list)
    all_score_list = [x/max_score for x in all_score_list]
    
    save_score_dist(scores_chosen, scores_rejected, dir_loc + "/"+task_name+"_scores.png", max_freq, max_example)

    with open(dir_loc+"/"+apply_rule_file,'r') as file:
        apply_prompt_response_list = json.load(file)

    for prompt_response in tqdm(apply_prompt_response_list):
        prompt = prompt_response["prompt"]
        response = prompt_response["response"]
        isi = prompt.find("Input:")
        instruction = prompt[:isi]
        input_content = prompt[isi:]

        for rule in high_quality_rules:

            if(rule in prompt):
                apply_rule_list.append(
                    {
                        "instruction": instruction,
                        "input": input_content,
                        "output": response
                    }
                )
                break

    print(dir_loc,len(sharegpt_pair_list),len(kto_list),len(generate_rule_list),len(apply_rule_list))

    return sharegpt_pair_list, kto_list, generate_rule_list, apply_rule_list,all_rule_len,all_score_list

task_data_locs = [
    (
    "./data/1d_arc/gpt-4",
    "rule_reward_set_train_25_1.0.json",
    "all_hypo_train_25_1.0.json",
    "rule_apply_train_25_1.0.json",
    1,
    "1D_ARC"
    ),
    (
    "./data/acre/gpt-4",
    "rule_reward_set_train_50_1.0.json",
    "all_hypo_train_50_1.0.json",
    "rule_apply_train_50_1.0.json",
    2,
    "ACRE"
    ),
    (
    "./data/list_func/gpt-4",
    "rule_reward_set_train_50_1.0.json",
    "all_hypo_train_50_1.0.json",
    "rule_apply_train_50_1.0.json",
    3,
    "List_Function"
    ),
    (
    "./data/scan/gpt-4",
    "rule_reward_set_train_50_1.0.json",
    "all_hypo_train_50_1.0.json",
    "rule_apply_train_50_1.0.json",
    4,
    "MiniSCAN"
    )
]

sharegpt_pair_list = []
kto_list = []
generate_rule_list = []
apply_rule_list = []
hyp_lengths = []
hyp_qualities = []

for task_data in task_data_locs:

    spl, kl, grl, arl,hl,hq = construct_dataset(task_data[0],task_data[1],task_data[2],task_data[3],task_data[4],task_data[5])
    sharegpt_pair_list += spl
    kto_list += kl
    generate_rule_list += grl
    apply_rule_list += arl
    hyp_lengths += hl
    hyp_qualities += hq

corr, _ = pearsonr(hyp_qualities, hyp_lengths)
print("All","pearson hypothesis length vs quality",corr)

if(not os.path.exists("./data/merged")):
    os.makedirs("./data/merged")

with open("data/merged/generate_rule_dpo.json",'w') as file:
    json.dump(sharegpt_pair_list,file,indent=4)

with open("data/merged/generate_rule_kto.json",'w') as file:
    json.dump(kto_list,file,indent=4)

with open("data/merged/generate_rule_sft.json",'w') as file:
    json.dump(generate_rule_list,file,indent=4)

with open("data/merged/apply_rule_sft.json",'w') as file:
    json.dump(apply_rule_list,file,indent=4)

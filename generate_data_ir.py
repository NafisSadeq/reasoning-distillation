import os
import json
import random
import argparse
from tqdm.auto import tqdm
from utils import extract_list_substring
from llm import ChatGPT, LlamaAdapter, QwenAdapter

parser = argparse.ArgumentParser(description='Run LLM with specific parameters.')
parser.add_argument('--llm_name', type=str, default='gpt-4o',choices=[
    'meta-llama/Meta-Llama-3-8B-Instruct',
    'mistralai/Mistral-7B-Instruct-v0.3', 
    'Qwen/Qwen2.5-7B-Instruct',
    'gpt-3.5-turbo-1106',
    'gpt-4o'
], help='Name of the language model')
parser.add_argument('--task', type=str, default="list_func",choices=[
    'list_func',
    '1d_arc', 
    'acre', 
    'scan'
], help='Task Name')
parser.add_argument('--hypo_size', type=int, default=50, help='Hypothesis sample size for rule generation')
parser.add_argument('--temperature', type=float, default=1.0, help='Temperature setting for text generation')

args = parser.parse_args()

random.seed(10)

llm_name = args.llm_name
adapter_name = None
task = args.task
hypo_size = args.hypo_size
temperature = args.temperature

if(llm_name.startswith("Qwen")):
    llm = QwenAdapter(llm_name,adapter_name)
    llm_tag = "qwen"
elif(llm_name.startswith("meta")):
    llm = LlamaAdapter(llm_name,adapter_name)
    llm_tag = "llama3"
elif(llm_name.startswith("mistralai")):
    llm = LlamaAdapter(llm_name,adapter_name)
    llm_tag = "mistral"
else:
    llm = ChatGPT(llm_name)
    llm_tag = llm_name[:5]

if(task == "list_func"):
    data_path = "./data/list_func/list_function.jsonl"
    test_data_path = "./data/list_func/list_function_test.jsonl"
elif(task == "1d_arc"):
    data_path = "./data/1d_arc/1D_arc.jsonl"
    test_data_path = "./data/1d_arc/1D_arc_test.jsonl"
elif(task == "acre"):
    data_path = "./data/acre/acre.jsonl"
    test_data_path = "./data/acre/acre_test.jsonl"
elif(task == "scan"):
    data_path = "./data/scan/scan.jsonl"
    test_data_path = "./data/scan/scan_test.jsonl"
else:
    data_path = None
    test_data_path = None

data = []
with open(data_path,'r') as infile:
    for line in infile:
        data.append(eval(line.strip()))

random.shuffle(data)

with open("./config/prompts.json",'r') as infile:
    prompts = json.load(infile)

print("# sample",len(data))

output_dir = "./data/"+task+"/"+llm_tag

if(not os.path.exists(output_dir)):
    os.makedirs(output_dir)

def process_split(data_split,split_name):

    rule_reward_list = []
    all_hypo_list = []
    rule_application_list = []
    
    for di,datum in enumerate(tqdm(data_split)):
    
        prompt = prompts[task]["generate_rule"]
        
        for example in datum['train']:
            prompt += "\n"
            prompt += "Input: "+ str(example['input'])+"\n"
            prompt += "Output: "+ str(example['output'])+"\n"
        
        prompt += "\n"
    
        rule_list = []
        rule_score_list = []
        curr_hypo = []
        rule_reward = {}
        
        for ri in range(hypo_size):
            try:
                rule = llm.generate(prompt, temperature=temperature)
            except:
                continue
            rule_score = 0
            for ei,example in enumerate(datum['train']):
                test_prompt = prompts[task]["apply_rule"]+"\n"+rule+"\n"
                test_prompt = test_prompt+ "Input: "+ str(example['input'])+"\n"
                response = llm.generate(test_prompt, temperature=temperature)

                rule_application_list.append(
                    {
                        "prompt": test_prompt,
                        "response": response
                    }
                )
                
                if(task=="list_func" or task=="1d_arc"):
                    prediction = extract_list_substring(response)
                else:
                    prediction = response
            
                if(prediction is not None and prediction==example['output']):
                    rule_score+=1
                    
            rule_list.append(rule)
            rule_score_list.append(rule_score)
            curr_hypo.append(
                {
                    "Rule": rule,
                    "Score": rule_score
                }
            )
    
        sorted_list = list(zip(rule_score_list,rule_list))
        sorted_list.sort(reverse=True)
        rule = sorted_list[0][1]
        rule_score = sorted_list[0][0]
    
        rule_reward['chosen'] = [
            {
                "content": prompt,
                "role": "user"
            },
            {
                "content": sorted_list[0][1],
                "role": "assistant"
            }]
    
        rule_reward['rejected'] = [
            {
                "content": prompt,
                "role": "user"
            },
            {
                "content": sorted_list[-1][1],
                "role": "assistant"
            }]
        
        rule_reward['score_chosen'] = sorted_list[0][0]
        rule_reward['score_rejected'] = sorted_list[-1][0]
        print(sorted_list[0][0],sorted_list[-1][0],llm.get_cost())
    
        rule_reward_list.append(rule_reward)
        all_hypo_list.append(curr_hypo)
    
        with open(output_dir+"/rule_reward_set_"+str(split_name)+"_"+str(hypo_size)+"_"+str(temperature)+".json",'w') as outfile:
            json.dump(rule_reward_list,outfile,indent=4)   
        with open(output_dir+"/all_hypo_"+str(split_name)+"_"+str(hypo_size)+"_"+str(temperature)+".json",'w') as outfile:
            json.dump(all_hypo_list,outfile,indent=4)
        with open(output_dir+"/rule_apply_"+str(split_name)+"_"+str(hypo_size)+"_"+str(temperature)+".json",'w') as outfile:
            json.dump(rule_application_list,outfile,indent=4)
    
    with open(output_dir+"/rule_reward_set_"+str(split_name)+"_"+str(hypo_size)+"_"+str(temperature)+".json",'w') as outfile:
        json.dump(rule_reward_list,outfile,indent=4)
    
    with open(output_dir+"/all_hypo_"+str(split_name)+"_"+str(hypo_size)+"_"+str(temperature)+".json",'w') as outfile:
        json.dump(all_hypo_list,outfile,indent=4)

    with open(output_dir+"/rule_apply_"+str(split_name)+"_"+str(hypo_size)+"_"+str(temperature)+".json",'w') as outfile:
        json.dump(rule_application_list,outfile,indent=4)

train_len = int(len(data)*0.9)

with open(test_data_path, 'w') as outfile:
    for entry in data[train_len:]:
        json.dump(entry, outfile)
        outfile.write('\n')

process_split(data[:train_len],"train")
process_split(data[train_len:],"valid")
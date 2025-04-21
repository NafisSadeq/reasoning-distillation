import os
import json
import random
import argparse
from tqdm.auto import tqdm
from utils import extract_list_substring
from llm import ChatGPT, LlamaAdapter, QwenAdapter

parser = argparse.ArgumentParser(description='Run LLM with specific parameters.')
parser.add_argument('--llm_name', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",choices=[
    'meta-llama/Meta-Llama-3-8B-Instruct',
    'mistralai/Mistral-7B-Instruct-v0.3', 
    'Qwen/Qwen2.5-7B-Instruct',
    'gpt-3.5-turbo-1106',
    'gpt-4o'
], help='Name of the language model')
parser.add_argument('--adapter_path', type=str, help='Name or path of adapter')
parser.add_argument('--task', type=str, default="list_func",choices=[
    'list_func',
    '1d_arc', 
    'acre', 
    'scan'
], help='Task Name')
parser.add_argument('--hypo_size', type=int, default=5, help='Hypothesis sample size for rule generation')
parser.add_argument('--rg_temp', type=float, default=0.9, help='Temperature for rule generation')
parser.add_argument('--rf_temp', type=float, default=0.7, help='Temperature for rule generation')

args = parser.parse_args()
random.seed(10)

llm_name = args.llm_name
adapter_path = args.adapter_path
task = args.task
hypo_size = args.hypo_size
rg_temp = args.rg_temp
rf_temp = args.rf_temp

if(llm_name.startswith("Qwen")):
    llm = QwenAdapter(llm_name,adapter_path)
    llm_tag = "qwen"
elif(llm_name.startswith("meta")):
    llm = LlamaAdapter(llm_name,adapter_path)
    llm_tag = "llama3"
elif(llm_name.startswith("mistralai")):
    llm = LlamaAdapter(llm_name,adapter_path)
    llm_tag = "mistral"
else:
    llm = ChatGPT(llm_name)
    llm_tag = llm_name[:5]

adapter_tag = adapter_path.split("/")[-3]

if(task == "list_func"):
    data_path = "./data/list_func/list_function_test.jsonl"
elif(task == "1d_arc"):
    data_path = "./data/1d_arc/1D_arc_test.jsonl"
elif(task == "acre"):
    data_path = "./data/acre/acre_test.jsonl"
elif(task == "scan"):
    data_path = "./data/scan/scan_test.jsonl"
else:
    data_path = None

data = []
with open(data_path,'r') as infile:
    for line in infile:
        data.append(eval(line.strip()))

with open("./config/prompts.json",'r') as infile:
    prompts = json.load(infile)

num_test = 0
num_corr = 0

print("# sample",len(data))

for di,datum in enumerate(tqdm(data)):

    prompt = prompts[task]["generate_rule"]
    
    for example in datum['train']:
        prompt += "\n"
        prompt += "Input: "+ str(example['input'])+"\n"
        prompt += "Output: "+ str(example['output'])+"\n"
    
    prompt += "\n"

    rule_list = []
    rule_score_list = []
    for ri in range(hypo_size):
        rule = llm.generate(prompt,temperature=rg_temp)
        rule_score = 0
        for ei,example in enumerate(datum['train']):
            test_prompt = prompts[task]["apply_rule"]+"\n"+rule+"\n"
            test_prompt = test_prompt+ "Input: "+ str(example['input'])+"\n"
            response = llm.generate(test_prompt,temperature=rf_temp)
            if(task=="list_func" or task=="1d_arc"):
                prediction = extract_list_substring(response)
            else:
                prediction = response
        
            if(prediction is not None and prediction==example['output']):
                rule_score+=1
        rule_list.append(rule)
        rule_score_list.append(rule_score)

    sorted_list = list(zip(rule_score_list,rule_list))
    sorted_list.sort(reverse=True)
    rule = sorted_list[0][1]
    rule_score = sorted_list[0][0]
       
    for ei,example in enumerate(datum['test']):
        test_prompt = prompts[task]["apply_rule"]+"\n"+rule+"\n"
        test_prompt = test_prompt+ "Input: "+ str(example['input'])+"\n"
        response = llm.generate(test_prompt,temperature=rf_temp)
        if(task=="list_func" or task=="1d_arc"):
            prediction = extract_list_substring(response)
        else:
            prediction = response
       
        num_test+=1
        if(prediction is not None and prediction==example['output']):
            num_corr+=1
        data[di]['test'][ei]["ir-rule"] = rule
        data[di]['test'][ei]["ir-output"] = prediction

accuracy = round(num_corr/num_test,3)
print("Temp",rg_temp,rf_temp)
print("Accuracy:",accuracy)
print("Cost:",llm.get_cost())


output_dir = "./outputs"

if(not os.path.exists(output_dir)):
    os.makedirs(output_dir)

file_prefix = "proposed_"+llm_tag+"_"+adapter_tag+"_"+task+"_"+str(hypo_size)

with open(output_dir+"/"+file_prefix+".log",'a') as outfile:
    outfile.write("Accuracy: "+str(accuracy)+"\n")
    outfile.write("\n")

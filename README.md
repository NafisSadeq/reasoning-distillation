## Summary

This repository contains the code for the paper [Improving In-Context Learning with Reasoning Distillation](https://arxiv.org/abs/2504.10647). In this work, we improve the inductive reasoning ability of language models using teacher-to-student reasoning distillation. The proposed method achieves improved accuracy and cost-to-performance ratio compared to the teacher model GPT-4o.

## News

Model checkpoints are now available in Hugging Face 🤗
1. [ReDis-Llama](https://huggingface.co/nsadeq/ReDis-Llama)
2. [ReDis-Qwen](https://huggingface.co/nsadeq/ReDis-Qwen)
3. [ReDis-Mistral](https://huggingface.co/nsadeq/ReDis-Mistral)

All training datasets are now available in Hugging Face 🤗
1. [Generate Rule Alignment](https://huggingface.co/datasets/nsadeq/redis_generate_rule_alignment)
2. [Generate Rule SFT](https://huggingface.co/datasets/nsadeq/redis_generate_rule_sft)
3. [Follow Rule SFT](https://huggingface.co/datasets/nsadeq/redis_follow_rule_sft)

## Install

Clone repository
```
git clone https://github.com/NafisSadeq/reasoning-distillation.git
cd reasoning-distillation/
```
Create conda environment using the provided requirements.txt

```
conda create -n redis python=3.9
pip install -r requirements.txt
```

## Inference (Hugging Face checkpoints)

You can directly use ReDis adapters from Hugging Face to perform inference.
```
python proposed.py --task acre --llm_name meta-llama/Meta-Llama-3-8B-Instruct --adapter_path nsadeq/ReDis-Llama --hypo_size 10 --rg_temp 0.9 --rf_temp 0.7
python proposed.py --task acre --llm_name Qwen/Qwen2.5-7B-Instruct --adapter_path nsadeq/ReDis-Qwen --hypo_size 10 --rg_temp 0.9 --rf_temp 0.7
python proposed.py --task acre --llm_name mistralai/Mistral-7B-Instruct-v0.3 --adapter_path nsadeq/ReDis-Mistral --hypo_size 10 --rg_temp 0.9 --rf_temp 0.7
```

## Data augmentation (Custom Training Only)

Perform data augmentation using a teacher model such GPT-4o

```
python generate_data_ir.py --llm_name gpt-4o --hypo_size 50 --task list_func
python generate_data_ir.py --llm_name gpt-4o --hypo_size 50 --task 1d_arc
python generate_data_ir.py --llm_name gpt-4o --hypo_size 50 --task acre
python generate_data_ir.py --llm_name gpt-4o --hypo_size 50 --task scan
```

Perform data filtering and merge the task-specific data segments

```
python construct_pair.py
```

This step will generate three datasets, generate_rule_sft.json, apply_rule_sft.json, and generate_rule_dpo.json. We already provide the three datasets within ./data/merged folder

## Model training (Custom Training Only)

We use LLaMA-Factory for model training. Install LLaMA-Factory following instructions [here](https://github.com/hiyouga/LLaMA-Factory/tree/main?tab=readme-ov-file#installation). Copy the three dataset files within [dataset folder](https://github.com/hiyouga/LLaMA-Factory/tree/main/data) and update [dataset info](LLaMA-Factorydata/dataset_info.json) as required. Follow the instructions [here](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md). Then you can perform suprvised fine-tuning as follows. 

```  
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```

After that you can perform alignment as follows.

```
llamafactory-cli train examples/train_lora/llama3_lora_dpo.yaml
```

Make sure to update the training configuration files llama3_lora_sft.yaml and llama3_lora_dpo.yaml with appropriate model name, dataset names, and hyper-parameters.

## Inference

Load the base models with corresponding LoRA adapters created during model tuning for inference.

```
python proposed.py --task list_func --llm_name meta-llama/Meta-Llama-3-8B-Instruct --adapter_path <adapter_path> --hypo_size 10 --rg_temp 0.9 --rf_temp 0.7
```

## Baselines
We also provide codes for corresponding baseline codes for both direct few-shot prompting and hypothesis search. For direct few-shot prompting, you can run

```
python baseline_io.py --task list_func --llm_name meta-llama/Meta-Llama-3-8B-Instruct
```

For hypothesis search with base LLaMA model, you can run

```
python baseline_ir.py --task list_func --llm_name meta-llama/Meta-Llama-3-8B-Instruct --hypo_size 10
```

## Citation

If you use the code, dataset or model checkpoints, please cite the following work.

```
@misc{sadeq2025improvingincontextlearningreasoning,
      title={Improving In-Context Learning with Reasoning Distillation}, 
      author={Nafis Sadeq and Xin Xu and Zhouhang Xie and Julian McAuley and Byungkyu Kang and Prarit Lamba and Xiang Gao},
      year={2025},
      eprint={2504.10647},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.10647}, 
}
```


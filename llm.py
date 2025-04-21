import os
import json
import time 
import torch
import openai
import transformers
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

class DEEPSEEK:

    def __init__(self, model_name="deepseek-chat"):
        
        self.model_name = model_name
        self.prompt_token=0
        self.gen_token=0
        self.cost=0
         
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.sleep_time = 0.5
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        
        self.prompt_token_cost=0
        self.gen_token_cost=0
            

    def generate(self,prompt,sys_prompt="You are a helpful assistant",temperature=0.6):

        message_list = []

        if(sys_prompt is not None):
            message_list.append({"role": "system", "content": sys_prompt})

        message_list.append({"role": "user", "content": prompt})

        try:
            response_object = self.client.chat.completions.create(
                model=self.model_name,
                messages=message_list,
                temperature = temperature
                )
        except:
            time.sleep(self.sleep_time)
            response_object = self.client.chat.completions.create(
                model=self.model_name,
                messages=message_list,
                temperature = temperature
                )

        response = response_object.choices[0].message.content
        self.prompt_token+=response_object.usage.prompt_tokens
        self.gen_token+=response_object.usage.completion_tokens
        
        return response
    
    def get_cost(self):
        
        cost = (self.prompt_token_cost*self.prompt_token+self.gen_token_cost*self.gen_token)/1000000

        return cost,self.prompt_token,self.gen_token

class ChatGPT:

    def __init__(self, model_name):
        
        self.model_name = model_name
        self.prompt_token=0
        self.gen_token=0
        self.cost=0
         
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.sleep_time = 0.5
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            )
        
        self.prompt_token_cost=0
        self.gen_token_cost=0
        
        if(self.model_name=="gpt-3.5-turbo-0613"):
            self.prompt_token_cost=1.5
            self.gen_token_cost=2

        elif(self.model_name=="gpt-3.5-turbo-1106"):
            self.prompt_token_cost=1
            self.gen_token_cost=2

        elif(self.model_name=="gpt-3.5-turbo-0125"):
            self.prompt_token_cost=0.5
            self.gen_token_cost=1.5

        elif(self.model_name=="gpt-4"):
            self.prompt_token_cost=30
            self.gen_token_cost=60
        
        elif(self.model_name=="gpt-4-0613"):
            self.prompt_token_cost=30
            self.gen_token_cost=60

        elif(self.model_name=="gpt-4-0125-preview"):
            self.prompt_token_cost=10
            self.gen_token_cost=30

        elif(self.model_name=="gpt-4-1106-preview"):
            self.prompt_token_cost=10
            self.gen_token_cost=30

        elif(self.model_name=="gpt-4-vision-preview"):
            self.prompt_token_cost=10
            self.gen_token_cost=30

        elif(self.model_name=="gpt-4o"):
            self.prompt_token_cost=2.5
            self.gen_token_cost=10
            

    def generate(self,prompt,sys_prompt=None,temperature=0.6):

        message_list = []

        if(sys_prompt is not None):
            message_list.append({"role": "system", "content": sys_prompt})

        message_list.append({"role": "user", "content": prompt})

        if(self.model_name=="o1"):
            try:
                response_object = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=message_list
                    )
            except:
                time.sleep(self.sleep_time)
                response_object = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=message_list
                    )            
        else:
            try:
                response_object = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=message_list,
                    temperature = temperature
                    )
            except:
                time.sleep(self.sleep_time)
                response_object = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=message_list,
                    temperature = temperature
                    )

        response = response_object.choices[0].message.content
        self.prompt_token+=response_object.usage.prompt_tokens
        self.gen_token+=response_object.usage.completion_tokens
        
        return response
    
    def get_cost(self):
        
        cost = (self.prompt_token_cost*self.prompt_token+self.gen_token_cost*self.gen_token)/1000000

        return cost,self.prompt_token,self.gen_token

class LlamaAdapter:

    def __init__(self, model_name,adapter_name=None):

        self.model_name = model_name
        if(adapter_name is None):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            ).to("cuda")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            ).to("cuda")
            self.model.load_adapter(adapter_name)

        self.prompt_token=0
        self.gen_token=0

    def generate(self,prompt,sys_prompt=None,temperature=0.6):

        messages = []

        if(sys_prompt is not None):
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": prompt})
        
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        self.prompt_token += input_ids.shape[1]
        
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
            top_p=1.0
        )
        response = outputs[0][input_ids.shape[-1]:]
        self.gen_token += response.shape[0]
        
        return self.tokenizer.decode(response, skip_special_tokens=True)

    def get_cost(self):

        return 0,self.prompt_token,self.gen_token

class QwenAdapter:

    def __init__(self, model_name,adapter_name=None):

        self.model_name = model_name
        if(adapter_name is None):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            ).to("cuda")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            ).to("cuda")
            self.model.load_adapter(adapter_name)

        self.prompt_token=0
        self.gen_token=0

    def generate(self,prompt,sys_prompt=None,temperature=0.6):

        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        self.prompt_token += model_inputs['input_ids'].shape[1]
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=temperature,
            top_p=1.0
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        self.gen_token += generated_ids[0].shape[0]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

    def get_cost(self):

        return 0,self.prompt_token,self.gen_token
                                                                                                                                                                
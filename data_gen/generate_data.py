import os
import random
import json
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
# from gpt import GPT
import random
import requests
import argparse
# from image_process import get_image 
from functions import *
import requests
import json
import random
from openai import OpenAI
import concurrent.futures
# from prompt_reasoning import Prompt_end, Prompts_start
import pandas as pd
import csv
from tqdm import tqdm as tqdm_progress
from collections import defaultdict
MAX_NUMS = 1000000
from prompt_final import Prompts
from concurrent.futures import ThreadPoolExecutor, as_completed
class GPT4:
    def __init__(self, model_name='gpt-4-turbo') -> None:
        self.key_ind = 0
        self.max_wrong_time = 5
        self.model_name = model_name
        self.url = "https://api.ai-gaochao.cn/v1/chat/completions"
        self.keys = [['sk-OSJYr1SDOVosYp6hB1695a010eC7425a8693D26c5a3fB2F8', '']]

        assert len(self.keys) > 0, 'have no key'
        self.wrong_time = [0] * len(self.keys)
        print(f'keys: {self.keys}')
        print(f'use model of {self.model_name}')

    def init_api_keys(self):
        self.keys = []
        with open('gpt_key.txt', encoding="utf-8", mode="r") as fr:
            for l in fr:
                cols = l.split('---')
                if len(cols[0]) < 45 or len(cols[0]) > 55:
                    continue
                if len(cols) == 1:
                    cols.append('None')
                self.keys.append((cols[0], cols[1]))
        assert len(self.keys) > 0, 'have no key'
        print(f'keys: {self.keys}')
        self.wrong_time = [0] * len(self.keys)
        random.shuffle(self.keys)

    def get_api_key(self):
        self.key_ind = (self.key_ind + 1) % len(self.keys)
        return self.keys[self.key_ind]

    def call(self, content, args={}, showkeys=False):
        api_key, organization = self.get_api_key()
        if showkeys:
            print(api_key, organization)
        if organization == 'None':
            organization = ''

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Organization": organization,
        }

        parameters = {
            "model": self.model_name,
            "messages": [{'role': 'user', 'content': content}],
            **args,
        }

        response = requests.post(
            self.url,
            headers=headers,
            json=parameters
        )
        response = json.loads(response.content.decode("utf-8"))
        if 'error' in response:
            self.wrong_time[self.key_ind] += 1
            if self.wrong_time[self.key_ind] > self.max_wrong_time:
                print(response)
                print(f'del {self.keys[self.key_ind]}')
                # del self.keys[self.key_ind]
                # del self.wrong_time[self.key_ind]
            assert False, str(response)
        return response['choices'][0]['message']['content']

    def test(self):
        for _ in range(len(self.keys)):
            try:
                print(self.call('你好', showkeys=True))
            except Exception as e:
                print(e)

    # @retry(wait_fixed=500, stop_max_attempt_number=5)
    def retry_call(self, content, args={"max_tokens": 4096}):
        return self.call(content, args)

def select_from_data(file_path, sample_size=4200):
    """
    从指定的JSON文件中读取数据，将每个item的"task"归类，并从每个类别中随机抽样。

    :param file_path: JSON文件的路径
    :param sample_size: 每个task类别中随机抽样的数量
    :return: 抽样得到的"context"列表
    """
    # 读取 JSON 文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 创建一个字典来存储任务类别
    task_classes = defaultdict(list)

    # 将每个 item 按照 task 分类
    for item in data:
        task = item.get('task')
        if task == "normal1_2_v1" or task == "normal1_3_v1":
            if len(item["context"]) > 1:
                item["context"] = [item["context"][0]]
        # item["retrieval_ctxs_old"] = item["retrieval_ctxs"]
        # del item["retrieval_ctxs"]
        task_classes[task].append(item)

    # 创建一个列表来存储抽取的 context
    sampled_data = []

    # 从每个任务类别中随机抽取指定数量的项
    for task, items in task_classes.items():
        sampled_items = random.sample(items, min(sample_size, len(items)))  # 确保不超过实际数量
        sampled_data.extend(sampled_items)

    return sampled_data


def call_gpt4_with_retry(task, gpt_instance, content, args={}, showkeys=False, max_retries=5):
    attempts = 0
    while attempts < max_retries:
        output = gpt_instance.call(content)
        try:
            if '{' != output[0]:
                output = find_bracket_content(output)
            output = json.loads(output)
            if 'error' in output:
                raise Exception(f"API Error: {output['error']}")
            if "arc" in task:
                if "Options" in output:
                    return output
            else:
                return output

        except json.JSONDecodeError:
            attempts += 1
            print(f"Attempt {attempts}/{max_retries} failed: JSON decode error. Retrying...")

    raise Exception("Failed to parse JSON response after maximum retries.")



def generate_select_data(item, gpt, task, prefix, ii, save_dir):
    contexts = item["context"]
    example_question = item["example_question"]
    input_text = ""
    if isinstance(contexts[0], dict):
        for i, it in enumerate(contexts):
            text = it["text"]
            input_text += f"[{i}]: {text}\n"
    else:
        input_text = "[1]: " + contexts[1] + "\n"
    prompt = Prompts[task[:-3]].format(doc=input_text, example=example_question)
    
    try:
        output = call_gpt4_with_retry(task, gpt, prompt)
        item["q*"] = output["q*"]
        item["a*"] = output["a*"]
        item["prompt"] = prompt
        question = output.get("q*", "")
        
        save_dir = save_dir + f'/{task}_{prefix}'  
        save_path = os.path.join(save_dir, f'output_{ii}.json')
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(item, f, ensure_ascii=False, indent=4)
        print(f"生成了第{ii}个数据，内容是{question}")
        
        return f"生成了第{ii}个数据，内容是{question}"
    
    except Exception as e:
        return f'处理失败: {e}'


import re
def find_bracket_content(s):
    # 使用正则表达式匹配从第一个'{'到最后一个'}'之间的所有内容，包括这两个字符
    match = re.search(r'\{.*\}', s, re.DOTALL)
    if match:
        return match.group(0)  # 返回找到的匹配内容
    return None  # 如果没有找到匹配，则返回None

        
def main(args):
    
    normal_data = select_from_data(args.data_path)
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        gpt = GPT4(model_name='gpt-4o')
        for i, item in enumerate(normal_data):
            task = item["task"]
            futures.append(executor.submit(generate_select_data, item, gpt, task, "v3", i, args.save_dir))
            
        for future in as_completed(futures):
            future.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/223040263/wanlong/data_generation/data_final/merged_output.json", help="Path to data for normal_from")
    parser.add_argument("--max_workers", type=int, default=40, help="Maximum number of workers for ThreadPoolExecutor")
    parser.add_argument("--save_dir", type=str, default="/223040263/wanlong/data_generation/data_final_v4", help="save_dir")

    args = parser.parse_args()
    main(args)
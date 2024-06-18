
import torch
from transformers import AutoTokenizer, DebertaForSequenceClassification 
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, GenerationConfig
from datasets import load_dataset, load_from_disk
import transformers
from transformers.pipelines.pt_utils import KeyDataset

import pandas as pd
from itertools import combinations, permutations
from typing import List

import random
from datasets import Dataset, DatasetDict
from datetime import datetime
import sys
import os
import logging
import random
random.seed(43) #43
from tabulate import tabulate
import openai
import time
from tqdm import tqdm
import anthropic
from safetensors.torch import load_model, save_model 
import shutil
import json
import subprocess

from datasets import Dataset, concatenate_datasets

#################################################

# Parameters

# Mixture of Agents Models
models = ["Qwen/Qwen1.5-72B-Chat", "Qwen/Qwen1.5-110B-Chat", "microsoft/WizardLM-2-8x22B",
          "mistralai/Mixtral-8x22B-Instruct-v0.1", "meta-llama/Llama-3-70b-chat-hf", "databricks/dbrx-instruct"]
MoA_models = models

models = ["Qwen/Qwen1.5-72B-Chat"]

# Generation Settings
generation_dict = {
    "batch_size": 8,
    "temperatures": [0.7], #0.9 #1.5
    "candidates_per_temp": [1],
    "generation_max_length": 512,
    "dataset_cutoff": 4, #3, None
    #"top_k": 10,
    #"top_p": 0.9
}

#################################################

for model_name in models:

    print(f"Generating candidates for model: {model_name}")
    
    model_id = model_name.split("/")[1]
    candidate_generation_command = f"python gen_model_answer.py --model-path {model_name} --model-id {model_id} --model-type TogetherAI --num_choices {generation_dict['candidates_per_temp'][0]}"

    print("Generation Command: ", candidate_generation_command)
    print("Generating candidates...")
    generation_result = subprocess.run(candidate_generation_command, shell=True, capture_output=True, text=True)
    #with subprocess.Popen(candidate_generation_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
    #    for line in process.stdout:
    #        print(line, end='')  # Print the output in real-time

    saved_jsonl_path = f"data/mt_bench/model_answer/{model_id}.jsonl"

    ##########################################

    judgement_command = f"python gen_judgment.py --model-list {model_name} --parallel 2"
    print("Generating judgements...")
    judgement_result = subprocess.run(judgement_command, shell=True, capture_output=True, text=True)
    #with subprocess.Popen(judgement_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
    #    for line in process.stdout:
    #        print(line, end='')  # Print the output in real-time

    ##########################################

    show_results_command = f"python show_result.py --model-list {model_name}"
    show_results_result = subprocess.run(show_results_command, shell=True, capture_output=True, text=True)
    #with subprocess.Popen(show_results_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
    #    for line in process.stdout:
    #        print(line, end='')  # Print the output in real-time



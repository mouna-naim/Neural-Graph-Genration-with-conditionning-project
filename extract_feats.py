import os
import re
from tqdm import tqdm
import random
from transformers import BertTokenizer, BertModel
import torch

random.seed(32)

bert_model_name = 'bert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name)

def extract_numbers_with_bert(text):
    numbers = re.findall(r'\d+\.\d+|\d+', text)
    stats = [float(num) for num in numbers]
    inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        bert_output = bert_model(**inputs)
    bert_embedding = bert_output.pooler_output.squeeze(0)
    return stats, bert_embedding.unsqueeze(0)


def extract_numbers(text):
    numbers = re.findall(r'\d+\.\d+|\d+', text)
    return [float(num) for num in numbers]

def extract_feats(file):
    with open(file, "r") as fread:
        line = fread.read().strip()
    stats = extract_numbers(line)
    inputs = bert_tokenizer(line, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        bert_output = bert_model(**inputs)
    bert_embedding = bert_output.pooler_output.squeeze(0)
    return stats, bert_embedding.unsqueeze(0)

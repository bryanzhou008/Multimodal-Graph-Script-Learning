import os
import sys
from info_nce import InfoNCE
infonce_loss_fct = InfoNCE(temperature=0.1, reduction='mean', negative_mode='paired')

import torchmetrics
from torchmetrics.text.bert import BERTScore
from torchmetrics import BLEUScore
from pprint import pprint

import evaluate
m_bert = evaluate.load("bertscore")
m_bleu = evaluate.load("bleu")


# file access and data structures
import json
import csv
import pandas as pd
from typing import Dict, List
import marisa_trie


# math/stats and operations
import numpy as np
from statistics import mean
import math
import random
import copy
from tqdm import tqdm


# ML libs
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import torch
import torch.nn.functional as F
from torch import nn

from sklearn.model_selection import train_test_split
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    TextDataset,
    DataCollatorForLanguageModeling,
    pipeline,
    Trainer, 
    TrainingArguments,
    BartForConditionalGeneration, 
    BartTokenizer,
    BartTokenizerFast,
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5TokenizerFast,
    
)

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Uses GPU if available

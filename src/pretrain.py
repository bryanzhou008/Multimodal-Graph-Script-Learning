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


def set_seed(seed = 42):
    """
    Set the random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class MarisaTrie(object):
    def __init__(
        self,
        sequences: List[List[int]] = [],
        cache_fist_branch=True,
        max_token_id=256001,
    ):

        self.int2char = [chr(i) for i in range(min(max_token_id, 55000))] + (
            [chr(i) for i in range(65000, max_token_id + 10000)]
            if max_token_id >= 55000
            else []
        )
        self.char2int = {self.int2char[i]: i for i in range(max_token_id)}

        self.cache_fist_branch = cache_fist_branch
        if self.cache_fist_branch:
            self.zero_iter = list({sequence[0] for sequence in sequences})
            assert len(self.zero_iter) == 1
            self.first_iter = list({sequence[1] for sequence in sequences})

        self.trie = marisa_trie.Trie(
            "".join([self.int2char[i] for i in sequence]) for sequence in sequences
        )

    def get(self, prefix_sequence: List[int]):
        if self.cache_fist_branch and len(prefix_sequence) == 0:
            return self.zero_iter
        elif (
            self.cache_fist_branch
            and len(prefix_sequence) == 1
            and self.zero_iter == prefix_sequence
        ):
            return self.first_iter
        else:
            key = "".join([self.int2char[i] for i in prefix_sequence])
            return list(
                {
                    self.char2int[e[len(key)]]
                    for e in self.trie.keys(key)
                    if len(e) > len(key)
                }
            )

    def __iter__(self):
        for sequence in self.trie.iterkeys():
            yield [self.char2int[e] for e in sequence]

    def __len__(self):
        return len(self.trie)

    def __getitem__(self, value):
        return self.get(value)

def gen_train_1_batch(Generator,gen_optimizer, tokenizer, batch_input_sequences, batch_output_sequences, mini_batch_size=2, accum_iter = 16):

    max_length = 1024
    loss_fn = nn.BCEWithLogitsLoss()

    

    for i in range(accum_iter):
        # get mini_batch
        start = i*mini_batch_size
        end = (i+1)*mini_batch_size
        input_sequences = batch_input_sequences[start:end]
        output_sequences = batch_output_sequences[start:end]
       

        # encode input data
        source_encoding = tokenizer(
            input_sequences,        # do not use task prefix
            padding="longest",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids, attention_mask = source_encoding.input_ids, source_encoding.attention_mask


        # encode target output
        target_encoding = tokenizer(
            [sequence for sequence in output_sequences],
            padding="longest", 
            max_length=max_length, 
            truncation=True,
            return_tensors="pt",
        )
        labels = target_encoding.input_ids
        

        # move everything to gpu
        input_ids=input_ids.cuda()
        attention_mask=attention_mask.cuda() 
        labels=labels.cuda()


        # forward pass
        out_with_labels = Generator(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels,
            output_hidden_states=True,
        )
        generator_loss_on_ground_truth = out_with_labels.loss

        

        loss = generator_loss_on_ground_truth / accum_iter

        loss.backward()

        if USE_WANDB == True:
            wandb.log({"generator_loss": loss})

    
    gen_optimizer.step()
    gen_optimizer.zero_grad()

def train_one_epoch(Generator, gen_optimizer, tokenizer, GROUNDED_DATA, WIKIHOW_INPUT,
                    mini_batch_size: int = 2, accum_iter: int = 16, train_steps: int = 400, num_epoch=0):
    
    '''this function implements the training loop for one epoch'''

    equivalent_batch_size = mini_batch_size * accum_iter


    Generator.train()
    


    # shuffle training data on every epoch
    c = list(zip(GROUNDED_DATA, WIKIHOW_INPUT))
    random.shuffle(c)
    GROUNDED_DATA, WIKIHOW_INPUT = zip(*c)



    # for each input sequence, find indices of all other same input sequences and randomly choose one
    # add these two sequences into the batch
    for i in tqdm(range(math.ceil(len(GROUNDED_DATA)/32))):
        input_sequences = []
        output_sequences = []
        contrastive_sequences = []
        for j in range(16):
            cur_index = i*16 + j        # the current index
            input = WIKIHOW_INPUT[cur_index]
            index_pos_list = [ i for i in range(len(WIKIHOW_INPUT)) if WIKIHOW_INPUT[i] == input ]  # find all indexes with the same input sequence as the current index
            other_index = random.choice(index_pos_list)     # randomly select one of them
            input_sequences += [WIKIHOW_INPUT[cur_index], WIKIHOW_INPUT[other_index]]
            output_sequences += [GROUNDED_DATA[cur_index], GROUNDED_DATA[other_index]]

        gen_train_1_batch(Generator, gen_optimizer, tokenizer, input_sequences, output_sequences, mini_batch_size, accum_iter)

def train_eval_loop(Generator, tokenizer, NEW_GROUNDED_DATA, WIKIHOW_INPUT_DATA, mini_batch_size: int = 2, accum_iter: int = 16,train_steps: int = 400, EPOCHS: int = 5):
    
    torch.cuda.empty_cache()


    # move the Generator to cuda
    Generator = Generator.cuda()

    # initialize gen optimizer with lr = e-4/e-5
    gen_optimizer = torch.optim.Adam(Generator.parameters(), lr=2e-5)


    for i in range(100):
        print("Epoch", i+1, ':')
        # train 1 epoch
  
        train_one_epoch(Generator, gen_optimizer, tokenizer, NEW_GROUNDED_DATA, WIKIHOW_INPUT_DATA, mini_batch_size = mini_batch_size, accum_iter = accum_iter, train_steps = train_steps, num_epoch=i)
        if (i+1) % 5 == 0:
            save_model(Generator, gen_optimizer, i+1)
    
    return Generator, gen_optimizer




def save_model(Generator, gen_optimizer, i):

    s = '/local1/bryanzhou008/Multimodal-Graph-Script-Learning/model/'+ MODEL_NAME + '_' + str(i) + 'epochs.tar'
    print("saving model to:",s)
    torch.save({
            'generator_state_dict': Generator.state_dict(),
            'generator_optimizer_state_dict': gen_optimizer.state_dict(),
            }, 
           s)

def main():

    # set random seed
    set_seed(42)

    # import models
    Generator = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    # tell the model that it needs to learn new tokens:
    bart_tokenizer.add_tokens("<->")
    Generator.resize_token_embeddings(len(bart_tokenizer))


    with open("/local1/bryanzhou008/Multimodal-Graph-Script-Learning/data/ht100m_wikihow_input_sequences.json", "r") as fp:
        WIKIHOW_INPUT_DATA = json.load(fp)
    with open("/local1/bryanzhou008/Multimodal-Graph-Script-Learning/data/ht100m_wikihow_output_sequences.json", "r") as fp:
        SEP_GROUNDED_DATA = json.load(fp)

    too_short = [i for i in range(len(SEP_GROUNDED_DATA)) if len(SEP_GROUNDED_DATA[i]) < 4]

    input_sequences = [WIKIHOW_INPUT_DATA[i] for i in range(len(WIKIHOW_INPUT_DATA)) if i not in too_short]
    output_sequences = [SEP_GROUNDED_DATA[i] for i in range(len(SEP_GROUNDED_DATA)) if i not in too_short]

    INPUT_WITH_SEP = []
    for line in input_sequences:
        line_with_sep = []
        for sent in line:
            line_with_sep.append((sent + " <->"))
        joined_recipe = ' '.join(line_with_sep)
        INPUT_WITH_SEP.append(joined_recipe)

    OUTPUT_WITH_SEP = []
    for line in output_sequences:
        line_with_sep = []
        for sent in line:
            line_with_sep.append((sent + " <->"))
        joined_recipe = ' '.join(line_with_sep)
        OUTPUT_WITH_SEP.append(joined_recipe)
    

    Generator, gen_optimizer = train_eval_loop(Generator, bart_tokenizer, OUTPUT_WITH_SEP, INPUT_WITH_SEP, mini_batch_size = 2, accum_iter = 16, train_steps = 40000, EPOCHS = 200)




if __name__ == '__main__':
    main()
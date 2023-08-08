# file access and data structures
import json
import csv
import pandas as pd
from typing import Dict, List
import marisa_trie
import os

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
import json
import graphviz
import logging
# logging.basicConfig(format='[%(levelname)s@%(name)s] %(message)s', level=logging.DEBUG)
# graphviz.__version__, graphviz.version()




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

def generate_paths(Generator, gen_tokenizer, WIKIHOW_INPUT_DATA, WIKIHOW_REFERENCE_GROUP, NUM_BEAMS = 20):
    '''this function does some generation on the validation set'''
    Generator.eval()
    max_input_length = 512
    max_output_length = 100


    SAMPLED_PATHS = []

    for i in tqdm(first_appearances):
        input_sequences = WIKIHOW_INPUT_DATA[i]
        choices = WIKIHOW_REFERENCE_GROUP[i]

        tokenized = gen_tokenizer(input_sequences, max_length=max_input_length, return_tensors="pt")

        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        # if USE_GPU == True:
        #     input_ids = input_ids.cuda()
        #     attention_mask = attention_mask.cuda()

        # construct marisa trie to store choices
        paths = []
        for choice in choices:
            encoded_sent = gen_tokenizer.encode(choice)     # do not add a 2 in the begining
            paths.append(encoded_sent)
        trie = MarisaTrie(paths)

        # print("choices:", choices)
        # print("paths:", paths)


        def cbs(batch_id, sent):
            '''given a input prefix sequence, this function returns all the possible next words in a list'''

            '''for bart model: "<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, ".": 4'''


            START_TOKEN = 0     # "<s>"
            SEP_TOKEN = 50265
            END_TOKEN = 2       # "</s>"
            nonlocal trie, paths  # this makes trie refer to the outside variable trie

            # print("sent size:",sent.size())

            # first convert the input tensor to list
            token_list = sent.tolist()


            # # remove all the start tokens generated in the middle of sentence 治标不治本
            while START_TOKEN in token_list[2:]:
                temp = token_list[2:]
                temp.remove(START_TOKEN)
                token_list[2:] = temp

            while 1 in token_list[2:]:
                temp = token_list[2:]
                temp.remove(1)
                token_list[2:] = temp

            while 2 in token_list[2:]:
                temp = token_list[2:]
                temp.remove(2)
                token_list[2:] = temp

            while 3 in token_list[2:]:
                temp = token_list[2:]
                temp.remove(3)
                token_list[2:] = temp

            



            # since the model always generates 2 as the first argument, we manually remove it every time
            if token_list == [2]:
                # print("reached start of sequence once!")
                results = START_TOKEN
                return results
            else:
                token_list = token_list[1:]


            # if the current last token is a period, then generate on a start sentence token (0) or end the generation with [1]
            if token_list[-1] == SEP_TOKEN:
                # now use the new tree to get results
                results = trie.get([START_TOKEN]) + [END_TOKEN]
                # results = trie.get([START_TOKEN])

                
            # if we are in the middle of generating the ith sentence (i>1), cut off the part from previous sentences
            elif SEP_TOKEN in token_list:
                res = len(token_list) - 1 - token_list[::-1].index(SEP_TOKEN)    # get the position of the last sep token
                prefix = [START_TOKEN] + token_list[res+1:]
                results = trie.get(prefix)


            # if we are generating the first sentence
            else:
                # print("at option 3")
                results = trie.get(token_list)

            # in the case where the model chose a token outside of our plans, end the generation here to save time
            if results == []:
                results = [END_TOKEN]


            return results



        outputs = Generator.generate(
            input_ids, 
            attention_mask=attention_mask,
            max_length=max_output_length,
            num_beams=NUM_BEAMS,
            num_return_sequences=NUM_BEAMS,
            prefix_allowed_tokens_fn = cbs,
        )

        # print("outputs:",outputs)



        # print the results
        if NUM_BEAMS == 1:
            decoded_outputs = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("input:", input_sequences)
            print("output:", decoded_outputs)
            print("output # sentences:", decoded_outputs.count('<->'))
            print('\n')
        else:
            decoded_outputs = gen_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            print("input:", input_sequences)
            for decoded_output in decoded_outputs:
                print("output:", decoded_output)
                print("output # sentences:", decoded_output.count('<->'))
            print('\n')

        SAMPLED_PATHS.append(decoded_outputs)

    return SAMPLED_PATHS






if __name__ == '__main__':

    NUM_BEAMS = 40
    THRESHOLD = 4
    DECIMAL_EDGE_WEIGHT = True
    MODEL_CHECKPOINT = "/content/drive/MyDrive/BLENDER/crosstask/model/dec_data/v3_new_data_linear_80epochs.tar"
    OUTPUT_DIR = '/content/drive/MyDrive/BLENDER/crosstask/out/graphs/Dec18_linear_model/'
    os.mkdir(OUTPUT_DIR)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Uses GPU if available



    Generator = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    # we now tell the model that it needs to learn new tokens:
    bart_tokenizer.add_tokens("<->")
    Generator.resize_token_embeddings(len(bart_tokenizer))

    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=torch.device('cpu'))
    Generator.load_state_dict(checkpoint['generator_state_dict'])



    with open("/content/drive/MyDrive/BLENDER/crosstask/data/nov_25/REFERENCE_GROUP_FOR_CHOICES", "r") as fp:
        WIKIHOW_REFERENCE_GROUP = json.load(fp)
    with open("/content/drive/MyDrive/BLENDER/crosstask/data/nov_25/CROSSTASK_INPUT_DATA", "r") as fp:
        WIKIHOW_INPUT_DATA = json.load(fp)
    with open("/content/drive/MyDrive/BLENDER/crosstask/data/nov_25/SEP_GROUNDED_DATA", "r") as fp:
        SEP_GROUNDED_DATA = json.load(fp)


    val_dict = {}
    for i in range(len(WIKIHOW_INPUT_DATA)):
        input = WIKIHOW_INPUT_DATA[i]
        if input not in val_dict or val_dict[input] == None:
            val_dict[input] = [SEP_GROUNDED_DATA[i]]
        else:
            print("exist:", val_dict[input])
            val_dict[input].append(SEP_GROUNDED_DATA[i])

    WIKIHOW_INPUT_DATA_nodup = [*set(WIKIHOW_INPUT_DATA)]
    first_appearances = [WIKIHOW_INPUT_DATA.index(x) for x in WIKIHOW_INPUT_DATA_nodup]



    SAMPLED_PATHS = generate_paths(Generator, bart_tokenizer, WIKIHOW_INPUT_DATA, WIKIHOW_REFERENCE_GROUP, NUM_BEAMS = NUM_BEAMS)
    GRAPH_REFERENCE_GROUP = [WIKIHOW_REFERENCE_GROUP[i] for i in first_appearances]



    # split based on '<->', remove the space at begining and/or end
    PROCESSED_SAMPLED_PATHS = []

    for paths in SAMPLED_PATHS:
        new_paths = []
        for p in paths:
            new_p = []
            temp = p.split('<->')[:-1]
            for sent in temp:
                if sent[0] == ' ':
                    sent = sent[1:]
                if sent[-1] == ' ':
                    sent = sent[:-1]
                new_p.append(sent)
            new_paths.append(new_p)
        PROCESSED_SAMPLED_PATHS.append(new_paths)

    # check if there could be more than one match for each sentence
    for i in range(len(GRAPH_REFERENCE_GROUP)):
        paths = PROCESSED_SAMPLED_PATHS[i]
        choices = GRAPH_REFERENCE_GROUP[i]
        # print(paths)
        # print(choices)


        for p in paths:
            for sent in p:
                # print(sent)
                FOUND = 0
                for s in choices:
                    if sent in s and (abs(len(s) - len(sent)) < 10):
                        FOUND += 1
                if FOUND != 1:
                    print("epic fail")
                    print(sent)
                    print(choices) 

    MATCHED_PATHS = []

    for i in range(len(GRAPH_REFERENCE_GROUP)):
        paths = PROCESSED_SAMPLED_PATHS[i]
        choices = GRAPH_REFERENCE_GROUP[i]

        GROUP = []
        for p in paths:
            PATH = []
            for sent in p:
                SENT = ''
                for j in range(len(choices)):
                    s = choices[j]
                    if sent in s and (abs(len(s) - len(sent)) < 10):
                        # if I want to remove the last <->, do it here
                        SENT = s[:-5]
                if SENT == -1:
                    print("epic fail")
                    print(sent)
                    print(choices) 
                else:
                    PATH.append(SENT)
            GROUP.append(PATH)
        MATCHED_PATHS.append(GROUP)


    # only when edge num > THRESHOLD, a pair is kept
    COUNTED_PROCESSED_SAMPLED_PAIRS = []

    for group in MATCHED_PAIRS:
        COUNTED_GROUP = []
        for pair in group:
            if group.count(pair) > THRESHOLD and len(pair[0]) > 5 and len(pair[1])>5:
                if DECIMAL_EDGE_WEIGHT:
                    COUNTED_GROUP.append(pair + (group.count(pair)/NUM_BEAMS,))
                else:
                    COUNTED_GROUP.append(pair + (group.count(pair),))
        # remove duplicates
        COUNTED_GROUP = [*set(COUNTED_GROUP)]
        COUNTED_PROCESSED_SAMPLED_PAIRS.append(COUNTED_GROUP)



    # generate graph
    for i in tqdm(range(len(COUNTED_PROCESSED_SAMPLED_PAIRS))):
    graph_name = str(i)

    f = graphviz.Digraph(graph_name, comment='test graph', format='png', node_attr={'shape': 'rectangle', 'color': 'lightblue2', 'style':"rounded,filled", 'fontsize': "16"})

    for edge_tuple in COUNTED_PROCESSED_SAMPLED_PAIRS[i]:
        f.node(edge_tuple[0])
        f.node(edge_tuple[1])

        f.edge(edge_tuple[0], edge_tuple[1], label=str(edge_tuple[2]), fontcolor = 'forestgreen')
        # if edge_tuple[2] < 0.2:
        #     f.edge(edge_tuple[0], edge_tuple[1], label=str(edge_tuple[2]), style='dashed', fontsize="12", fontcolor = 'forestgreen')
        # else:
        #     f.edge(edge_tuple[0], edge_tuple[1], label=str(edge_tuple[2]), fontcolor = 'forestgreen')


    f.render(directory=OUTPUT_DIR, view=True)

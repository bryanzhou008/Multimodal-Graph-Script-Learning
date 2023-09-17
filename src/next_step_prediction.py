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

def shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def generate_halfway(Generator, gen_tokenizer, input_sequences, first_half, choices):
    Generator.eval()
    max_input_length = 512
    max_output_length = 200


    tokenized = gen_tokenizer(input_sequences, max_length=max_input_length, return_tensors="pt")

    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask

    paths = []
    for choice in choices:
        encoded_sent = gen_tokenizer.encode(choice)     # do not add a 2 in the begining
        paths.append(encoded_sent)
    trie = MarisaTrie(paths)



    first_half_list = [gen_tokenizer.encode(first_half)]
    first_half_trie = MarisaTrie(first_half_list)
    


    first_half_len = len(gen_tokenizer.encode(first_half))


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

            # if token_list[-1] == 38:
            # print(token_list)


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


            if (len(token_list) < first_half_len-1):
                results = first_half_trie.get(token_list)  
            elif token_list[-1] == SEP_TOKEN:
                # now use the new tree to get results
                results = trie.get([START_TOKEN]) + [END_TOKEN]
            elif SEP_TOKEN in token_list:
                res = len(token_list) - 1 - token_list[::-1].index(SEP_TOKEN)
                prefix = [START_TOKEN] + token_list[res+1:]
                results = trie.get(prefix)

            # in the case where the model chose a token outside of our plans, end the generation here to save time
            if results == []:
                # if DEBUG == True:
                # print("results == []")
                # print("sent:",sent)
                results = [END_TOKEN]

            # if results == [SEP_TOKEN]:
                # print("result is:",SEP_TOKEN)

            if(START_TOKEN in results):
                print("why is start token in results???")

            if 1 in results:
                print("why is the padding token in results?????")
            return results

    outputs = Generator.generate(
            input_ids, 
            attention_mask=attention_mask,
            max_length=max_output_length,
            num_beams=1,
            prefix_allowed_tokens_fn = cbs,
        )

    decoded_outputs = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return decoded_outputs

def generate_paths(Generator, gen_tokenizer, WIKIHOW_INPUT_DATA, WIKIHOW_REFERENCE_GROUP, NUM_BEAMS = 20):
    '''this function does some generation on the validation set'''
    Generator.eval()
    max_input_length = 512
    max_output_length = 100

    SAMPLED_PATHS = []

    for i in tqdm(first_appearances):
        input_sequences = WIKIHOW_INPUT_DATA[i]
        choices = WIKIHOW_REFERENCE_GROUP[i]
        # choices = [j[:-1] for j in WIKIHOW_REFERENCE_GROUP[i]]


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

        print("choices:", choices)
        print("paths:", paths)


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

            # if token_list[-1] == 38:
            # print(token_list)


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


            # print(sent)

            # if the current last token is a period, then generate on a start sentence token (0) or end the generation with [1]
            if token_list[-1] == SEP_TOKEN:
                # now use the new tree to get results
                results = trie.get([START_TOKEN]) + [END_TOKEN]
                # results = trie.get([START_TOKEN])

                
            # if we are in the middle of generating the ith sentence (i>1), cut off the part from previous sentences
            elif SEP_TOKEN in token_list:
                # print("at option 2")
                # print("sep here")
                res = len(token_list) - 1 - token_list[::-1].index(SEP_TOKEN)    # get the position of the last sep token
                # print("res is:", res)
                prefix = [START_TOKEN] + token_list[res+1:]
                # print("prefix is:",prefix)
                results = trie.get(prefix)
                # print(choices)

            # if we are generating the first sentence
            else:
                # print("at option 3")
                results = trie.get(token_list)

            # in the case where the model chose a token outside of our plans, end the generation here to save time
            if results == []:
                # if DEBUG == True:
                # print("results == []")
                # print("sent:",sent)
                results = [END_TOKEN]

            # if results == [SEP_TOKEN]:
                # print("result is:",SEP_TOKEN)


            if(START_TOKEN in results):
                print("why is start token in results???")

            if 1 in results:
                print("why is the padding token in results?????")

            # if 38 in results:
            #     print('why is 38 in results???')
            # print(results, '\n')

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



def main():

    # set random seed
    set_seed(42)


    # load model
    Generator = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    bart_tokenizer.add_tokens("<->")
    Generator.resize_token_embeddings(len(bart_tokenizer))
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=torch.device('cpu'))
    Generator.load_state_dict(checkpoint['generator_state_dict'])


    # load data
    
    with open("Multimodal-Graph-Script-Learning/data//NSP_Dataset/INPUT_SENTS", "w") as fp:
        INPUT_SENTS = json.load(fp)

    with open("Multimodal-Graph-Script-Learning/data/NSP_Dataset/CHOICES_SENTS", "w") as fp:
        CHOICES_SENTS = json.load(fp)

    with open("Multimodal-Graph-Script-Learning/data/NSP_Dataset/CORRECT_CHOICES", "w") as fp:
        CORRECT_CHOICES = json.load(fp)

    with open("Multimodal-Graph-Script-Learning/data/NSP_Dataset/CORRECT_RATES", "w") as fp:
        CORRECT_RATES = json.load(fp)

    with open("Multimodal-Graph-Script-Learning/data/NSP_Dataset/STEP_LIBRARY", "r") as fp:
        STEP_LIBRARY = json.load(fp)
        
    
    # Multimedia Model
    count = 0
    correct_count = 0

    for i in tqdm(range(len(INPUT_SENTS))):
        step_library = " <-> ".join(STEP_LIBRARY[i])
        first_half = " <-> ".join(INPUT_SENTS[i]) + " <->"
        available_steps = [x + " <-> " for x in CHOICES_SENTS[i]]
        correct_steps = CORRECT_CHOICES[i]

        outputs = generate_halfway(Generator, bart_tokenizer, step_library, first_half, available_steps)
        try:
            result = (outputs.split(" <-> "))[len(INPUT_SENTS[i])]
            if result in correct_steps:
                correct_count += 1
            count += 1
            print('Model Choice:', result)
            print("Input steps:",first_half)
            print("Choices:", available_steps)
            print("Correct Choices:", correct_steps)
            print("Accuracy:",correct_count/count)
            print("\n")
        except:
            count += 1
            print("Model Output:", outputs.split(" <-> "))
            print("Input steps:",first_half)
            print("Choices:", available_steps)
            print("Correct Choices:", correct_steps)
            print("Accuracy:",correct_count/count)
            print("\n")


    print("Final Acc:", correct_count/count)



if __name__ == '__main__':
    main()
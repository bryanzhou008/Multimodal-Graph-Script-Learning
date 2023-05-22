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


def save_model(Generator, gen_optimizer, i):
    SAVE_FILE = 'Multimodal-Graph-Script-Learning/model' + MODEL_NAME + '_' + str(i) + 'epochs.tar'
    print("saving model to:", SAVE_FILE)
    torch.save({
            'generator_state_dict': Generator.state_dict(),
            'generator_optimizer_state_dict': gen_optimizer.state_dict(),
            }, 
           SAVE_FILE)

def gen_train_1_batch(Generator, gen_optimizer, tokenizer, task_prefix, batch_input_sequences, batch_output_sequences, mini_batch_size=2, accum_iter = 16):

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
            # [task_prefix + sequence for sequence in input_sequences],
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

def contrastive_train_1_batch(Generator, gen_optimizer, tokenizer, task_prefix, batch_input_sequences, batch_output_sequences, batch_contrastive_sequences, mini_batch_size=2, accum_iter = 16):

    max_input_length = 1024
    max_output_length = 512
    loss_fn = nn.BCEWithLogitsLoss()

    

    for i in range(accum_iter):
        # get mini_batch
        start = i*mini_batch_size
        end = (i+1)*mini_batch_size
        input_sequences = batch_input_sequences[start:end]
        output_sequences = batch_output_sequences[start:end]
        contrastive_sequences = batch_contrastive_sequences[start:end]

       
        # ----------------------------------------------InfoNCE contrastive loss----------------------------------------------
        
        # infonce_loss = torch.zeros([1], device='cuda')

        for i in range(mini_batch_size):
            # torch.cuda.empty_cache()        # add this line to solve cuda out of memory
            all_neg_tgt_ids = []
            for neg_example in contrastive_sequences[i]:
                contrastive_encoding = tokenizer(
                    neg_example,
                    padding="longest", 
                    max_length=max_output_length, 
                    truncation=True,
                    return_tensors="pt",
                )
                all_neg_tgt_ids.append(contrastive_encoding.input_ids)


            # just for the fun of it can I use the other true example as the query
            query_example = output_sequences[i]
            query_tgt_ids = tokenizer(
                query_example,
                padding="longest", 
                max_length=max_output_length, 
                truncation=True,
                return_tensors="pt",
            ).input_ids


            pos_example = output_sequences[1-i]
            pos_tgt_ids = tokenizer(
                pos_example,
                padding="longest", 
                max_length=max_output_length, 
                truncation=True,
                return_tensors="pt",
            ).input_ids

            

            source_encoding = tokenizer(
                input_sequences[i],
                padding="longest",
                max_length=max_input_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids, attention_mask = source_encoding.input_ids, source_encoding.attention_mask



            # convert to decoder ids by shifting tokens right
            query_decoder_input_ids = shift_tokens_right(query_tgt_ids, Generator.config.pad_token_id, Generator.config.decoder_start_token_id)
            pos_decoder_input_ids = shift_tokens_right(pos_tgt_ids, Generator.config.pad_token_id, Generator.config.decoder_start_token_id)
            all_neg_decoder_input_ids = [shift_tokens_right(neg_tgt_ids, Generator.config.pad_token_id, Generator.config.decoder_start_token_id) for neg_tgt_ids in all_neg_tgt_ids]

            # move everything to cuda
            input_ids=input_ids.cuda()
            attention_mask=attention_mask.cuda() 
            pos_decoder_input_ids = pos_decoder_input_ids.cuda()
            query_decoder_input_ids = query_decoder_input_ids.cuda()
            for i in range(len(all_neg_decoder_input_ids)):
                all_neg_decoder_input_ids[i] = all_neg_decoder_input_ids[i].cuda()


            # generate the three types of outputs
            # note that outputs[1][-1] gives us the last of decoder_hidden_states
            # for now I am using free form generation to generate anchor embeddings
            # freeform_outputs = Generator(input_ids, attention_mask=attention_mask, use_cache=False, output_hidden_states=True)
            # query = torch.mean(freeform_outputs[1][-1], dim=1)


            # try query outputs with the other representation
            query_outputs = Generator(input_ids, attention_mask=attention_mask, decoder_input_ids=query_decoder_input_ids, use_cache=False, output_hidden_states=True)
            query_embeddings = torch.mean(query_outputs[1][-1], dim=1)

            # here query_outputs[1][-1] is the last decoder hiden layer
            # print("query_outputs:",query_outputs)
            # print("query_outputs[1] len:", len(query_outputs[1]))
            # print("query_outputs[1][-1].size():", query_outputs[1][-1].size())
            # print("query_embeddings:", query_embeddings.size())

            pos_outputs = Generator(input_ids, attention_mask=attention_mask, decoder_input_ids=pos_decoder_input_ids, use_cache=False, output_hidden_states=True)
            positive_embeddings = torch.mean(pos_outputs[1][-1], dim=1)

            all_neg_outputs = [Generator(input_ids, attention_mask=attention_mask, decoder_input_ids=neg_decoder_input_ids, use_cache=False, output_hidden_states=True) for neg_decoder_input_ids in all_neg_decoder_input_ids]
            all_neg_outputs = [torch.mean(neg_outputs[1][-1], dim=1) for neg_outputs in all_neg_outputs]
            negative_embeddings = torch.stack(all_neg_outputs, dim=1)


            infonce_loss_fct = InfoNCE(temperature=0.1, reduction='mean', negative_mode='paired')
            infonce_loss = 0.1 * (infonce_loss_fct(query_embeddings, positive_embeddings, negative_embeddings)/mini_batch_size) # account for minibatch size

            loss = infonce_loss / accum_iter
            loss.backward()

        
            if USE_WANDB == True:
                wandb.log({"infonce_loss": infonce_loss})

    
    gen_optimizer.step()
    gen_optimizer.zero_grad()

def train_one_epoch(Generator, gen_optimizer, tokenizer, task_prefix, GROUNDED_CONTRASTIVE_DATA, GROUNDED_DATA, WIKIHOW_INPUT,
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
    for i in tqdm(range(math.ceil(len(WIKIHOW_INPUT)/32))):
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
            contrastive_sequences += [GROUNDED_CONTRASTIVE_DATA[cur_index], GROUNDED_CONTRASTIVE_DATA[other_index]]

        gen_train_1_batch(Generator,gen_optimizer, tokenizer, task_prefix, input_sequences, output_sequences, mini_batch_size, accum_iter)
        contrastive_train_1_batch(Generator,gen_optimizer, tokenizer, task_prefix, input_sequences, output_sequences, contrastive_sequences, mini_batch_size, accum_iter)

def train_eval_loop(Generator, tokenizer, SEP_GROUNDED_CONTRASTIVE_DATA, NEW_GROUNDED_DATA, WIKIHOW_INPUT_DATA, WIKIHOW_REFERENCE_GROUP, mini_batch_size: int = 2, accum_iter: int = 16,train_steps: int = 400, EPOCHS: int = 5):
    
    torch.cuda.empty_cache()
    # choose task prefix
    task_prefix = "reorder: "

    

    val_indices = [WIKIHOW_INPUT_DATA.index(i) for i in [*set(WIKIHOW_INPUT_DATA)]] 
    WIKIHOW_INPUT_VAL = [WIKIHOW_INPUT_DATA[i] for i in val_indices]
    GROUNDED_DATA_VAL = [NEW_GROUNDED_DATA[i] for i in val_indices]
    WIKIHOW_REFERENCE_GROUP_VAL = [WIKIHOW_REFERENCE_GROUP[i] for i in val_indices]



    
    # print(len(GROUNDED_DATA_VAL))
    # print(len(GROUNDED_DATA_TRAIN))
    # len(GROUNDED_DATA_TRAIN) == 795
    
    # if we want to split the dataset with similar samples on the same side


    # move the Generator and Discriminator to cuda
    Generator = Generator.cuda()


    # initialize gen optimizer with lr = e-4/e-5
    gen_optimizer = torch.optim.Adam(Generator.parameters(), lr=2e-5)
    dis_optimizer = torch.optim.Adam(Discriminator.parameters(), lr=2e-5)


    # check performence once before any training
    # evaluate(Generator, Discriminator, tokenizer, task_prefix, GROUNDED_DATA_VAL, WIKIHOW_INPUT_VAL, WIKIHOW_REFERENCE_GROUP_VAL)

    best_score = 0
    
        

    for i in range(EPOCHS):
        print("Epoch", i+1, ':')
        # train 1 epoch
  
        train_one_epoch(Generator, gen_optimizer, tokenizer, task_prefix, SEP_GROUNDED_CONTRASTIVE_DATA, NEW_GROUNDED_DATA, WIKIHOW_INPUT_DATA, mini_batch_size = mini_batch_size, accum_iter = accum_iter, train_steps = train_steps, num_epoch=i)
        
        # if i % 10 == 0:
        # new_score = evaluate(Generator, Discriminator, tokenizer, task_prefix, GROUNDED_DATA_VAL, WIKIHOW_INPUT_VAL, WIKIHOW_REFERENCE_GROUP_VAL)

        # if new_score > 0.3 and new_score > best_score:
        #     best_score = new_score
        #     save_model(Generator, gen_optimizer, Discriminator, dis_optimizer)
        #     with open('/content/drive/MyDrive/BLENDER/contrastive_learning/model/' + MODEL_NAME + '.txt', 'a') as f:
        #         f.write('current saved best performace is: ' + str(new_score) + '\n')
        if i % 5 == 0:
            save_model(Generator, gen_optimizer, i)
            

        
    
    # print out the results
    # test(Generator, tokenizer, task_prefix, GROUNDED_DATA_VAL, WIKIHOW_INPUT_VAL, WIKIHOW_REFERENCE_GROUP_VAL)


    return Generator, gen_optimizer

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
    with open("Multimodal-Graph-Script-Learning/data/crosstask_wikihow_reference_step_libraries.json", "r") as fp:
        WIKIHOW_REFERENCE_GROUP = json.load(fp)
    with open("Multimodal-Graph-Script-Learning/data/crosstask_wikihow_input_sequences.json", "r") as fp:
        WIKIHOW_INPUT_DATA = json.load(fp)
    with open("Multimodal-Graph-Script-Learning/data/crosstask_wikihow_output_sequences.json", "r") as fp:
        SEP_GROUNDED_DATA = json.load(fp)
    with open("Multimodal-Graph-Script-Learning/data/crosstask_wikihow_contrastive_sequences.json" + CONTRASTIVE_METHOD, "r") as fp:
        SEP_GROUNDED_CONTRASTIVE_DATA = json.load(fp)

    
    for group in WIKIHOW_REFERENCE_GROUP:
        for i in range(len(group)):
            group[i] = group[i] + ' '

    val_dict = {}
    for i in range(len(WIKIHOW_INPUT_DATA)):
        input = WIKIHOW_INPUT_DATA[i]
        if input not in val_dict or val_dict[input] == None:
            val_dict[input] = [SEP_GROUNDED_DATA[i]]
        else:
            val_dict[input].append(SEP_GROUNDED_DATA[i])
    

    Generator, gen_optimizer, Discriminator, dis_optimizer = train_eval_loop(Generator, Discriminator, bart_tokenizer, SEP_GROUNDED_CONTRASTIVE_DATA, SEP_GROUNDED_DATA, WIKIHOW_INPUT_DATA, WIKIHOW_REFERENCE_GROUP, mini_batch_size = 2, accum_iter = 16, train_steps = 400, EPOCHS = 200)




if __name__ == '__main__':
    main()
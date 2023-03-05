from torchtext.vocab import build_vocab_from_iterator
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Union, List, Tuple, Iterable
from torch.utils.data import Dataset
from trans_former import Transformer, create_mask
import math
from my_dataset import TextDataset


# I use:
# https://pytorch.org/tutorials/beginner/translation_transformer.html
# https://www.analyticsvidhya.com/blog/2021/06/language-translation-with-transformer-in-python/
# code's of seminar 8 "Deep Learning HSE"

train_dataset = pd.DataFrame()
with open(f'data/train.de-en.de') as file:
    train_dataset['de'] = file.readlines()
with open(f'data/train.de-en.en') as file:
    train_dataset['en'] = file.readlines()

test_dataset = pd.DataFrame()
with open(f'data/val.de-en.de') as file:
    test_dataset['de'] = file.readlines()
with open(f'data/val.de-en.en') as file:
    test_dataset['en'] = file.readlines()

token_transform = {}
vocab_transform = {}

def get_tokenizing(text):
    return text.split()

token_transform['en'] = get_tokenizing
token_transform['de'] = get_tokenizing

def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([2]),
                      torch.tensor(token_ids),
                      torch.tensor([3])))

def yield_tokens(data_iter: Iterable, language: str) -> List[str]:

    for data_sample in data_iter[language]:
        yield token_transform[language](data_sample)


vocab_transform['en'] = build_vocab_from_iterator(yield_tokens(train_dataset, 'en'), min_freq=1, specials=['<unk>', '<pad>', '<bos>', '<eos>'], special_first=True)
vocab_transform['de'] = build_vocab_from_iterator(yield_tokens(train_dataset, 'de'), min_freq=1, specials=['<unk>', '<pad>', '<bos>', '<eos>'], special_first=True)
vocab_transform['en'].set_default_index(0)
vocab_transform['de'].set_default_index(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device is: ', device)

torch.manual_seed(0)
transformer =  Transformer(3, 3, 512, 8, len(vocab_transform['de']), len(vocab_transform['en']), 512)
for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
transformer = transformer.to(device)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=1)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func


def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([2]),
                      torch.tensor(token_ids),
                      torch.tensor([3])))

text_transform = {'en':  sequential_transforms(token_transform['en'], vocab_transform['en'], tensor_transform),
                  'de':  sequential_transforms(token_transform['de'], vocab_transform['de'], tensor_transform)}


def apply_func(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform['de'](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform['en'](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=1)
    tgt_batch = pad_sequence(tgt_batch, padding_value=1)
    return src_batch, tgt_batch

def train_epoch(model, dataset, device, optimizer=None, training=False):
    
    if training:
        model.train()
    else:
        model.eval()
    losses = 0
    temp = TextDataset(dataset)
    train_dataloader = DataLoader(temp, batch_size=128, collate_fn=apply_func)
    
    cnt = 0
    for src, tgt in train_dataloader:
        cnt += 1
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device)
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        
        if training:
            optimizer.zero_grad()
        
        tgt_out = tgt[1:, :]
        loss = torch.nn.CrossEntropyLoss(ignore_index=1)(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        
        if training:
            loss.backward()
            optimizer.step()
            
        losses += loss.item()

    return losses / cnt

NUM_EPOCHS = 10

for epoch in range(1, NUM_EPOCHS+1):
    train_loss = train_epoch(transformer, train_dataset,  device, optimizer=optimizer, training=True)
    val_loss = train_epoch(transformer, test_dataset,  device, optimizer=None, training=False)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}"))


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        memory = memory.to(device)
        tgt_mask = (torch.triu(torch.ones((ys.size(0), ys.size(0)), device=device)) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        tgt_mask = (tgt_mask.type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.layer(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == 3:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform['de'](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=2).flatten()
    return " ".join(vocab_transform['en'].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

val = []
with open(f'data/test1.de-en.de') as file:
    val = file.readlines()

f = open('prediciton.txt', 'w')
all_f = ''
for line in val:
    all_f += translate(transformer, line).lower()+'\n'
f.write(all_f)
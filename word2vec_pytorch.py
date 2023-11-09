#!/usr/bin/python3

# word2vec reference: https://github.com/OlgaChernytska/word2vec-pytorch

# Usage: word2vec.py [-h] [-c configfile]

# Import builtins
import argparse
import csv
import functools
import json
import os
import random

# Import external packages
import numpy
import torch
import torch.nn
import torch.utils.data
import torchdata.datapipes
import torchtext.data
import torch.optim
import torch.optim.lr_scheduler
import torchtext.vocab

import bs4
import nltk

parser = argparse.ArgumentParser(description="An implementation of word2vec using PyTorch",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-c", "--config", type=str, help="file containing configurations", default="config.json")
args = parser.parse_args()

with open(args.config, "r", encoding="UTF-8") as conf:
    config = json.load(conf)
    
CBOW_N_WORDS = config["CBOW_N_WORDS"]
SKIPGRAM_N_WORDS = config["SKIPGRAM_N_WORDS"]
MIN_WORD_FREQUENCY = config["MIN_WORD_FREQUENCY"]
MAX_SEQUENCE_LENGTH = config["MAX_SEQUENCE_LENGTH"]
EMBED_DIMENSION = config["EMBED_DIMENSION"]
EMBED_MAX_NORM = config["EMBED_MAX_NORM"]

tokenizer = nltk.tokenize.RegexpTokenizer(r'[a-zA-Z0-9-]+').tokenize

# Step 0: Parse XML files and write abstract to TSV
if config["filetype"] == 'pubmedxml':
    files = list()
    with open(config["filelist"], "r") as filelist:
        files = filelist.read().splitlines()
    
    training = set(random.sample(files, round(len(files) * config["trainingratio"])))
    valid = set(files) - training
    
    outname = config["trainingcorpus"]
    for file in training:
        ## Read XML
        with open(file, "r", encoding="UTF-8") as xmlfile:
            soup = bs4.BeautifulSoup(xmlfile, 'xml')
        try:
            abstracttext = soup.AbstractText.text
            tokens = nltk.tokenize.RegexpTokenizer(r'[a-zA-Z0-9-]+').tokenize(abstracttext)
        except:
            abstracttext = ""
            tokens = list()
        with open(outname, 'a') as outfile:
            writer = csv.writer(outfile, dialect='excel-tab')
            writer.writerow(tokens)
    
    outname = config["validationcorpus"]
    for file in valid:
        ## Read XML
        with open(file, "r", encoding="UTF-8") as xmlfile:
            soup = bs4.BeautifulSoup(xmlfile, 'xml')
        try:
            abstracttext = soup.AbstractText.text
            tokens = nltk.tokenize.RegexpTokenizer(r'[a-zA-Z0-9-]+').tokenize(abstracttext)
        except:
            abstracttext = ""
            tokens = list()
        with open(outname, 'a') as outfile:
            writer = csv.writer(outfile, dialect='excel-tab')
            writer.writerow(tokens)

    print("[INFO] XML files now parsed to TSV. Please change the filetype in the config file from 'pubmedxml' to 'tsv' to continue with model training.")
    print("[INFO] Program will now exit.")
    exit(0)

# Step 1: Preprocess file to build custom dataset
# (reference: https://pytorch.org/tutorials/beginner/torchtext_custom_dataset_tutorial.html)

# Build pipeline

def get_tokens(data_iter):
    for d in data_iter:
        yield tokenizer(d)

def build_vocab(data_iter):
    """Builds vocabulary from iterator"""
    vocab = torchtext.vocab.build_vocab_from_iterator(
        data_iter,
        specials=["<unk>"],
        min_freq=MIN_WORD_FREQUENCY
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab

def get_data_iterator(ds_name, ds_type):
    if ds_name == "SLE":
        FILE_PATH = config["trainingcorpus"]
        data_iter = torchdata.datapipes.iter.IterableWrapper([FILE_PATH])
        data_iter = torchdata.datapipes.iter.FileOpener(data_iter, mode='rb')
        data_iter = data_iter.parse_csv(skip_lines=0, delimiter='\t')
        data_iter = torchtext.data.to_map_style_dataset(data_iter)
    else:
        raise ValueError("Choose dataset from: SLE")
    data_iter = torchtext.data.to_map_style_dataset(data_iter)
    return data_iter

##### Using CBOW method #####

# Utility function: collate_cbow
def collate_cbow(batch, pipeline):
    """
    Collation function for CBOW model, for use by Dataloader.
    `batch` is expected to be list of strings (paragraphs)
.    
    Context is represented as N=CBOW_N_WORDS past words 
    and N=CBOW_N_WORDS future words.
    
    Long paragraphs will be truncated to contain a maximum of MAX_SEQUENCE_LENGTH tokens.
    
    Each element in `batch_input` is N=CBOW_N_WORDS*2 context words.
    Each element in `batch_output` is a middle word.
    """
    batch_input, batch_output = [], []

    for paragraph in batch:
        text_tokens_ids = pipeline(paragraph)
        if len(text_tokens_ids) < CBOW_N_WORDS * 2 + 1:
            continue
        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]
        # idx: index within bag
        for idx in range(len(text_tokens_ids) - CBOW_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + CBOW_N_WORDS * 2 + 1)]
            output = token_id_sequence.pop(CBOW_N_WORDS)
            input_ = token_id_sequence
            batch_input.append(input_)
            batch_output.append(output)
    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output

##### Using Skipgram method #####
def collate_skipgram(batch, pipeline):
    """
    Collation function for Skipgram model, for use by Dataloader.
    `batch` is expected to be list of strings (paragraphs)
.    
    Context is represented as N=SKIPGRAM_N_WORDS past words 
    and N=SKIPGRAM_N_WORDS future words.
    
    Long paragraphs will be truncated to contain a maximum of MAX_SEQUENCE_LENGTH tokens.
    
    Each element in `batch_input` is N=CBOW_N_WORDS*2 context words.
    Each element in `batch_output` is a middle word.
    """
    batch_input, batch_output = [], []
    for paragraph in batch:
        text_tokens_ids = pipeline(paragraph)
        if len(text_tokens_ids) < SKIPGRAM_N_WORDS * 2 + 1:
            continue
        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]
        # idx: index within bag
        for idx in range(len(text_tokens_ids) - SKIPGRAM_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + SKIPGRAM_N_WORDS * 2 + 1)]
            input_ = token_id_sequence.pop(SKIPGRAM_N_WORDS)
            outputs = token_id_sequence
            for output in outputs:
                batch_input.append(input_)
                batch_output.append(output)
    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output

def get_dataloader_and_vocab(model_name, ds_name, ds_type, batch_size, shuffle, vocab=None):
    data_iter = get_data_iterator(ds_name, ds_type)
    tokenizer = nltk.tokenize.RegexpTokenizer(r'[a-zA-Z0-9-]+').tokenize
    if not vocab:
        vocab = build_vocab(data_iter)      
    pipeline = lambda x: vocab(x)
    if model_name == "cbow":
        collate_fn = collate_cbow
    elif model_name == "skipgram":
        collate_fn = collate_skipgram
    else:
        raise ValueError("Choose model from: cbow, skipgram")
    dataloader = torch.utils.data.DataLoader(
        data_iter,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=functools.partial(collate_fn, pipeline=pipeline),
    )
    return dataloader, vocab

train_dataloader, vocab = get_dataloader_and_vocab(
        model_name=config["model_name"],
        ds_name=config["dataset"],
        ds_type="train",
        batch_size=config["train_batch_size"],
        shuffle=config["shuffle"],
        vocab=None,
    )

val_dataloader, _ = get_dataloader_and_vocab(
        model_name=config["model_name"],
        ds_name=config["dataset"],
        ds_type="valid",
        batch_size=config["val_batch_size"],
        shuffle=config["shuffle"],
        vocab=vocab,
    )

##### Set model #####

def get_model_class(model_name: str):
    if model_name == "cbow":
        return CBOW_Model
    elif model_name == "skipgram":
        return SkipGram_Model
    else:
        raise ValueError("Choose model_name from: cbow, skipgram")
        return

class Trainer:
    """Main class for model training"""
    
    def __init__(
        self,
        model,
        epochs,
        train_dataloader,
        train_steps,
        val_dataloader,
        val_steps,
        checkpoint_frequency,
        criterion,
        optimizer,
        lr_scheduler,
        device,
        model_dir,
        model_name,
    ):  
        self.model = model
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.train_steps = train_steps
        self.val_dataloader = val_dataloader
        self.val_steps = val_steps
        self.criterion = criterion
        self.optimizer = optimizer
        self.checkpoint_frequency = checkpoint_frequency
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model_dir = model_dir
        self.model_name = model_name
        self.loss = {"train": [], "val": []}
        self.model.to(self.device)
    def train(self):
        for epoch in range(self.epochs):
            self._train_epoch()
            self._validate_epoch()
            print(
                "Epoch: {}/{}, Train Loss={:.5f}, Val Loss={:.5f}".format(
                    epoch + 1,
                    self.epochs,
                    self.loss["train"][-1],
                    self.loss["val"][-1],
                )
            )
            self.lr_scheduler.step()
            if self.checkpoint_frequency:
                self._save_checkpoint(epoch)
    def _train_epoch(self):
        self.model.train()
        running_loss = []
        for i, batch_data in enumerate(self.train_dataloader, 1):
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss.append(loss.item())
            if i == self.train_steps:
                break
        epoch_loss = numpy.mean(running_loss)
        self.loss["train"].append(epoch_loss)
    def _validate_epoch(self):
        self.model.eval()
        running_loss = []
        with torch.no_grad():
            for i, batch_data in enumerate(self.val_dataloader, 1):
                inputs = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss.append(loss.item())
                if i == self.val_steps:
                    break
        epoch_loss = numpy.mean(running_loss)
        self.loss["val"].append(epoch_loss)
    def _save_checkpoint(self, epoch):
        """Save model checkpoint to `self.model_dir` directory"""
        epoch_num = epoch + 1
        if epoch_num % self.checkpoint_frequency == 0:
            model_path = "checkpoint_{}.pt".format(str(epoch_num).zfill(3))
            model_path = os.path.join(self.model_dir, model_path)
            torch.save(self.model, model_path)
    def save_model(self):
        """Save final model to `self.model_dir` directory"""
        model_path = os.path.join(self.model_dir, "model.pt")
        torch.save(self.model, model_path)
    def export_weights(self):
        """Save final model to `self.model_dir` directory"""
        weights_path = os.path.join(self.model_dir, "weights.txt")
        with open(weights_path, "a") as fp:
            for i in self.model.state_dict():
                print(i, self.model.state_dict()[i], file=fp)
    def save_loss(self):
        """Save train/val loss as json file to `self.model_dir` directory"""
        loss_path = os.path.join(self.model_dir, "loss.json")
        with open(loss_path, "w") as fp:
            json.dump(self.loss, fp)

class CBOW_Model(torch.nn.Module):
    """
    Implementation of CBOW model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, vocab_size: int):
        super(CBOW_Model, self).__init__()
        self.embeddings = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )
        self.linear = torch.nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size,
        )
    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x

class SkipGram_Model(torch.nn.Module):
    """
    Implementation of Skip-Gram model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, vocab_size: int):
        super(SkipGram_Model, self).__init__()
        self.embeddings = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )
        self.linear = trch.nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size,
        )
    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = self.linear(x)
        return x

def get_model_class(model_name: str):
    if model_name == "cbow":
        return CBOW_Model
    elif model_name == "skipgram":
        return SkipGram_Model
    else:
        raise ValueError("Choose model_name from: cbow, skipgram")
        return

def get_optimizer_class(name: str):
    if name == "Adam":
        return torch.optim.Adam
    else:
        raise ValueError("Choose optimizer from: Adam")
        return
    

def get_lr_scheduler(optimizer, total_epochs: int, verbose: bool = True):
    """
    Scheduler to linearly decrease learning rate, 
    so that learning rate after the last epoch is 0.
    """
    lr_lambda = lambda epoch: (total_epochs - epoch) / total_epochs
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=verbose)
    return lr_scheduler


def save_config(config: dict, model_dir: str):
    """Save config file to `model_dir` directory"""
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "w") as stream:
        json.dump(config, stream)
        
        
def save_vocab(vocab, model_dir: str):
    """Save vocab file to `model_dir` directory"""
    vocab_path = os.path.join(model_dir, "vocab.pt")
    torch.save(vocab, vocab_path)
    

def train(config):
    os.makedirs(config["model_dir"])
    train_dataloader, vocab = get_dataloader_and_vocab(
        model_name=config["model_name"],
        ds_name=config["dataset"],
        ds_type="train",
        batch_size=config["train_batch_size"],
        shuffle=config["shuffle"],
        vocab=None,
    )
    val_dataloader, _ = get_dataloader_and_vocab(
        model_name=config["model_name"],
        ds_name=config["dataset"],
        ds_type="valid",
        batch_size=config["val_batch_size"],
        shuffle=config["shuffle"],
        vocab=vocab,
    )
    vocab_size = len(vocab.get_stoi())
    print(f"Vocabulary size: {vocab_size}")
    model_class = get_model_class(config["model_name"])
    model = model_class(vocab_size=vocab_size)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_class = get_optimizer_class(config["optimizer"])
    optimizer = optimizer_class(model.parameters(), lr=config["learning_rate"])
    lr_scheduler = get_lr_scheduler(optimizer, config["epochs"], verbose=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(
        model=model,
        epochs=config["epochs"],
        train_dataloader=train_dataloader,
        train_steps=config["train_steps"],
        val_dataloader=val_dataloader,
        val_steps=config["val_steps"],
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_frequency=config["checkpoint_frequency"],
        lr_scheduler=lr_scheduler,
        device=device,
        model_dir=config["model_dir"],
        model_name=config["model_name"],
    )
    trainer.train()
    print("Training finished.")
    trainer.export_weights()
    trainer.save_model()
    trainer.save_loss()
    save_vocab(vocab, config["model_dir"])
    save_config(config, config["model_dir"])
    print("Model artifacts saved to folder:", config["model_dir"])
    
with open(args.config, "r", encoding="UTF-8") as conf:
    config = json.load(conf)
    
train(config)
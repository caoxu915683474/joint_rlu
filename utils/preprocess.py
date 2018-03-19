import numpy as np
import random

flatten = lambda l: [item for sublist in l for item in sublist]
index_seq2slot = lambda s, index2slot: [index2slot[i] for i in s]
index_seq2word = lambda s, index2word: [index2word[i] for i in s]

def data_pre(data, length=50):

    data = [t.strip() for t in data]
    data = [[t.split("\t")[0].split(" "), t.split("\t")[1].split(" ")[:-1], t.split("\t")[1].split(" ")[-1]] for t in data]
    data = [[t[0][1:-1], t[1][1:], t[2]]for t in data]
    seq_in, seq_out, intent = list(zip(*data))
    sin = []
    sout = []
    for i in range(len(seq_in)):
        temp = seq_in[i]
        if len(temp) < length:
            temp.append("<EOS>")
            while len(temp) < length:
                temp.append("<PAD>")
        else:
            temp = temp[:length]
            temp[-1] = "<EOS>"
        sin.append(temp)
        temp = seq_out[i]
        if len(temp) < length:
            while len(temp) <length:
                temp.append("<PAD>")
        else:
            temp = temp[:length]
            temp[-1] = "<EOS>"
        sout.append(temp)
        data = list(zip(sin, sout, intent))
    return data

def get_info_from_training_data(data):
    seq_in, seq_out, intent = list(zip(*data))
    vocab = set(flatten(seq_in))
    slot_tag = set(flatten(seq_out))
    intent_tag = set(intent)

    word2index = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2,"<EOS>": 3}
    for token in vocab:
        if token not in word2index:
          word2index[token] = len(word2index)
    index2word = {v: k for k, v in word2index.items()}
    tag2index = {"<PAD>": 0, "<UNK>": 1, "O": 2}
    for tag in slot_tag:
        if tag not in tag2index:
            tag2index[tag] = len(tag2index)
    index2tag = {v: k for k, v in tag2index.items()}
    intent2index ={"<UNK>": 0}
    for ii in intent_tag:
        if ii not in intent2index:
            intent2index[ii] = len(intent2index)
    index2intent = {v: k for k, v in intent2index.items()}
    return word2index, index2word, tag2index, index2tag, intent2index, index2intent

def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex:eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch

def to_index(train, word2index, slot2index, intent2index):
    new_train = []
    for sin, sout, intent in train:
        sin_ix = list(map(lambda i:word2index[i] if i in word2index else word2index["<UNK>"], sin))
        true_length = sin.index("<EOS>")
        sout_ix = list(map(lambda i:slot2index[i] if i in slot2index else slot2index["<UNK>"], sout))
        intent_ix = intent2index[intent] if intent in intent2index else intent2index["<UNK>"]
        new_train.append([sin_ix, true_length, sout_ix, intent_ix])
    return new_train
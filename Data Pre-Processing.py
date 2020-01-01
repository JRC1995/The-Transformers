#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


filename = 'glove.6B.100d.txt' 
# (glove data set from: https://nlp.stanford.edu/projects/glove/)

def loadEmbeddings(filename):
    vocab2embd = {}

    with open(filename) as infile:
        for line in infile:
            row = line.strip().split(' ')
            word = row[0].lower()
            # print(word)
            if word not in vocab2embd:
                vec = np.asarray(row[1:], np.float32)
                if len(vec) == 100:
                    vocab2embd[word] = vec

    print('Embedding Loaded.')
        
    return vocab2embd

# Pre-trained word embedding
vocab2embd = loadEmbeddings(filename)

word_vec_dim = 100 # word_vec_dim = dimension of each word vectors


vocab2embd['<UNK>'] = np.random.randn(word_vec_dim)
vocab2embd['<GO>'] = np.random.randn(word_vec_dim)
vocab2embd['<PRED>'] = np.random.randn(word_vec_dim)
vocab2embd['<EOS>'] = np.random.randn(word_vec_dim)
vocab2embd['<PAD>'] = np.zeros(word_vec_dim)


# In[2]:


import csv
import nltk
#nltk.download('punkt')
from nltk import word_tokenize
import string

summaries = []
texts = []

def clean(text):
    text = text.lower()
    printable = set(string.printable)
    text = "".join(list(filter(lambda x: x in printable, text))) #filter funny characters, if any.
    return text

counter={}
max_len_text = 100
max_len_sum = 20

#max_data = 100000
i=0
with open('Reviews.csv', 'rt') as csvfile: #Data from https://www.kaggle.com/snap/amazon-fine-food-reviews
    Reviews = csv.DictReader(csvfile)
    count=0
    for row in Reviews:
        #if count<max_data:
        clean_text = word_tokenize(clean(row['Text']))
        clean_summary = word_tokenize(clean(row['Summary']))
        
        if len(clean_text) <= max_len_text and len(clean_summary) <= max_len_sum:
            
            for word in clean_text:
                if word in vocab2embd:
                    counter[word]=counter.get(word,0)+1
            for word in clean_summary:
                if word in vocab2embd:
                    counter[word]=counter.get(word,0)+1

            summaries.append(clean_summary)
            texts.append(clean_text)
        #count+=1
        if i%10000==0:
            print("Processing data {}".format(i))
        i+=1
        
print("Current size of data: "+str(len(texts)))


# In[3]:



vocab = [word for word in counter]


counts = [counter[word] for word in vocab]

sorted_idx = sorted(range(len(counts)), key=counts.__getitem__)
sorted_idx.reverse()

vocab = [vocab[idx] for idx in sorted_idx]

special_tags = ["<UNK>","<GO>","<PRED>","<EOS>","<PAD>"]
if len(vocab) > 40000-len(special_tags):
    vocab = vocab[0:40000-len(special_tags)]
    

vocab += special_tags 

vocab_dict = {word:i for i,word in enumerate(vocab)}

embeddings = []
for word in vocab:
    embeddings.append(vocab2embd[word].tolist())


# In[4]:


# SHUFFLE

import random

texts_idx = [idx for idx in range(0,len(texts))]
random.shuffle(texts_idx)

texts = [texts[idx] for idx in texts_idx]
summaries = [summaries[idx] for idx in texts_idx]


# In[5]:


import random

index = random.randint(0,len(texts)-1)

print("SAMPLE CLEANED & TOKENIZED TEXT: \n\n"+str(texts[index]))
print("\nSAMPLE CLEANED & TOKENIZED SUMMARY: \n\n"+str(summaries[index]))


# In[6]:


train_len = int(.7*len(texts))
val_len = int(.2*len(texts))

train_summaries = summaries[0:train_len]
train_texts = texts[0:train_len]

val_summaries = summaries[train_len:val_len+train_len]
val_texts = texts[train_len:train_len+val_len]

test_summaries = summaries[train_len+val_len:]
test_texts = texts[train_len+val_len:]


# In[7]:


def bucket_and_batch(texts, summaries, batch_size=32):
    
    global vocab_dict
    vocab2idx = vocab_dict
    
    PAD = vocab2idx['<PAD>']
    EOS = vocab2idx['<EOS>']
    UNK = vocab2idx['<UNK>']

    true_seq_lens = np.zeros((len(texts)), dtype=int)
    for i in range(len(texts)):
        true_seq_lens[i] = len(texts[i])

    # sorted in descending order after flip
    sorted_by_len_indices = np.flip(np.argsort(true_seq_lens), 0)

    sorted_texts = []
    sorted_summaries = []

    for i in range(len(texts)):
        sorted_texts.append(texts[sorted_by_len_indices[i]])
        sorted_summaries.append(summaries[sorted_by_len_indices[i]])

    i = 0
    batches_texts = []
    batches_summaries = []
    batches_true_seq_in_lens = []
    batches_true_seq_out_lens = []

    while i < len(sorted_texts):

        if i+batch_size > len(sorted_texts):
            batch_size = len(sorted_texts)-i

        batch_texts = []
        batch_summaries = []
        batch_true_seq_in_lens = []
        batch_true_seq_out_lens = []

        max_in_len = len(sorted_texts[i])
        max_out_len = max([len(sorted_summaries[j])+1 for j in range(i,i+batch_size)])

        for j in range(i, i + batch_size):

            text = sorted_texts[j]
            summary = sorted_summaries[j]
            
            text = [vocab2idx.get(word,UNK) for word in text]
            summary = [vocab2idx.get(word,UNK) for word in summary]
            
            init_in_len = len(text)
            init_out_len = len(summary)+1 # +1 for EOS

            while len(text) < max_in_len:
                text.append(PAD)
                
            summary.append(EOS)
            
            while len(summary) < max_out_len:
                summary.append(PAD)

            batch_summaries.append(summary)
            batch_texts.append(text)
            batch_true_seq_in_lens.append(init_in_len)
            batch_true_seq_out_lens.append(init_out_len)

        #batch_texts = np.asarray(batch_texts, dtype=np.int32)
        #batch_summaries = np.asarray(batch_summaries, dtype=np.int32)
        #batch_true_seq_in_lens = np.asarray(batch_true_seq_in_lens, dtype=np.int32)
        #batch_true_seq_out_lens = np.asarray(batch_true_seq_out_lens, dtype=np.int32)

        batches_texts.append(batch_texts)
        batches_summaries.append(batch_summaries)
        batches_true_seq_in_lens.append(batch_true_seq_in_lens)
        batches_true_seq_out_lens.append(batch_true_seq_out_lens)

        i += batch_size

    return batches_texts, batches_summaries, batches_true_seq_in_lens, batches_true_seq_out_lens


# In[8]:


train_batches_x,train_batches_y,train_batches_in_lens, train_batches_out_lens = bucket_and_batch(train_texts,train_summaries)
val_batches_x,val_batches_y,val_batches_in_lens,val_batches_out_lens= bucket_and_batch(val_texts,val_summaries)
test_batches_x,test_batches_y,test_batches_in_lens,test_batches_out_lens= bucket_and_batch(test_texts,test_summaries)


# In[9]:


#Saving processed data in another file.

import json

diction = {}
diction['vocab']=vocab
diction['embd']=embeddings
diction['train_batches_x']=train_batches_x
diction['train_batches_y']=train_batches_y
diction['train_batches_in_len'] = train_batches_in_lens
diction['train_batches_out_len'] = train_batches_out_lens
diction['val_batches_x']=val_batches_x
diction['val_batches_y']=val_batches_y
diction['val_batches_in_len'] = val_batches_in_lens
diction['val_batches_out_len'] = val_batches_out_lens
diction['test_batches_x']=test_batches_x
diction['test_batches_y']=test_batches_y
diction['test_batches_in_len'] = test_batches_in_lens
diction['test_batches_out_len'] = test_batches_out_lens

with open('ProcessedData.json', 'w') as fp:
    json.dump(diction, fp)


# In[ ]:





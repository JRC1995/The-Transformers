
# coding: utf-8

# In[1]:


import numpy as np
from __future__ import division

filename = 'glove.6B.300d.txt' 
# (glove data set from: https://nlp.stanford.edu/projects/glove/)

#filename = 'numberbatch-en.txt'
#(https://github.com/commonsense/conceptnet-numberbatch)

def loadembeddings(filename):
    vocab = []
    embd = []
    file = open(filename,'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Word vector embeddings Loaded.')
    file.close()
    return vocab,embd

# Pre-trained word embedding
vocab,embd = loadembeddings(filename)

word_vec_dim = len(embd[0]) # word_vec_dim = dimension of each word vectors

#e = np.zeros((word_vec_dim,),np.float32)+0.0001

vocab.append('<UNK>') #<UNK> represents unknown word
embdunk = np.random.randn(word_vec_dim) #np.asarray(embd[vocab.index('unk')],np.float32)+e
    
vocab.append('<EOS>') #<EOS> represents end of sentence
embdeos = np.random.randn(word_vec_dim) #np.asarray(embd[vocab.index('eos')],np.float32)+e

vocab.append('<PAD>') #<PAD> represents paddings


embd.append(embdunk)  
embd.append(embdeos)  
    
embdpad = np.zeros(word_vec_dim)
embd.append(embdpad)

embedding = np.asarray(embd)
embedding = embedding.astype(np.float32)


# In[2]:


def word2vec(word):  # converts a given word into its vector representation
    if word in vocab:
        return embedding[vocab.index(word)]
    else:
        return embedding[vocab.index('<UNK>')]


# In[3]:


def most_similar_cosine(x):
    #embed = embedding[0:len(embedding)-1]
    embed = embedding
    xdoty = np.multiply(embed,x) #element-wise
    xdoty = np.sum(xdoty,1)
    xlen = np.square(x)
    xlen = np.sum(xlen,0)
    xlen = np.sqrt(xlen)
    ylen = np.square(embed)
    ylen = np.sum(ylen,1)
    ylen = np.sqrt(ylen)
    xlenylen = np.multiply(xlen,ylen)
    cosine_similarities = np.divide(xdoty,xlenylen)
    return np.flip(np.argsort(cosine_similarities),0)

def most_similar_eucli(x):
    xminusy = np.subtract(embedding,x)
    sq_xminusy = np.square(xminusy)
    sum_sq_xminusy = np.sum(sq_xminusy,1)
    eucli_dists = np.sqrt(sum_sq_xminusy)
    return np.argsort(eucli_dists)

word = 'frog'

most_similars = most_similar_eucli(word2vec(word))

print("TOP TEN MOST SIMILAR WORDS TO '"+str(word)+"':\n")
for i in range(0,10):
    print(str(i+1)+". "+str(vocab[most_similars[i]]))
    


# In[4]:


def vec2word(vec):   # converts a given vector representation into the represented word 
    most_similars = most_similar_eucli(np.asarray(vec,np.float32))
    return vocab[most_similars[0]]


# In[5]:


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
    #print(text)
    text = text.translate(str.maketrans('', '',string.punctuation))
    #print(text)
    return text

#max_data = 100000
i=0
with open('Reviews.csv', 'rt') as csvfile: #Data from https://www.kaggle.com/snap/amazon-fine-food-reviews
    Reviews = csv.DictReader(csvfile)
    count=0
    for row in Reviews:
        #if count<max_data:
        clean_text = clean(row['Text'])
        clean_summary = clean(row['Summary'])
        summaries.append(word_tokenize(clean_summary))
        texts.append(word_tokenize(clean_text))
        #count+=1
        if i%10000==0:
            print("Processing data {}".format(i))
        i+=1


# In[6]:


i = 0
texts_v2 = []
summaries_v2 = []

max_len_text = 80
max_len_sum = 4
for text in texts:
    if(len(text)<=max_len_text and len(summaries[i])<=max_len_sum): 
        #remove data pairs with review length more than max_len_text
        #or summary length more than max_len_sum
        texts_v2.append(text)
        summaries_v2.append(summaries[i])
    i+=1
    
print("Current size of data: "+str(len(texts_v2)))


# In[7]:


vocab_dict = {word:i for i,word in enumerate(vocab)}


# In[8]:


i = 0
texts = []
summaries = []

for summary in summaries_v2:
    flag = 0    
    for word in summary:
        if word not in vocab_dict:
            flag = 1
            
    #Remove summary and its corresponding text 
    #if out of vocabulary word present in summary
    
    if flag == 0:
        summaries.append(summary)
        texts.append(texts_v2[i])
    i+=1

print("Current size of data: "+str(len(texts)))


# In[9]:


"""
#REDUCE DATA (FOR SPEEDING UP THE NEXT STEPS)

MAXIMUM_DATA_NUM = 20000

texts = texts_v3[0:MAXIMUM_DATA_NUM]
summaries = summaries_v3[0:MAXIMUM_DATA_NUM]
"""


# In[10]:


# SHUFFLE

import random

texts_idx = [idx for idx in range(0,len(texts))]
random.shuffle(texts_idx)

texts = [texts[idx] for idx in texts_idx]
summaries = [summaries[idx] for idx in texts_idx]


# In[11]:


import random

index = random.randint(0,len(texts)-1)

print("SAMPLE CLEANED & TOKENIZED TEXT: \n\n"+str(texts[index]))
print("\nSAMPLE CLEANED & TOKENIZED SUMMARY: \n\n"+str(summaries[index]))


# In[12]:


train_len = int(.7*len(texts))
val_len = int(.2*len(texts))

train_summaries = summaries[0:train_len]
train_texts = texts[0:train_len]

val_summaries = summaries[train_len:val_len+train_len]
val_texts = texts[train_len:train_len+val_len]

test_summaries = summaries[train_len+val_len:]
test_texts = texts[train_len+val_len:]


# In[13]:


def bucket_and_batch(texts,summaries):
    lentexts = []

    i=0
    for text in texts:
        lentexts.append(len(text))
        i+=1
        
    #print(len(texts))
    #print(len(summaries))

    sortedindex = np.flip(np.argsort(lentexts),axis=0)
    #sort indexes according to the sequence length of corresponding texts. 
    
    batch_size = 50

    bi=0

    batches_x = []
    batches_y = []
    batch_x = []
    batch_y = []

    for i in range(0,len(texts)):

        if bi>=batch_size:
            bi=0
            #print(batch_x)
            #print(batch_y)
            batches_x.append(batch_x)
            batches_y.append(batch_y)
            batch_x = []
            batch_y = []
            
        if bi==0:
            max_len = len(texts[int(sortedindex[i])])
        
        text = []
        summary = []
        
        for j in range(0,max_len):
            if j==len(texts[int(sortedindex[i])]):
                text.append(vocab_dict['<EOS>'])
            elif j > len(texts[int(sortedindex[i])]):
                text.append(vocab_dict['<PAD>'])
            else:
                text.append(vocab_dict.get(texts[int(sortedindex[i])][j],vocab_dict['<UNK>']))
                
        for j in range(0,5):
            if j==len(summaries[int(sortedindex[i])]):
                summary.append(vocab_dict['<EOS>'])
            elif j > len(summaries[int(sortedindex[i])]):
                summary.append(vocab_dict['<PAD>'])
            else:
                summary.append(vocab_dict.get(summaries[int(sortedindex[i])][j],vocab_dict['<UNK>']))
                

        batch_x.append(text)
        batch_y.append(summary)

        bi+=1
        
    return batches_x,batches_y
    


# In[14]:


train_batches_x,train_batches_y = bucket_and_batch(train_texts,train_summaries)
val_batches_x,val_batches_y = bucket_and_batch(val_texts,val_summaries)
test_batches_x,test_batches_y = bucket_and_batch(test_texts,test_summaries)


# In[15]:


#Saving processed data in another file.

import pickle

PICK = [vocab,embd,50,train_batches_x,train_batches_y,val_batches_x,val_batches_y,test_batches_x,test_batches_y]

with open('AmazonPICKLE', 'wb') as fp:
    pickle.dump(PICK, fp)


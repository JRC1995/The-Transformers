
# coding: utf-8

# In[1]:


import numpy as np
import math
from __future__ import division

filename = 'glove.6B.50d.txt' 
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

e = np.zeros((word_vec_dim,),np.float32)+0.0001

vocab.append('<UNK>') #<UNK> represents unknown word
embdunk = np.asarray(embd[vocab.index('unk')],np.float32)+e
    
vocab.append('<EOS>') #<EOS> represents end of sentence
embdeos = np.asarray(embd[vocab.index('eos')],np.float32)+e

vocab.append('<PAD>') #<PAD> represents paddings

flag1=0
flag2=0

for vec in embd:
    
    if np.all(np.equal(np.asarray(vec,np.float32),embdunk)):
        flag1=1
        print "FLAG1"   
    if np.all(np.equal(np.asarray(vec,np.float32),embdeos)):
        flag2=1
        print "FLAG2"

if flag1==0:
    embd.append(embdunk)  
if flag2 == 0:
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

def most_similar_eucli(x):
    xminusy = np.subtract(embedding,x)
    sq_xminusy = np.square(xminusy)
    sum_sq_xminusy = np.sum(sq_xminusy,1)
    eucli_dists = np.sqrt(sum_sq_xminusy)
    return np.argsort(eucli_dists)

def vec2word(vec):   # converts a given vector representation into the represented word 
    most_similars = most_similar_eucli(np.asarray(vec,np.float32))
    return vocab[most_similars[0]]
    


# In[3]:


import pickle


with open ('AmazonPICKLE', 'rb') as fp:
    PICK = pickle.load(fp)

vocab_limit = PICK[0]
vocab_len = len(vocab_limit)

batch_size = int(PICK[1])

batches_x = PICK[2]
batches_y = PICK[3]

batches_x_pe = PICK[4] #already position encoded

max_len = len(batches_y[0][0]) #max output len
    


# In[4]:


embd_limit = []

for i in xrange(0,vocab_len):
    embd_limit.append(word2vec(vocab_limit[i]))

np_embd_limit = np.asarray(embd_limit,dtype=np.float32)


# In[5]:


def positional_encoding(seq_len,model_dimensions):
    pe = np.zeros((seq_len,model_dimensions,),np.float32)
    for pos in xrange(0,seq_len):
        for i in xrange(0,model_dimensions):
            pe[pos][i] = math.sin(pos/(10000**(2*i/model_dimensions)))
    return pe.reshape((seq_len,model_dimensions))


# In[6]:


#Prepare training data

train_len = int(0.75*len(batches_x))

train_batches_x= batches_x[0:train_len]
train_batches_x_pe = batches_x_pe[0:train_len]

train_batches_y = batches_y[0:train_len]

# (Rest of the data can be used for validating and testing)


# In[7]:


import tensorflow as tf

h=5
N=4 #no. of decoder and encoder layers
learning_rate=0.002
iters = 100
keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None,None,word_vec_dim])
y = tf.placeholder(tf.int32, [None,None])


# In[8]:


#modified version of def LN used here: 
#https://theneuralperspective.com/2016/10/27/gradient-topics/

def layer_norm(inputs, epsilon = 1e-5):

    mean, var = tf.nn.moments(inputs, [1,2], keep_dims=True)
    
    scale = tf.Variable(tf.ones([1,1,word_vec_dim]),
                        dtype=tf.float32)
    shift = tf.Variable(tf.ones([1,1,word_vec_dim]),
                        dtype=tf.float32)
    LN = scale * (inputs - mean) / tf.sqrt(var + epsilon) + shift
 
    return LN


# In[9]:



def attention(Q,K,V,d):

    K_ = tf.transpose(K,[0,2,1])
 
    d = tf.cast(d,tf.float32)
    result = tf.matmul(tf.nn.dropout(tf.nn.softmax(tf.div(tf.matmul(Q,K_),tf.sqrt(d))),keep_prob),V)
 
    return result

def masked_attention(Q,K,V,d,pos):
    
    K_ = tf.transpose(K,[0,2,1])
    d = tf.cast(d,tf.float32)
    
    softmax_component = tf.div(tf.matmul(Q,K_),tf.sqrt(d))
    
    softmax_component = tf.reshape(softmax_component,[batch_size,max_len,max_len])
    
    #Create mask
    #0 for normal positions, -2**30 for positions influenced by illegal input
    #Add mask to softmax component. 
    
    mask = np.zeros((max_len,max_len),dtype=np.float32)
    mask[pos:max_len,:] = -2**30
    mask[:,pos:max_len] = -2**30
    mask = tf.convert_to_tensor(mask)
    mask = tf.reshape(mask,[1,max_len,max_len])
    
    softmax_component = softmax_component + mask
    
    result = tf.matmul(tf.nn.dropout(tf.nn.softmax(softmax_component),keep_prob),V)
    
    return result
       

def multihead_attention(Q,K,V,pos=0,mask=False):
    
    Q_ = tf.reshape(Q,[-1,tf.shape(Q)[2]])
    K_ = tf.reshape(K,[-1,tf.shape(Q)[2]])
    V_ = tf.reshape(V,[-1,tf.shape(Q)[2]])
    
    d = 30
    
    heads = tf.TensorArray(size=h,dtype=tf.float32)
    
    Wq = tf.Variable(tf.truncated_normal(shape=[h,word_vec_dim,d],stddev=0.01))
    Wk = tf.Variable(tf.truncated_normal(shape=[h,word_vec_dim,d],stddev=0.01))
    Wv = tf.Variable(tf.truncated_normal(shape=[h,word_vec_dim,d],stddev=0.01))
    Wo = tf.Variable(tf.truncated_normal(shape=[h*d,word_vec_dim],stddev=0.01))
    
    for i in xrange(0,h):
        
        Q_w = tf.matmul(Q_,Wq[i])
        Q_w = tf.reshape(Q_w,[tf.shape(Q)[0],tf.shape(Q)[1],d])
        
        K_w = tf.matmul(K_,Wk[i])
        K_w = tf.reshape(K_w,[tf.shape(K)[0],tf.shape(K)[1],d])
        
        V_w = tf.matmul(V_,Wv[i])
        V_w = tf.reshape(V_w,[tf.shape(V)[0],tf.shape(V)[1],d])
        
        if mask == False:
            head = attention(Q_w,K_w,V_w,d)
        else:
            head = masked_attention(Q_w,K_w,V_w,d,pos)
            
        heads = heads.write(i,head)
        
    heads = heads.stack()
    
    concated = heads[0]
    
    for i in xrange(1,h):
        concated = tf.concat([concated,heads[i]],2)

    concated = tf.reshape(concated,[-1,h*d])
    out = tf.matmul(concated,Wo)
    out = tf.reshape(out,[tf.shape(heads)[1],tf.shape(heads)[2],word_vec_dim])
    
    return out
    


# In[10]:


def encoder(x):
    
    d=1024
    
    W1 = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,d],stddev=0.01))
    b1 = tf.Variable(tf.truncated_normal(shape=[1,d],stddev=0.01))
    W2 = tf.Variable(tf.truncated_normal(shape=[d,word_vec_dim],stddev=0.01))
    b2 = tf.Variable(tf.truncated_normal(shape=[1,word_vec_dim],stddev=0.01))
    
    sublayer1 = multihead_attention(x,x,x)
    sublayer1 = tf.nn.dropout(sublayer1,keep_prob)
    sublayer1 = layer_norm(sublayer1 + x)
    
    sublayer1_ = tf.reshape(sublayer1,[-1,word_vec_dim])
    
    sublayer2 = tf.matmul(tf.nn.relu(tf.matmul(sublayer1_,W1)+b1),W2) + b2
    sublayer2 = tf.reshape(sublayer2,tf.shape(x))
    sublayer2 = tf.nn.dropout(sublayer2,keep_prob)
    sublayer2 = layer_norm(sublayer2 + sublayer1)
    
    return sublayer2


# In[11]:


def decoder(y,enc_out,mask=False,pos=0):
    
    d=1024
    
    W1 = tf.Variable(tf.truncated_normal(shape=[word_vec_dim,d],stddev=0.01))
    b1 = tf.Variable(tf.truncated_normal(shape=[1,d],stddev=0.01))
    W2 = tf.Variable(tf.truncated_normal(shape=[d,word_vec_dim],stddev=0.01))
    b2 = tf.Variable(tf.truncated_normal(shape=[1,word_vec_dim],stddev=0.01))

    sublayer1 = multihead_attention(y,y,y,pos,mask)
    sublayer1 = tf.nn.dropout(sublayer1,keep_prob)
    sublayer1 = layer_norm(sublayer1 + y)
    
    sublayer2 = multihead_attention(sublayer1,enc_out,enc_out)
    sublayer2 = tf.nn.dropout(sublayer2,keep_prob)
    sublayer2 = layer_norm(sublayer2 + sublayer1)
    
    sublayer2_ = tf.reshape(sublayer2,[-1,word_vec_dim])
    
    sublayer3 = tf.matmul(tf.nn.relu(tf.matmul(sublayer2_,W1)+b1),W2) + b2
    sublayer3 = tf.reshape(sublayer3,tf.shape(y))
    sublayer3 = tf.nn.dropout(sublayer3,keep_prob)
    sublayer3 = layer_norm(sublayer3 + sublayer2)
    
    return sublayer3


# In[12]:


def model(x):
    
    encoderin = tf.nn.dropout(x,keep_prob)
    
    #"we apply dropout to the sums of the embeddings and the
    # positional encodings in both the encoder and decoder stacks."
    
    pe = tf.constant(positional_encoding(max_len,word_vec_dim)) #all position encodings
    pe = tf.reshape(pe,[max_len,1,word_vec_dim])
    
    #encoder layers
    for i in xrange(0,N):
        encoderin = encoder(encoderin)
    encoderout = encoderin
    
    #decoder_layers
    
    #tf.shape(x)[0] == batch_size
    
    decoderin_part1 = tf.ones([tf.shape(x)[0],1,word_vec_dim],dtype=tf.float32)
    filled = 1 #no. of output words that are filled
    
    d=512
    
    Wpd = tf.Variable(tf.truncated_normal(shape=[max_len*word_vec_dim,vocab_len],stddev=0.01))
    bpd = tf.Variable(tf.truncated_normal(shape=[1,vocab_len],stddev=0.01))
    
    tf_embd_limit = tf.convert_to_tensor(np_embd_limit)
    
    for i in xrange(0,max_len):
        
        decoderin_part2 = tf.zeros([tf.shape(x)[0],(max_len-filled),word_vec_dim],dtype=tf.float32)
        
        decoderin = tf.concat([decoderin_part1,decoderin_part2],1)
        
        decoderin = tf.nn.dropout(decoderin,keep_prob)
        
        #"we apply dropout to the sums of the embeddings and the
        # positional encodings in both the encoder and decoder stacks."
        
        for j in xrange(0,N):
            decoderin = decoder(decoderin,encoderout,mask=True,pos=filled)
            
        decoderout = decoderin
        #decoderout shape = batch_size x seq_len x word_vec_dim
        
        #converting to probability distributions
        decoderout = tf.reshape(decoderout,[tf.shape(x)[0],max_len*word_vec_dim])
        out_prob_dist = tf.nn.relu(tf.matmul(decoderout,Wpd)+bpd)
        out_prob_dist = tf.reshape(out_prob_dist,[tf.shape(x)[0],vocab_len])
   
        #Finding the word vector of the word of highest probability in the predicted distribution:
        
        out_index = tf.cast(tf.argmax(out_prob_dist,1),tf.int32)        
        output = tf.gather(tf_embd_limit,out_index)

        # Position Encoding
        output = output + pe[i]
        output = tf.reshape(output,[tf.shape(x)[0],1,word_vec_dim])
                                
        #concatenate with previous batch_outputs
        if i==0:
            decoderin_part1 = output #right shift
            filled = 1 
            out_probs = tf.reshape(out_prob_dist,[tf.shape(x)[0],1,vocab_len])
        else:
            decoderin_part1 = tf.concat([decoderin_part1,output],1)
            filled += 1
            out_probs = tf.concat([out_probs,tf.reshape(out_prob_dist,[tf.shape(x)[0],1,vocab_len])],1)
    
    return out_probs          


# In[13]:


output = model(x)

#OPTIMIZER

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9,beta2=0.98,epsilon=1e-9).minimize(cost)

#wanna add some temperature?

"""temperature = 0.7
scaled_output = tf.log(output)/temperature
softmax_output = tf.nn.softmax(scaled_output)"""

#(^Use it with "#prediction_int = np.random.choice(range(vocab_len), p=array.ravel())")

softmax_output = tf.nn.softmax(output)


# In[14]:


def transform_out(output_batch):
    out = []
    for output_text in output_batch:
        output_len = len(output_text)
        transformed_output = np.zeros([output_len],dtype=np.int32)
        for i in xrange(0,output_len):
            transformed_output[i] = vocab_limit.index(vec2word(output_text[i]))
        out.append(transformed_output)
    return np.asarray(out,np.int32)


# In[ ]:


import string
from __future__ import print_function

init = tf.global_variables_initializer()

with tf.Session() as sess: # Start Tensorflow Session
    
    saver = tf.train.Saver() 
    # Prepares variable for saving the model
    sess.run(init) #initialize all variables
    step = 0   
    best_loss = 999
    display_step = 1
    
    while step < iters:
           
        batch_len = len(train_batches_x_pe)
        for i in xrange(0,batch_len):
            
            sample_no = np.random.randint(0,batch_size)
            print("\nCHOSEN SAMPLE NO.: "+str(sample_no))
            
            train_out = transform_out(train_batches_y[i])
            
            if i%display_step==0:
                print("\nEpoch: "+str(step+1)+" Iteration: "+str(i+1))
                print("\nSAMPLE TEXT:")
                for vec in train_batches_x[i][sample_no]:
                    print(str(vec2word(vec)),end=" ")
                print("\n")


            # Run optimization operation (backpropagation)
            _,loss,out = sess.run([optimizer,cost,softmax_output],feed_dict={x: train_batches_x_pe[i], 
                                                                             y: train_out,
                                                                             keep_prob: 0.9})
            
            if i%display_step==0:
                print("\nPREDICTED SUMMARY OF THE SAMPLE:\n")
                flag = 0
                for array in out[sample_no]:
                    
                    prediction_int = np.random.choice(range(vocab_len), p=array.ravel()) 
                    #(^use this if you want some variety)
                    #(or use this what's below:)
                    
                    #prediction_int = np.argmax(array)
                    
                    if vocab_limit[prediction_int] in string.punctuation or flag==0: 
                        print(str(vocab_limit[prediction_int]),end='')
                    else:
                        print(" "+str(vocab_limit[prediction_int]),end='')
                    flag=1
                print("\n")
                
                print("ACTUAL SUMMARY OF THE SAMPLE:\n")
                for vec in batches_y[i][sample_no]:
                    print(str(vec2word(vec)),end=" ")
                print("\n")
            
            print("loss="+str(loss))
                  
            if(loss<best_loss):
                best_loss = loss
                saver.save(sess, 'Model_Backup/allattmodel.ckpt')

        step=step+1
    

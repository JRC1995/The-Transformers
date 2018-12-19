
# coding: utf-8

# In[1]:


import numpy as np
import math
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


import pickle


with open ('AmazonPICKLE', 'rb') as fp:
    PICK = pickle.load(fp)

vocab_limit = PICK[0]
vocab_len = len(vocab_limit)

embd = PICK[1]

batch_size = int(PICK[2])

train_batches_x = PICK[3]
train_batches_y = PICK[4]

val_batches_x = PICK[5]
val_batches_y = PICK[6]

#print(len(val_batches_x))

test_batches_x = PICK[7]
test_batches_y = PICK[8]

max_len = len(train_batches_y[0][0]) #max output len
#print(max_len)
    


# In[3]:


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
    


# In[4]:


vocab_dict = {word:idx for idx,word in enumerate(vocab)}
vec2vocab = {idx:word for word,idx in vocab_dict.items()}


# In[5]:



np_embd_limit = np.asarray(embd,dtype=np.float32)


# In[6]:


import tensorflow as tf

#Hyperparamters

h=8 #no. of heads
N=1 #no. of decoder and encoder layers
learning_rate=0.001
epochs = 200
keep_prob = tf.placeholder(tf.float32)

#Placeholders

x = tf.placeholder(tf.int32, [None,None])
y = tf.placeholder(tf.int32, [None,None])

output_len = tf.placeholder(tf.int32)

teacher_forcing = tf.placeholder(tf.bool)

tf_pad_mask = tf.placeholder(tf.float32,[None,None])
tf_illegal_position_masks = tf.placeholder(tf.float32,[None,None,None])

#tf_pe_out = tf.placeholder(tf.float32,[None,None,None]) #positional codes for output


# In[7]:



# Dimensions for Q (Query),K (Keys) and V (Values) for attention layers.

dqkv = 32 
 
#Parameters for attention sub-layers for all n encoders

Wq_enc = tf.Variable(tf.truncated_normal(shape=[N,h,word_vec_dim,dqkv],stddev=0.01))
Wk_enc = tf.Variable(tf.truncated_normal(shape=[N,h,word_vec_dim,dqkv],stddev=0.01))
Wv_enc = tf.Variable(tf.truncated_normal(shape=[N,h,word_vec_dim,dqkv],stddev=0.01))
Wo_enc = tf.Variable(tf.truncated_normal(shape=[N,h*dqkv,word_vec_dim],stddev=0.01))

#Parameters for position-wise fully connected layers for n encoders

d = 1024
W1_enc = tf.Variable(tf.truncated_normal(shape=[N,1,1,word_vec_dim,d],stddev=0.01))
b1_enc = tf.Variable(tf.constant(0,tf.float32,shape=[N,d]))
W2_enc = tf.Variable(tf.truncated_normal(shape=[N,1,1,d,word_vec_dim],stddev=0.01))
b2_enc = tf.Variable(tf.constant(0,tf.float32,shape=[N,word_vec_dim]))
 
#Parameters for 2 attention sub-layers for all n decoders

Wq_dec_1 = tf.Variable(tf.truncated_normal(shape=[N,h,word_vec_dim,dqkv],stddev=0.01))
Wk_dec_1 = tf.Variable(tf.truncated_normal(shape=[N,h,word_vec_dim,dqkv],stddev=0.01))
Wv_dec_1 = tf.Variable(tf.truncated_normal(shape=[N,h,word_vec_dim,dqkv],stddev=0.01))
Wo_dec_1 = tf.Variable(tf.truncated_normal(shape=[N,h*dqkv,word_vec_dim],stddev=0.01))
Wq_dec_2 = tf.Variable(tf.truncated_normal(shape=[N,h,word_vec_dim,dqkv],stddev=0.01))
Wk_dec_2 = tf.Variable(tf.truncated_normal(shape=[N,h,word_vec_dim,dqkv],stddev=0.01))
Wv_dec_2 = tf.Variable(tf.truncated_normal(shape=[N,h,word_vec_dim,dqkv],stddev=0.01))
Wo_dec_2 = tf.Variable(tf.truncated_normal(shape=[N,h*dqkv,word_vec_dim],stddev=0.01))
 
#Parameters for position-wise fully connected layers for n decoders

d = 1024
W1_dec = tf.Variable(tf.truncated_normal(shape=[N,1,1,word_vec_dim,d],stddev=0.01))
b1_dec = tf.Variable(tf.constant(0,tf.float32,shape=[N,d]))
W2_dec = tf.Variable(tf.truncated_normal(shape=[N,1,1,d,word_vec_dim],stddev=0.01))
b2_dec = tf.Variable(tf.constant(0,tf.float32,shape=[N,word_vec_dim]))
 
#Layer Normalization parameters for encoder and decoder   

scale_enc_1 = tf.Variable(tf.ones([N,1,1,word_vec_dim]),dtype=tf.float32)
shift_enc_1 = tf.Variable(tf.zeros([N,1,1,word_vec_dim]),dtype=tf.float32)

scale_enc_2 = tf.Variable(tf.ones([N,1,1,word_vec_dim]),dtype=tf.float32)
shift_enc_2 = tf.Variable(tf.zeros([N,1,1,word_vec_dim]),dtype=tf.float32)

#Layer Normalization parameters for decoder   

scale_dec_1 = tf.Variable(tf.ones([N,1,1,word_vec_dim]),dtype=tf.float32)
shift_dec_1 = tf.Variable(tf.zeros([N,1,1,word_vec_dim]),dtype=tf.float32)

scale_dec_2 = tf.Variable(tf.ones([N,1,1,word_vec_dim]),dtype=tf.float32)
shift_dec_2 = tf.Variable(tf.zeros([N,1,1,word_vec_dim]),dtype=tf.float32)

scale_dec_3 = tf.Variable(tf.ones([N,1,1,word_vec_dim]),dtype=tf.float32)
shift_dec_3 = tf.Variable(tf.zeros([N,1,1,word_vec_dim]),dtype=tf.float32)


# In[8]:


def positional_encoding(seq_len,model_dimensions=50):
    pe = np.zeros((seq_len,model_dimensions,),np.float32)
    for pos in range(0,seq_len):
        for i in range(0,model_dimensions):
            if i%2==0:
                pe[pos][i] = math.sin(pos/(10000**(2*i/model_dimensions)))
            elif i%(2+1)==0:
                pe[pos][i] = math.cos(pos/(10000**(2*i/model_dimensions)))
    return pe.reshape((1,seq_len,model_dimensions))


# In[9]:



def layer_norm(inputs,scale,shift,epsilon = 1e-9):

    mean, var = tf.nn.moments(inputs, [1,2], keep_dims=True)

    LN = tf.multiply((scale / tf.sqrt(var + epsilon)),(inputs - mean)) + shift
 
    return LN


# In[10]:


def generate_masks_for_illegal_positions(out_len):
    
    masks=np.zeros((out_len-1,out_len,out_len),dtype=np.float32)
    
    for i in range(1,out_len):
        mask = np.zeros((out_len,out_len),dtype=np.float32)
        mask[i:out_len,:] = -2**30
        mask[:,i:out_len] = -2**30
        masks[i-1] = mask
        
    return masks


# In[11]:



def attention(Q,K,V,d,filled=0,mask=False):

    K = tf.transpose(K,[0,2,1])
    d = tf.cast(d,tf.float32)
    
    softmax_component = tf.div(tf.matmul(Q,K),tf.sqrt(d))
    
    if mask == True:
        softmax_component = softmax_component + tf_illegal_position_masks[filled-1]
        
    result = tf.matmul(tf.nn.dropout(tf.nn.softmax(softmax_component),keep_prob),V)
 
    return result
       

def multihead_attention(Q,K,V,d,weights,filled=0,mask=False):
    
    Q_ = tf.reshape(Q,[-1,tf.shape(Q)[2]])
    K_ = tf.reshape(K,[-1,tf.shape(Q)[2]])
    V_ = tf.reshape(V,[-1,tf.shape(Q)[2]])

    heads = tf.TensorArray(size=h,dtype=tf.float32)
    
    Wq = weights['Wq']
    Wk = weights['Wk']
    Wv = weights['Wv']
    Wo = weights['Wo']
    
    heads=[]
    
    for i in range(0,h):
        
        Q_w = tf.matmul(Q_,Wq[i])
        Q_w = tf.reshape(Q_w,[tf.shape(Q)[0],tf.shape(Q)[1],d])
        
        K_w = tf.matmul(K_,Wk[i])
        K_w = tf.reshape(K_w,[tf.shape(K)[0],tf.shape(K)[1],d])
        
        V_w = tf.matmul(V_,Wv[i])
        V_w = tf.reshape(V_w,[tf.shape(V)[0],tf.shape(V)[1],d])

        head = attention(Q_w,K_w,V_w,d,filled,mask)
            
        heads.append(head)
        
    concated = tf.concat(heads,axis=-1)

    concated = tf.reshape(concated,[-1,h*d])
    out = tf.matmul(concated,Wo)
    out = tf.reshape(out,[tf.shape(heads)[1],tf.shape(heads)[2],word_vec_dim])
    
    return out
    


# In[12]:


max_text_len = 80

def encoder(x,weights,attention_weights,dqkv):

    W1 = weights['W1']
    W2 = weights['W2']
    b1 = weights['b1']
    b2 = weights['b2']
    
    scale1 = weights['scale1']
    shift1 = weights['shift1']
    scale2 = weights['scale2']
    shift2 = weights['shift2']
    
    # SUBLAYER 1 (MASKED MULTI HEADED SELF ATTENTION)
    
    # positional_encoding 
    x = x+ tf.constant(positional_encoding(max_text_len,word_vec_dim),tf.float32)[:,0:tf.shape(x)[1],0:tf.shape(x)[2]]
    
    sublayer1 = multihead_attention(x,x,x,dqkv,attention_weights)
    sublayer1 = tf.nn.dropout(sublayer1,keep_prob)
    sublayer1 = layer_norm(sublayer1 + x,scale1,shift1)
    
    sublayer1_ = tf.reshape(sublayer1,[tf.shape(sublayer1)[0],1,tf.shape(sublayer1)[1],word_vec_dim])
    
    # SUBLAYER 2 (TWO 1x1 CONVOLUTIONAL LAYERS AKA POSITION WISE FULLY CONNECTED NETWORKS)
    
    sublayer2 = tf.nn.conv2d(sublayer1_, W1, strides=[1,1,1,1], padding='SAME')
    sublayer2 = tf.nn.bias_add(sublayer2,b1)
    sublayer2 = tf.nn.relu(sublayer2)
    
    sublayer2 = tf.nn.conv2d(sublayer2, W2, strides=[1,1,1,1], padding='SAME')
    sublayer2 = tf.nn.bias_add(sublayer2,b2)
    
    sublayer2 = tf.reshape(sublayer2,[tf.shape(sublayer2)[0],tf.shape(sublayer2)[2],word_vec_dim])
    
    sublayer2 = tf.nn.dropout(sublayer2,keep_prob)
    sublayer2 = layer_norm(sublayer2 + sublayer1,scale2,shift2)
    
    return sublayer2


# In[13]:


max_summary_len = 5
def decoder(y,enc_out,weights,masked_attention_weights,attention_weights,dqkv,mask=False,filled=0):

    W1 = weights['W1']
    W2 = weights['W2']
    b1 = weights['b1']
    b2 = weights['b2']
    
    scale1 = weights['scale1']
    shift1 = weights['shift1']
    scale2 = weights['scale2']
    shift2 = weights['shift2']
    scale3 = weights['scale3']
    shift3 = weights['shift3']
    
    # SUBLAYER 1 (MASKED MULTI HEADED SELF ATTENTION)
    
    # positional_encoding
    y = y+ tf.constant(positional_encoding(max_summary_len,word_vec_dim),tf.float32)[:,0:max_summary_len,0:tf.shape(y)[2]]

    sublayer1 = multihead_attention(y,y,y,dqkv,masked_attention_weights,filled,mask)
    sublayer1 = tf.nn.dropout(sublayer1,keep_prob)
    sublayer1 = layer_norm(sublayer1 + y,scale1,shift1)
    
    # SUBLAYER 2 (MULTIHEADED ENCODER-DECODER INTERLAYER ATTENTION)
    
    sublayer1 = sublayer1+ tf.constant(positional_encoding(max_summary_len,word_vec_dim),tf.float32)[:,0:max_summary_len,0:word_vec_dim]
    enc_out = enc_out + tf.constant(positional_encoding(max_text_len,word_vec_dim),tf.float32)[:,0:tf.shape(enc_out)[1],0:word_vec_dim]
    
    sublayer2 = multihead_attention(sublayer1,enc_out,enc_out,dqkv,attention_weights)
    sublayer2 = tf.nn.dropout(sublayer2,keep_prob)
    sublayer2 = layer_norm(sublayer2 + sublayer1,scale2,shift2)
    
    # SUBLAYER 3 (TWO 1x1 CONVOLUTIONAL LAYERS AKA POSITION WISE FULLY CONNECTED NETWORKS)
    
    sublayer2_ = tf.reshape(sublayer2,[tf.shape(sublayer2)[0],1,tf.shape(sublayer2)[1],word_vec_dim])
    
    sublayer3 = tf.nn.conv2d(sublayer2_, W1, strides=[1,1,1,1], padding='SAME')
    sublayer3 = tf.nn.bias_add(sublayer3,b1)
    sublayer3 = tf.nn.relu(sublayer3)
    
    sublayer3 = tf.nn.conv2d(sublayer3, W2, strides=[1,1,1,1], padding='SAME')
    sublayer3 = tf.nn.bias_add(sublayer3,b2)
    
    sublayer3 = tf.reshape(sublayer3,[tf.shape(sublayer3)[0],tf.shape(sublayer3)[2],word_vec_dim])
    
    sublayer3 = tf.nn.dropout(sublayer3,keep_prob)
    sublayer3 = layer_norm(sublayer3 + sublayer2,scale3,shift3)
    
    return sublayer3


# In[14]:


def stacked_encoders(layer_num,encoderin):
    
    for i in range(0,layer_num):
        
        encoder_weights = {
            
            'W1': W1_enc[i],
            'W2': W2_enc[i],
            'b1': b1_enc[i],
            'b2': b2_enc[i],
            'scale1': scale_enc_1[i],
            'shift1': shift_enc_1[i],
            'scale2': scale_enc_2[i],
            'shift2': shift_enc_2[i],
        }
        
        attention_weights = {
            
            'Wq': Wq_enc[i],
            'Wk': Wk_enc[i],
            'Wv': Wv_enc[i],
            'Wo': Wo_enc[i],                       
        }
        
        encoderin = encoder(encoderin,encoder_weights,attention_weights,dqkv)
    
    return encoderin
    


# In[15]:


def stacked_decoders(layer_num,decoderin,encoderout,filled):
    
    for j in range(0,layer_num):
        
        decoder_weights = {
            
            'W1': W1_dec[j],
            'W2': W2_dec[j],
            'b1': b1_dec[j],
            'b2': b2_dec[j],
            'scale1': scale_dec_1[j],
            'shift1': shift_dec_1[j],
            'scale2': scale_dec_2[j],
            'shift2': shift_dec_2[j],
            'scale3': scale_dec_3[j],
            'shift3': shift_dec_3[j],
        }
            
        masked_attention_weights = {
            
            'Wq': Wq_dec_1[j],
            'Wk': Wk_dec_1[j],
            'Wv': Wv_dec_1[j],
            'Wo': Wo_dec_1[j],                       
        }
        
        attention_weights = {
            
            'Wq': Wq_dec_2[j],
            'Wk': Wk_dec_2[j],
            'Wv': Wv_dec_2[j],
            'Wo': Wo_dec_2[j],                       
        }
            
        decoderin = decoder(decoderin,encoderout,
                            decoder_weights,
                            masked_attention_weights,
                            attention_weights,
                            dqkv,
                            mask=True,filled=filled)
    return decoderin
    


# In[16]:


def predicted_embedding(out_prob_dist,tf_embd):
    out_index = tf.cast(tf.argmax(out_prob_dist,1),tf.int32)
    return tf.gather(tf_embd,out_index)

def replaceSOS(output,out_prob_dist):
    return output,tf.constant(1),tf.reshape(out_prob_dist,[tf.shape(x)[0],1,vocab_len])

def add_pred_to_output_list(decoderin_part_1,output,filled,out_probs,out_prob_dist):
    decoderin_part_1 = tf.concat([decoderin_part_1,output],1)
    filled += 1
    out_probs = tf.concat([out_probs,tf.reshape(out_prob_dist,[tf.shape(x)[0],1,vocab_len])],1)
    return decoderin_part_1,filled,out_probs


# In[17]:


def model(x,teacher_forcing=True):
    
    tf_embd = tf.convert_to_tensor(np_embd_limit)

    x = tf.nn.embedding_lookup(tf_embd,x)
    
        
    # NOTE: tf.shape(x)[0] == batch_size
    
    encoderin = x # (should be already positionally encoded) 
    encoderin = tf.nn.dropout(encoderin,keep_prob)

    
    # ENCODER LAYERS

    encoderout = stacked_encoders(N,encoderin)
    

    # DECODER LAYERS

    decoderin_part_1 = tf.ones([tf.shape(x)[0],1,word_vec_dim],dtype=tf.float32) #represents SOS
    
    filled = tf.constant(1) 
    # no. of output words that are filled
    # filled value is used to retrieve appropriate mask for illegal positions. 
    
    
   
    Wpd = tf.transpose(tf_embd)
    # Wpd the transpose of the output embedding matrix will be used to convert the decoder output
    # into a probability distribution over the output language vocabulary. 
    
    out_probs = tf.zeros([tf.shape(x)[0],output_len,vocab_len],tf.float32)
    # out_probs will contain the list of probability distributions.

    #tf_while_loop since output_len will be dynamically defined during session run
    
    i=tf.constant(0)
    
    def cond(i,filled,decoderin_part_1,out_probs):
        return i<output_len
    
    def body(i,filled,decoderin_part_1,out_probs):
        
        decoderin_part_2 = tf.zeros([tf.shape(x)[0],(output_len-filled),word_vec_dim],dtype=tf.float32)
        
        decoderin = tf.concat([decoderin_part_1,decoderin_part_2],1)
        
        decoderin = tf.nn.dropout(decoderin,keep_prob)
        
        decoderout = stacked_decoders(N,decoderin,encoderout,filled)
        
        # decoderout shape (now) = batch_size x seq_len x word_vec_dim

        decoderout = tf.reduce_sum(decoderout,1) 
        # A weighted summation of the attended decoder input
        # decoderout shape (now) = batch_size x word_vec_dim
        
        # converting decoderout to probability distributions
        
        out_prob_dist = tf.matmul(decoderout,Wpd)
   
        # If teacher forcing is false, initiate predicted_embedding(). It guesses the output embeddings
        # to be that whose vocabulary index has maximum probability in out_prob_dist
        # (the current output probability distribution). The embedding is used in the next
        # iteration. 
        
        # If teacher forcing is true, use the embedding of target index from y (laebls) 
        # for the next iteration.
        
        output = tf.cond(tf.equal(teacher_forcing,tf.convert_to_tensor(False)),
                         lambda: predicted_embedding(out_prob_dist,tf_embd),
                         lambda: tf.gather(tf_embd,y[:,i]))
        
        # Position Encoding the output
        
        #output = output + tf_pe_out[i]
        output = tf.reshape(output,[tf.shape(x)[0],1,word_vec_dim])
                                
        
        #concatenate with list of previous predicted output tokens
        
        decoderin_part_1,filled,out_probs = tf.cond(tf.equal(i,0),
                                        lambda:replaceSOS(output,out_prob_dist),
                                        lambda:add_pred_to_output_list(decoderin_part_1,output,filled,out_probs,out_prob_dist))
        
        return i+1,filled,decoderin_part_1,out_probs
            
    _,_,_,out_probs = tf.while_loop(cond,body,[i,filled,decoderin_part_1,out_probs],
                      shape_invariants=[i.get_shape(),
                                        filled.get_shape(),
                                        tf.TensorShape([None,None,word_vec_dim]),
                                        tf.TensorShape([None,None,vocab_len])])

    return out_probs          


# In[18]:


# Construct Model
output = model(x,teacher_forcing)

#OPTIMIZER

trainables = tf.trainable_variables()
beta=1e-7

regularization = tf.reduce_sum([tf.nn.l2_loss(var) for var in trainables])

cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=y)
cost = tf.multiply(cost,tf_pad_mask) #mask used to remove loss effect due to PADS
cost = tf.reduce_mean(cost) + beta*regularization

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9,beta2=0.98,epsilon=1e-9).minimize(cost)

#wanna add some temperature?

"""temperature = 0.7
scaled_output = tf.log(output)/temperature
softmax_output = tf.nn.softmax(scaled_output)"""

#(^Use it with "#prediction_int = np.random.choice(range(vocab_len), p=array.ravel())")

softmax_output = tf.nn.softmax(output)


# In[19]:




def create_pad_Mask(output_batch):
    pad_index = vocab_limit.index('<PAD>')
    mask = np.ones_like((output_batch),np.float32)
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if output_batch[i,j]==pad_index:
                mask[i,j]=0
    return mask


# In[20]:


import string
import random
import nltk
from __future__ import print_function

init = tf.global_variables_initializer()

with tf.Session() as sess: # Start Tensorflow Session
    
    saver = tf.train.Saver() 
    # Prepares variable for saving the model
    sess.run(init) #initialize all variables
    step = 0   
    best_BLEU = 0
    display_step = 100
    
    while step < epochs:
           
        batch_len = len(train_batches_x)
        rand_idx = [idx for idx in range(batch_len)]
        random.shuffle(rand_idx)
        rand_idx = rand_idx[0:2000]
        count=0
        for i in rand_idx:
            
            sample_no = np.random.randint(0,batch_size)
            
            
            #train_out = transform_out(train_batches_y[i])

            if count%display_step==0:
                print("\nEpoch: "+str(step+1)+" Iteration: "+str(count+1))
                print("\nCHOSEN SAMPLE NO.: "+str(sample_no))
                print("\nSAMPLE TEXT:")
                for vec in train_batches_x[i][sample_no]:
                    print(str(vec2vocab[vec]),end=" ")
                print("\n")
                
            
                
            rand = random.randint(0,4) #determines chance of using Teacher Forcing
            if rand==2:
                random_bool = False
            else:
                random_bool = True
                
            output_seq_len = len(train_batches_y[i][0])
            
            illegal_position_masks = generate_masks_for_illegal_positions(output_seq_len)
            
            pad_mask = create_pad_Mask(np.asarray(train_batches_y[i]))
            
            train_batch_x = np.asarray(train_batches_x[i],np.int32)
            train_batch_y = np.asarray(train_batches_y[i],np.int32)
            
            #print(train_batch_x.shape)
            #print(train_batch_y.shape)

            # Run optimization operation (backpropagation)
            _,loss,out = sess.run([optimizer,cost,softmax_output],feed_dict={x: train_batch_x, 
                                                                             y: train_batch_y,
                                                                             keep_prob: 0.9,
                                                                             output_len: output_seq_len,
                                                                             tf_illegal_position_masks: illegal_position_masks,
                                                                             tf_pad_mask: pad_mask,
                                                                             teacher_forcing: random_bool})
            
            if count%display_step==0:
                print("\nPREDICTED SUMMARY OF THE SAMPLE:\n")
                flag = 0
                for array in out[sample_no]:
                    
                    #prediction_int = np.random.choice(range(vocab_len), p=array.ravel()) 
                    #(^use this if you want some variety)
                    #(or use this what's below:)
                    
                    prediction_int = np.argmax(array)
                    
                    if vocab_limit[prediction_int] in string.punctuation or flag==0: 
                        print(str(vocab_limit[prediction_int]),end='')
                    else:
                        print(" "+str(vocab_limit[prediction_int]),end='')
                    flag=1
                print("\n")
                
                print("ACTUAL SUMMARY OF THE SAMPLE:\n")
                for vec in train_batches_y[i][sample_no]:
                    print(str(vec2vocab[vec]),end=" ")
                print("\n")
            
                print("loss="+str(loss))
                
            count+=1
                
        print("\n\nSTARTING VALIDATION\n\n")
                
        batch_len = len(val_batches_x)
        #print(batch_len)
        total_BLEU_argmax=0
        
        total_len=0
        for i in range(0,batch_len):
            
            sample_no = np.random.randint(0,batch_size)
            
            
            #train_out = transform_out(train_batches_y[i])

            if i%display_step==0:
                print("\nEpoch: "+str(step+1)+" Iteration: "+str(i+1))
                print("\nCHOSEN SAMPLE NO.: "+str(sample_no))
                print("\nSAMPLE TEXT:")
                for vec in val_batches_x[i][sample_no]:
                    print(str(vec2vocab[vec]),end=" ")
                print("\n")
                
            output_seq_len = len(val_batches_y[i][0])
            
            illegal_position_masks = generate_masks_for_illegal_positions(output_seq_len)
            
            pad_mask = create_pad_Mask(np.asarray(val_batches_y[i]))
            
            val_batch_x = np.asarray(val_batches_x[i],np.int32)
            val_batch_y = np.asarray(val_batches_y[i],np.int32)
            
            #print(train_batch_x.shape)
            #print(train_batch_y.shape)

            loss,out = sess.run([cost,softmax_output],feed_dict={x: val_batch_x, 
                                                                 y: val_batch_y,
                                                                 keep_prob: 1,
                                                                 output_len: output_seq_len,
                                                                 tf_illegal_position_masks: illegal_position_masks,
                                                                 tf_pad_mask: pad_mask,
                                                                 teacher_forcing: 0})
            
            batch_summaries = val_batch_y
            batch_argmax_preds = np.argmax(out,axis=-1)

            batch_BLEU_argmax = 0
            batch_BLEU_argmax_list=[]
            
            for summary, argmax_pred in zip(batch_summaries, batch_argmax_preds):

                str_summary = []
                str_argmax_pred = []
                gold_EOS_flag = 0

                for t in range(len(summary)):

                    if gold_EOS_flag == 0:

                        gold_idx = summary[t]
                        argmax_idx = argmax_pred[t]

                        if vec2vocab.get(gold_idx, vocab_dict['<UNK>']) == "<EOS>":
                            gold_EOS_flag = 1
                        else:
                            str_summary.append(str(gold_idx))
                            str_argmax_pred.append(str(argmax_idx))

                if len(str_summary) < 2:
                    n_gram = len(str_summary)
                else:
                    n_gram = 2

                weights = [1/n_gram for id in range(n_gram)]
                weights = tuple(weights)

                BLEU_argmax = nltk.translate.bleu_score.sentence_bleu(
                    [str_summary], str_argmax_pred, weights=weights)

                batch_BLEU_argmax += BLEU_argmax
                batch_BLEU_argmax_list.append(BLEU_argmax)

            total_BLEU_argmax += batch_BLEU_argmax
            total_len += batch_size
            
            if i%display_step==0:
                print("\nPREDICTED SUMMARY OF THE SAMPLE:\n")
                flag = 0
                for array in out[sample_no]:
                    
                    #prediction_int = np.random.choice(range(vocab_len), p=array.ravel()) 
                    #(^use this if you want some variety)
                    #(or use this what's below:)
                    
                    prediction_int = np.argmax(array)
                    
                    if vocab_limit[prediction_int] in string.punctuation or flag==0: 
                        print(str(vocab_limit[prediction_int]),end='')
                    else:
                        print(" "+str(vocab_limit[prediction_int]),end='')
                    flag=1
                print("\n")
                
                print("ACTUAL SUMMARY OF THE SAMPLE:\n")
                for vec in val_batches_y[i][sample_no]:
                    print(str(vec2vocab[vec]),end=" ")
                print("\n")
            
                print("loss="+str(loss))
                print("BLEU-2=",batch_BLEU_argmax_list[sample_no])
        
        avg_BLEU = total_BLEU_argmax/total_len
        print("AVERAGE VALIDATION BLEU:",avg_BLEU)
        
        if(avg_BLEU>=best_BLEU):
            best_BLEU = avg_BLEU
            saver.save(sess, 'Model_Backup/allattmodel.ckpt')
            print("\nCheckpoint Created\n")

        step=step+1
    


# In[25]:


# TESTING

init = tf.global_variables_initializer()


with tf.Session() as sess: # Start Tensorflow Session
    
    saver = tf.train.Saver() 
    
    saver.restore(sess, 'Model_Backup/allattmodel.ckpt')
    #sess.run(init) #initialize all variables
    print("\nCheckpoint Restored\n")
    step = 0   
    best_BLEU = 0
    display_step = 100

                
    print("\n\nSTARTING TEST\n\n")

    batch_len = len(test_batches_x)
    #print(batch_len)
    total_BLEU_argmax=0

    total_len=0
    for i in range(0,batch_len):

        sample_no = np.random.randint(0,batch_size)


        #train_out = transform_out(train_batches_y[i])

        if i%display_step==0:
            print("\nEpoch: "+str(step+1)+" Iteration: "+str(i+1))
            print("\nCHOSEN SAMPLE NO.: "+str(sample_no))
            print("\nSAMPLE TEXT:")
            for vec in test_batches_x[i][sample_no]:
                print(str(vec2vocab[vec]),end=" ")
            print("\n")

        output_seq_len = len(test_batches_y[i][0])

        illegal_position_masks = generate_masks_for_illegal_positions(output_seq_len)

        pad_mask = create_pad_Mask(np.asarray(test_batches_y[i]))

        test_batch_x = np.asarray(test_batches_x[i],np.int32)
        test_batch_y = np.asarray(test_batches_y[i],np.int32)

        #print(train_batch_x.shape)
        #print(train_batch_y.shape)

        loss,out = sess.run([cost,softmax_output],feed_dict={x: test_batch_x, 
                                                             y: test_batch_y,
                                                             keep_prob: 1,
                                                             output_len: output_seq_len,
                                                             tf_illegal_position_masks: illegal_position_masks,
                                                             tf_pad_mask: pad_mask,
                                                             teacher_forcing: 0})

        batch_summaries = test_batch_y
        batch_argmax_preds = np.argmax(out,axis=-1)

        batch_BLEU_argmax = 0
        batch_BLEU_argmax_list=[]

        for summary, argmax_pred in zip(batch_summaries, batch_argmax_preds):

            str_summary = []
            str_argmax_pred = []
            gold_EOS_flag = 0

            for t in range(len(summary)):

                if gold_EOS_flag == 0:

                    gold_idx = summary[t]
                    argmax_idx = argmax_pred[t]

                    if vec2vocab.get(gold_idx, vocab_dict['<UNK>']) == "<EOS>":
                        gold_EOS_flag = 1
                    else:
                        str_summary.append(str(gold_idx))
                        str_argmax_pred.append(str(argmax_idx))

            if len(str_summary) < 2:
                n_gram = len(str_summary)
            else:
                n_gram = 2

            weights = [1/n_gram for id in range(n_gram)]
            weights = tuple(weights)

            BLEU_argmax = nltk.translate.bleu_score.sentence_bleu(
                [str_summary], str_argmax_pred, weights=weights)

            batch_BLEU_argmax += BLEU_argmax
            batch_BLEU_argmax_list.append(BLEU_argmax)

        total_BLEU_argmax += batch_BLEU_argmax
        total_len += batch_size

        if i%display_step==0:
            print("\nPREDICTED SUMMARY OF THE SAMPLE:\n")
            flag = 0
            for array in out[sample_no]:

                #prediction_int = np.random.choice(range(vocab_len), p=array.ravel()) 
                #(^use this if you want some variety)
                #(or use this what's below:)

                prediction_int = np.argmax(array)

                if vocab_limit[prediction_int] in string.punctuation or flag==0: 
                    print(str(vocab_limit[prediction_int]),end='')
                else:
                    print(" "+str(vocab_limit[prediction_int]),end='')
                flag=1
            print("\n")

            print("ACTUAL SUMMARY OF THE SAMPLE:\n")
            for vec in test_batches_y[i][sample_no]:
                print(str(vec2vocab[vec]),end=" ")
            print("\n")

            print("loss="+str(loss))
            print("BLEU-2=",batch_BLEU_argmax_list[sample_no])

    avg_BLEU = total_BLEU_argmax/total_len
    print("AVERAGE TEST BLEU:",avg_BLEU)

    


## Attention is All You Need (For Abstractive Summarization)

Based on [Transformers](https://arxiv.org/pdf/1706.03762.pdf)

Download Dataset from [here](https://www.kaggle.com/snap/amazon-fine-food-reviews/data) and keep review.csv in the same directory as the ipynb files.
Download glove 840B common crawl from [here](https://nlp.stanford.edu/projects/glove/) and keep glove.840B.300d.txt in the same directory as the ipynb files.

Run Data_Pre-Processing.ipynb to process data.

I updated the previous code with [relative positional encoding](https://arxiv.org/abs/1901.02860) and [depth-scaled initialization](https://www.aclweb.org/anthology/D19-1083.pdf). I also changed the overall structure. It should work with tensorflow 2.0+ now, though it's still running in Tensorflow 1 format (no eager or such). I also changed the decoder from the previous implementation. The current decoder setup is closer to the original implementation (I implemented it differently before). 

Note, this is just a toy model (just 1 layered encoder-1-layered-decoder) with a toy dataset.


```python
import json
import numpy as np
import math


with open ('ProcessedData.json') as fp:
    diction = json.load(fp)

    
vocab = diction['vocab']
embd = diction['embd']
train_batches_x = diction['train_batches_x']
train_batches_y = diction['train_batches_y']
val_batches_x = diction['val_batches_x']
val_batches_y = diction['val_batches_y']
test_batches_x = diction['test_batches_x']
test_batches_y = diction['test_batches_y']
train_batches_in_lens = diction['train_batches_in_len']
train_batches_out_lens = diction['train_batches_out_len'] 
val_batches_in_lens = diction['val_batches_in_len']
val_batches_out_lens = diction['val_batches_out_len']
test_batches_in_lens = diction['test_batches_in_len']
test_batches_out_lens = diction['test_batches_out_len']

vocab_len = len(vocab)

vocab2idx = {word:idx for idx,word in enumerate(vocab)}
idx2vocab = {idx:word for word,idx in vocab2idx.items()}

embeddings = np.asarray(embd,dtype=np.float32)
word_vec_dim = embeddings.shape[-1]

```

## Define Placeholders and Hyperparameters


```python
import tensorflow.compat.v1 as tf 

tf.disable_v2_behavior()
tf.disable_eager_execution()

#Hyperparamters

heads = 6
max_decoding_len = 21
max_pos_len = 5000
learning_rate=1e-3
epochs = 10
fc_dim = 512
dropout_rate=0.1
attention_dropout_rate=0.1
encoder_layers = 1
decoder_layers = 1

#Placeholders

tf_texts = tf.placeholder(tf.int32, [None,None])
tf_summaries = tf.placeholder(tf.int32, [None,None])
tf_text_lens = tf.placeholder(tf.int32,[None])
tf_summary_lens = tf.placeholder(tf.int32,[None])
tf_teacher_forcing = tf.placeholder(tf.bool)
tf_train = tf.placeholder(tf.bool)
tf_no_eval = tf.placeholder(tf.bool)
```

    WARNING:tensorflow:From /home/jishnu/miniconda3/envs/ML/lib/python3.6/site-packages/tensorflow_core/python/compat/v2_compat.py:65: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
    Instructions for updating:
    non-resource variables are not supported in the long term


## GELU approximation

(used by BERT)


https://arxiv.org/abs/1606.08415

https://github.com/hendrycks/GELUs


```python
def gelu(x):
    return 0.5 * x * (1 + tf.nn.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))
```

## Dropout Function


```python
def dropout(x,rate,training):
    return tf.cond(training,
                  lambda: tf.nn.dropout(x,rate=rate),
                  lambda:x)
```

## Layer Normalization Function

https://arxiv.org/abs/1607.06450


```python
def layerNorm(inputs, dim, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE, dtype=tf.float32):
        scale = tf.get_variable("scale", shape=[1, 1, dim],
                                dtype=tf.float32,
                                initializer=tf.ones_initializer())

        shift = tf.get_variable("shift", shape=[1, 1, dim],
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer())

    mean, var = tf.nn.moments(inputs, [-1], keep_dims=True)

    epsilon = 1e-9

    LN = tf.multiply((scale / tf.sqrt(var + epsilon)), (inputs - mean)) + shift

    return LN
```

## Sine-Cosine Positional Encoding


```python
def spatial_encoding(D):
    
    global max_pos_len
    
    S = max_pos_len

    pe = np.zeros((2*S+1, D,), np.float32)

    for pos in range(-S, S+1):
        for i in range(0, D):
            if i % 2 == 0:
                pe[pos+S, i] = math.sin(pos/(10000**(i/D)))
            else:
                pe[pos+S, i] = math.cos(pos/(10000**((i-1)/D)))

    return tf.constant(pe.reshape((2*S+1, D)), tf.float32)

PE = spatial_encoding(word_vec_dim)
```

## Attention Mask




```python
def create_mask(Q,V,Q_mask,V_mask,neg_inf = -2.0**32):
    
    global heads
    
    N = tf.shape(Q)[0]
    qS = tf.shape(Q)[1]
    vS = tf.shape(V)[1]

    y = tf.zeros([N, qS, vS], tf.float32)
    x = tf.cast(tf.fill([N, qS, vS], neg_inf), tf.float32)

    binary_mask = tf.reshape(V_mask, [N, 1, vS])
    binary_mask = tf.tile(binary_mask, [1, qS, 1])
    binary_mask = binary_mask*Q_mask

    mask = tf.where(tf.equal(binary_mask, tf.constant(0, tf.float32)),
                    x=x,
                    y=y)

    mask = tf.reshape(mask, [1, N, qS, vS])
    mask = tf.tile(mask, [heads, 1, 1, 1])
    mask = tf.reshape(mask, [heads*N,qS,vS])

    return mask
```

## Relative Positional Embeddings

ADAPTED FROM: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py

Transformer XL version: https://arxiv.org/abs/1901.02860


```python
    
def generate_relative_embd(qS,vS,embeddings):
    
    global max_pos_len
    
    S = tf.maximum(qS,vS)

    range_vec = tf.reshape(tf.range(S), [1, S])
    range_mat = tf.tile(range_vec, [S, 1])

    relative_pos_mat = range_mat - tf.transpose(range_mat)
    relative_pos_mat = relative_pos_mat[0:qS,0:vS]

    relative_pos_mat_shifted = relative_pos_mat + max_pos_len
    # will represent -max_pos_len by 0,-max_pos_len+1 by 1, and so on

    RE = tf.nn.embedding_lookup(embeddings, relative_pos_mat_shifted)

    return RE
```

## Multiheaded attention

with depth scaling: https://arxiv.org/abs/1908.11365


```python
def multiheaded_attention(Q,V,
                          true_q_len,true_v_len,
                          train,name,
                          causal=False,
                          current_timestep=1,
                          current_depth=1,
                          attention_dropout_rate = 0.1):
    
    global heads
    global PE # position encoding
    global word_vec_dim
    global vocab2idx
    
    PRED = vocab2idx['<PRED>']

    N = tf.shape(Q)[0]
    qS = tf.shape(Q)[1]
    vS = tf.shape(V)[1]
    D = word_vec_dim

    d = D//heads

    Q_mask = tf.sequence_mask(true_q_len, maxlen=qS, dtype=tf.float32)
    Q_mask = tf.reshape(Q_mask, [N, qS, 1])

    V_mask = tf.sequence_mask(true_v_len, maxlen=vS, dtype=tf.float32)
    V_mask = tf.reshape(V_mask, [N, vS, 1])
    
    if causal:
        attention_len = tf.tile(tf.reshape(current_timestep+1,[1]),[N])
        causal_mask = tf.sequence_mask(attention_len, maxlen=vS, dtype=tf.float32)
        causal_mask = tf.reshape(causal_mask,[N,vS,1])
        
        Q_mask = tf.ones([1,1,1],tf.float32)
        
        attention_mask = create_mask(Q,V,Q_mask,V_mask*causal_mask)
    else:
        attention_mask = create_mask(Q,V,Q_mask,V_mask)
    
    l = current_depth

    init = tf.initializers.variance_scaling(
        scale=1/l, mode='fan_avg', distribution='uniform')

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE, dtype=tf.float32):

        Wq = tf.get_variable("Wq", [heads, D,  d],
                             dtype=tf.float32, initializer=init)

        Wk = tf.get_variable("Wk", [heads, D, d],
                             dtype=tf.float32, initializer=init)

        Wv = tf.get_variable("Wv", [heads, D, d],
                             dtype=tf.float32, initializer=init)

        Wq = tf.transpose(Wq, [1, 0, 2])
        Wq = tf.reshape(Wq, [D, heads*d])

        Wk = tf.transpose(Wk, [1, 0, 2])
        Wk = tf.reshape(Wk, [D, heads*d])

        Wv = tf.transpose(Wv, [1, 0, 2])
        Wv = tf.reshape(Wv, [D, heads*d])

        Wo = tf.get_variable("Wo", [heads*d, D],
                             dtype=tf.float32, initializer=init)

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE, dtype=tf.float32):

        u = tf.get_variable("u_bias", [heads, 1, 1, d],
                            dtype=tf.float32, initializer=tf.zeros_initializer())

        v = tf.get_variable("v_bias", [heads, 1, 1, d],
                            dtype=tf.float32, initializer=tf.zeros_initializer())

        Wrk = tf.get_variable("Wrk", [heads, D, d],
                              dtype=tf.float32, initializer=tf.glorot_uniform_initializer())

        Wrk = tf.transpose(Wrk, [1,0,2])
        Wrk = tf.reshape(Wrk, [D, heads*d])


    Q = tf.reshape(Q*Q_mask, [N*qS, D])
    K = tf.reshape(V*V_mask, [N*vS, D])
    V = tf.reshape(V*V_mask, [N*vS, D])

    Q = tf.matmul(Q, Wq)
    K = tf.matmul(K, Wk)
    V = tf.matmul(V, Wv)

    Q = tf.reshape(Q, [N, qS, heads*d])
    K = tf.reshape(K, [N, vS, heads*d])
    V = tf.reshape(V, [N, vS, heads*d])
    
    # Turn to head x N x S x d format

    Q = tf.concat(tf.split(Q, heads, axis=-1), axis=0)  
    K = tf.concat(tf.split(K, heads, axis=-1), axis=0)  
    V = tf.concat(tf.split(V, heads, axis=-1), axis=0)  

    # ATTENTION

    Q = tf.reshape(Q, [heads, N, qS, d])
    Qc = tf.reshape(Q+u, [heads*N, qS, d])

    content_scores = tf.matmul(Qc, tf.transpose(K, [0, 2, 1]))

    PEk = tf.matmul(PE, Wrk)
    REk = generate_relative_embd(qS,vS,PEk)

    REk = tf.reshape(REk, [qS, vS, heads, d])
    REk = tf.transpose(REk, [2, 0, 1, 3])

    Qr = Q+v
    Qr = tf.transpose(Qr, [0, 2, 1, 3])
    position_scores = tf.matmul(Qr, tf.transpose(REk, [0, 1, 3, 2]))
    position_scores = tf.transpose(position_scores, [0, 2, 1, 3])
    position_scores = tf.reshape(position_scores, [heads*N, qS, vS])

    compatibility = content_scores + position_scores

    scalar_d = tf.sqrt(tf.constant(d, tf.float32))

    compatibility = (content_scores + position_scores)/scalar_d

    compatibility = compatibility+attention_mask
    compatibility = tf.nn.softmax(compatibility,axis=-1)

    compatibility = dropout(compatibility, rate=attention_dropout_rate, training=train)

    attended_content = tf.matmul(compatibility, V)

    attended_heads = attended_content
    
    # Convert to form N x S x heads*d
    
    attended_heads = tf.concat(tf.split(attended_heads, heads, axis=0), axis=2)
    attended_heads = tf.reshape(attended_heads, [N*qS, heads*d])

    head_composition = tf.matmul(attended_heads, Wo)

    head_composition = tf.reshape(head_composition, [N, qS, D])
    
    return head_composition

```

## Transformer Encoder Block


```python
def encoder_layer(Q, true_q_len, current_depth, train, name):

    global word_vec_dim
    global fc_dim
    global dropout_rate
    global attention_dropout_rate
    
    D = word_vec_dim
    l = current_depth
    
    N = tf.shape(Q)[0]
    qS = tf.shape(Q)[1]

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE, dtype=tf.float32):

        init = tf.initializers.variance_scaling(scale=1/l, mode='fan_avg', distribution='uniform')

        W1 = tf.get_variable("W1", [D, fc_dim], dtype=tf.float32,
                             initializer=init)
        B1 = tf.get_variable("Bias1", [fc_dim], dtype=tf.float32,
                             initializer=tf.zeros_initializer())

        W2 = tf.get_variable("W2", [fc_dim, D], dtype=tf.float32,
                             initializer=init)
        B2 = tf.get_variable("Bias2", [D], dtype=tf.float32,
                             initializer=tf.zeros_initializer())


    Q = layerNorm(Q, D, name+"/layer_norm1")

    sublayer1 = multiheaded_attention(Q=Q,V=Q,
                                      true_q_len=true_q_len,
                                      true_v_len=true_q_len,
                                      train=train,name=name,
                                      causal=False,
                                      current_depth=current_depth,
                                      attention_dropout_rate=attention_dropout_rate)

    sublayer1 = dropout(sublayer1, rate=dropout_rate, training=train)
    sublayer1 = layerNorm(sublayer1+Q, D, name+"/layer_norm2")

    sublayer2 = tf.reshape(sublayer1, [N*qS, D])
    sublayer2 = gelu(tf.matmul(sublayer2, W1)+B1)
    sublayer2 = tf.matmul(sublayer2, W2)+B2

    sublayer2 = tf.reshape(sublayer2, [N, qS, D])
    sublayer2 = dropout(sublayer2, rate=dropout_rate, training=train)
    sublayer2 = sublayer2 + sublayer1

    return sublayer2

```

## Transformer Decoder Block


```python
def decoder_layer(encoder_Q,decoder_Q, 
                  true_encoder_len,true_decoder_len,
                  timestep,
                  current_depth, train, name):

    global word_vec_dim
    global fc_dim
    global dropout_rate
    global attention_dropout_rate
    
    D = word_vec_dim
    l = current_depth
    
    N = tf.shape(decoder_Q)[0]
    qS = tf.shape(decoder_Q)[1]

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE, dtype=tf.float32):

        init = tf.initializers.variance_scaling(scale=1/l, mode='fan_avg', distribution='uniform')

        W1 = tf.get_variable("W1", [D, fc_dim], dtype=tf.float32,
                             initializer=init)
        B1 = tf.get_variable("Bias1", [fc_dim], dtype=tf.float32,
                             initializer=tf.zeros_initializer())
        W2 = tf.get_variable("W2", [fc_dim, D], dtype=tf.float32,
                             initializer=init)
        B2 = tf.get_variable("Bias2", [D], dtype=tf.float32,
                             initializer=tf.zeros_initializer())

    decoder_Q = layerNorm(decoder_Q, D, name+"/layer_norm1")

    sublayer1 = multiheaded_attention(Q=decoder_Q,V=decoder_Q,
                                      true_q_len=true_decoder_len,
                                      true_v_len=true_decoder_len,
                                      train=train,
                                      name=name+"_self_attention",
                                      causal=True,
                                      current_timestep=timestep,
                                      current_depth=current_depth,
                                      attention_dropout_rate=attention_dropout_rate)

    sublayer1 = dropout(sublayer1, rate=dropout_rate, training=train)
    sublayer1 = layerNorm(sublayer1+decoder_Q, D, name+"/layer_norm2")
    
    sublayer2 = multiheaded_attention(Q=sublayer1,V=encoder_Q,
                                      true_q_len=true_decoder_len,
                                      true_v_len=true_encoder_len,
                                      train=train,
                                      name=name+"_interlayer_attention",
                                      causal=False,
                                      current_timestep=timestep,
                                      current_depth=current_depth,
                                      attention_dropout_rate=attention_dropout_rate)
    
    sublayer2 = dropout(sublayer2, rate=dropout_rate, training=train)
    sublayer2 = layerNorm(sublayer2+sublayer1, D, name+"/layer_norm3")

    sublayer3 = tf.reshape(sublayer2, [N*qS, D])
    sublayer3 = gelu(tf.matmul(sublayer3, W1)+B1)
    sublayer3 = tf.matmul(sublayer3, W2)+B2

    sublayer3 = tf.reshape(sublayer3, [N, qS, D])
    sublayer3 = dropout(sublayer3, rate=dropout_rate, training=train)
    sublayer3 = sublayer3 + sublayer2

    return sublayer3

```

## Encoder


```python
def encode(Q, true_q_len, current_depth, train, name):
    
    global encoder_layers
    layers = encoder_layers
    
    
    Q = dropout(Q, rate=dropout_rate, training=train)

    for t in range(layers):
        Q = encoder_layer(Q=Q, 
                          true_q_len=true_q_len, 
                          current_depth=current_depth+t, 
                          train=train, 
                          name=name+"_"+str(t))
        
    return Q, current_depth+layers
```

## Decoder


```python
def decode(encoder_Q,decoder_Q, true_encoder_len, timestep, current_depth, train, name):
    
    global decoder_layers
    layers = decoder_layers
    
    N = tf.shape(decoder_Q)[0]
    qS = tf.shape(decoder_Q)[1]
    
    true_decoder_len = tf.tile(tf.reshape(qS,[1]),[N])
    
    for t in range(layers):
        decoder_Q = decoder_layer(encoder_Q=encoder_Q,
                                  decoder_Q=decoder_Q, 
                                  true_encoder_len=true_encoder_len,
                                  true_decoder_len=true_decoder_len,
                                  timestep=timestep,
                                  current_depth=current_depth+t,
                                  train=train, 
                                  name=name+"_"+str(t))
        
    return decoder_Q, current_depth+layers
```

## Encoder-Decoder Setup


```python
def encoder_decoder(texts,summaries,
                    true_text_lens,true_summary_lens,
                    train,
                    no_eval):
    
    global vocab2idx
    global word_vec_dim
    global tf_teacher_forcing
    
    GO = vocab2idx['<GO>']
    PRED = vocab2idx['<PRED>']
    
    N = tf.shape(texts)[0]
    D = word_vec_dim
    
    tf_embd = tf.convert_to_tensor(embeddings)
    tf_softmax_wt = tf.transpose(tf_embd,[1,0])
    texts = tf.nn.embedding_lookup(tf_embd,texts)
    
    Q,current_depth = encode(Q=texts,
                            true_q_len=true_text_lens,
                            current_depth=1,
                            train=train,
                            name="Enocder")
    
    encoder_Q = layerNorm(Q, D, "encoder_layer_norm")
    
    decoder_Q = tf.constant([GO,PRED],tf.int32)
    decoder_Q = tf.nn.embedding_lookup(tf_embd,decoder_Q)
    decoder_Q = tf.reshape(decoder_Q,[1,2,D])
    decoder_Q = tf.tile(decoder_Q,[N,1,1])
    
    
    PRED_embd = tf.reshape(decoder_Q[:,-1,:],[N,1,D])
    
    i=tf.constant(0)
                           
    decode_length = tf.cond(no_eval,
                            lambda: tf.constant(22,tf.int32),
                            lambda: tf.shape(summaries)[1])
                           
    logits=tf.TensorArray(size=1, dynamic_size=True, dtype=tf.float32)
    predictions=tf.TensorArray(size=1, dynamic_size=True, dtype=tf.int32)
                           
    
    def cond(i,decoder_Q,logits,predictions):
        return i<decode_length
    
    def body(i,decoder_Q,logits,predictions):

                           
        decoder_Q,_ = decode(encoder_Q=encoder_Q,
                                       decoder_Q=decoder_Q, 
                                       true_encoder_len=true_text_lens, 
                                       timestep=i, 
                                       current_depth=current_depth, 
                                       train=train, 
                                       name="Decoder")

                           
        decoderout = decoder_Q[:,tf.shape(decoder_Q)[1]-1,:]
                           
        out_prob_dist = tf.matmul(decoderout,tf_softmax_wt)
                           
        
        
        pred_idx = tf.cast(tf.argmax(out_prob_dist,axis=-1),tf.int32)
        
        logits = logits.write(i,out_prob_dist)
        predictions = predictions.write(i,pred_idx)
        
        next_idx = tf.cond(tf_teacher_forcing,
                           lambda: summaries[:,i],
                           lambda: pred_idx)
        
        
                           
        next_embd = tf.nn.embedding_lookup(tf_embd,next_idx)
        next_embd = tf.reshape(next_embd,[N,1,D])
                           
        decoder_Q = tf.concat([decoder_Q[:,0:-1,:],next_embd,PRED_embd],axis=1)
           
        
        return i+1,decoder_Q,logits,predictions
    
    _,_,logits,predictions = tf.while_loop(cond,body,[i,decoder_Q,logits,predictions],
                                          shape_invariants=[i.get_shape(),
                                                            tf.TensorShape([None,None,D]),
                                                            tf.TensorShape(None),
                                                            tf.TensorShape(None)])
    
    logits = logits.stack()
    logits = tf.transpose(logits,[1,0,2])
    predictions = predictions.stack()
    predictions = tf.transpose(predictions,[1,0])

    return logits,predictions   
```


```python
# Construct Model
logits, predictions = encoder_decoder(tf_texts,tf_summaries,
                                      tf_text_lens,tf_summary_lens,
                                      tf_train,
                                      tf_no_eval)

#OPTIMIZER
trainables = tf.trainable_variables()
beta=1e-7

regularization = tf.reduce_sum([tf.nn.l2_loss(var) for var in trainables])

pad_mask = tf.sequence_mask(tf_summary_lens, maxlen=tf.shape(tf_summaries)[1], dtype=tf.float32)

cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf_summaries)
cost = tf.multiply(cost,pad_mask) #mask used to remove loss effect due to PADS
cost = tf.reduce_mean(cost) + beta*regularization

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9,beta2=0.98,epsilon=1e-9).minimize(cost)

"""temperature = 0.7
scaled_output = tf.log(logits)/temperature
logits = tf.nn.softmax(scaled_output)"""

#(^Use it with "#prediction_int = np.random.choice(range(vocab_len), p=array.ravel())")

```




    'temperature = 0.7\nscaled_output = tf.log(logits)/temperature\nlogits = tf.nn.softmax(scaled_output)'




```python
import string
import random
import nltk

init = tf.global_variables_initializer()

with tf.Session() as sess: # Start Tensorflow Session
    
    saver = tf.train.Saver() 
    # Prepares variable for saving the model
    sess.run(init) #initialize all variables
    step = 0   
    best_BLEU = 0
    display_step = 500
    epochs = 10
    
    while step < epochs:
           
        batch_len = len(train_batches_x)
        rand_idx = [idx for idx in range(batch_len)]
        random.shuffle(rand_idx)
        #rand_idx = rand_idx[0:2000]
        count=0
        for i in rand_idx: 
            
            batch_size = len(train_batches_x[i])
            
            sample_no = np.random.randint(0,batch_size)
            
            if count%display_step==0:
                print("\nEpoch: "+str(step+1)+" Iteration: "+str(count+1))
                print("\nCHOSEN SAMPLE NO.: "+str(sample_no))
                print("\nSAMPLE TEXT:")
                for vec in train_batches_x[i][sample_no]:
                    print(str(idx2vocab[vec]),end=" ")
                print("\n")
                
            
                
            rand = random.randint(0,4) #determines chance of using Teacher Forcing
            if rand==2:
                random_bool = False
            else:
                random_bool = True

            train_batch_x = np.asarray(train_batches_x[i],np.int32)
            train_batch_y = np.asarray(train_batches_y[i],np.int32)
            train_batch_in_lens = np.asarray(train_batches_in_lens[i],np.int32)
            train_batch_out_lens = np.asarray(train_batches_out_lens[i],np.int32)
            
            #print(train_batch_x.shape)
            #print(train_batch_y.shape)

            # Run optimization operation (backpropagation)
            _,loss,out = sess.run([optimizer,cost,logits],feed_dict={tf_texts: train_batch_x, 
                                                                             tf_summaries: train_batch_y,
                                                                             tf_text_lens: train_batch_in_lens,
                                                                             tf_summary_lens: train_batch_out_lens,
                                                                             tf_train: True,
                                                                             tf_no_eval: False,
                                                                             tf_teacher_forcing: random_bool})
            
            if count%display_step==0:
                print("\nPREDICTED SUMMARY OF THE SAMPLE:\n")
                flag = 0
                for array in out[sample_no]:
                    
                    #prediction_int = np.random.choice(range(vocab_len), p=array.ravel()) 
                    #(^use this if you want some variety)
                    #(or use this what's below:)
                    
                    prediction_int = np.argmax(array)
                    
                    if vocab[prediction_int] in string.punctuation or flag==0: 
                        print(str(vocab[prediction_int]),end='')
                    else:
                        print(" "+str(vocab[prediction_int]),end='')
                    flag=1
                print("\n")
                
                print("ACTUAL SUMMARY OF THE SAMPLE:\n")
                for vec in train_batches_y[i][sample_no]:
                    print(str(idx2vocab[vec]),end=" ")
                print("\n")
            
                print("loss="+str(loss))
                
            count+=1
                
        print("\n\nSTARTING VALIDATION\n\n")
                
        batch_len = len(val_batches_x)
        #print(batch_len)
        total_BLEU_argmax=0
        
        total_len=0
        for i in range(0,batch_len):
            
            batch_size = len(val_batches_x[i])
            
            sample_no = np.random.randint(0,batch_size)

            if i%display_step==0:
                print("\nEpoch: "+str(step+1)+" Iteration: "+str(i+1))
                print("\nCHOSEN SAMPLE NO.: "+str(sample_no))
                print("\nSAMPLE TEXT:")
                for vec in val_batches_x[i][sample_no]:
                    print(str(idx2vocab[vec]),end=" ")
                print("\n")
                
            
            val_batch_x = np.asarray(val_batches_x[i],np.int32)
            val_batch_y = np.asarray(val_batches_y[i],np.int32)
            val_batch_in_lens = np.asarray(val_batches_in_lens[i],np.int32)
            val_batch_out_lens = np.asarray(val_batches_out_lens[i],np.int32)
       
            loss,out = sess.run([cost,logits],feed_dict={tf_texts: val_batch_x, 
                                                         tf_summaries: val_batch_y,
                                                         tf_text_lens: val_batch_in_lens,
                                                         tf_summary_lens: val_batch_out_lens,
                                                         tf_no_eval: False,
                                                         tf_train: False,
                                                         tf_teacher_forcing: False})
            
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

                        if idx2vocab.get(gold_idx, '<UNK>') == "<EOS>":
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
                    
                    if vocab[prediction_int] in string.punctuation or flag==0: 
                        print(str(vocab[prediction_int]),end='')
                    else:
                        print(" "+str(vocab[prediction_int]),end='')
                    flag=1
                print("\n")
                
                print("ACTUAL SUMMARY OF THE SAMPLE:\n")
                for vec in val_batches_y[i][sample_no]:
                    print(str(idx2vocab[vec]),end=" ")
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
    
```

    
    Epoch: 1 Iteration: 1
    
    CHOSEN SAMPLE NO.: 5
    
    SAMPLE TEXT:
    omg - ca n't even believe these are sugar free ! lots of great flavors , there 's not one i would choose ove the other . fresh ! these are really a treat . i wish they came in smaller bags because they are hard to put down once the bag is open . these disappear quickly at our house . great product . 
    
    
    PREDICTED SUMMARY OF THE SAMPLE:
    
    <PRED> <PRED> <PRED> <PRED> <PRED> <PRED> <PRED> <PRED> <PRED> <PRED> <PRED>
    
    ACTUAL SUMMARY OF THE SAMPLE:
    
    yummy ! ! <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=122.70911
    
    Epoch: 1 Iteration: 501
    
    CHOSEN SAMPLE NO.: 4
    
    SAMPLE TEXT:
    i recently developed allergies to <UNK> <UNK> , and i was devastated ! that is , until i found sunbutter ... omg is all i have to say . and from someone who used ot eat peanut butter like it was going out of style , this is waaaay better ! its crunchy , `` nutty , '' yummy goodness ! even my non allergic husband , family members , and friends love this ... i 've made converts out of everyone ! 
    
    
    PREDICTED SUMMARY OF THE SAMPLE:
    
    great good taste <EOS> this!! <EOS> <EOS>!!!
    
    ACTUAL SUMMARY OF THE SAMPLE:
    
    even non <UNK> love this ! ! <EOS> <PAD> <PAD> <PAD> <PAD> 
    
    loss=2.856789
    
    Epoch: 1 Iteration: 1001
    
    CHOSEN SAMPLE NO.: 29
    
    SAMPLE TEXT:
    my husband and i had n't had beef jerky for years ( since we became vegetarians and then pescatarians ) , and this stuff was a great find for us ! yay ! 
    
    
    PREDICTED SUMMARY OF THE SAMPLE:
    
    delicious jerky jerky a dog <EOS> <UNK>! <EOS>
    
    ACTUAL SUMMARY OF THE SAMPLE:
    
    perfect jerky for the pescatarian ... <EOS> <PAD> <PAD> 
    
    loss=2.5716243
    
    Epoch: 1 Iteration: 1501
    
    CHOSEN SAMPLE NO.: 27
    
    SAMPLE TEXT:
    these popchips are the best and they only have 100 calories per bag ! i can have a bag everyday they are so good . 
    
    
    PREDICTED SUMMARY OF THE SAMPLE:
    
    great <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS>
    
    ACTUAL SUMMARY OF THE SAMPLE:
    
    yum yum <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=1.9416369
    
    Epoch: 1 Iteration: 2001
    
    CHOSEN SAMPLE NO.: 30
    
    SAMPLE TEXT:
    strong coffee with average coffee taste . in the starbucks category as i see it . 
    
    
    PREDICTED SUMMARY OF THE SAMPLE:
    
    good good <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS>
    
    ACTUAL SUMMARY OF THE SAMPLE:
    
    not bad <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=1.5254364
    
    Epoch: 1 Iteration: 2501
    
    CHOSEN SAMPLE NO.: 24
    
    SAMPLE TEXT:
    i actually tasted the <UNK> flavor . if you 've ever eaten a fig newton , then try to imagine a fig newton , minus the fig center , with rice crispies mixed up in it . it tastes a little grainy at first , but once i read the ingredients and realized that the grains were puffed rice , i was more at ease . really a good taste . one of the better gluten <UNK> <UNK> free cookies i 've eaten . 
    
    
    PREDICTED SUMMARY OF THE SAMPLE:
    
    good taste <EOS> but good <EOS> it good taste to! eat <EOS> <EOS> <EOS> <EOS>
    
    ACTUAL SUMMARY OF THE SAMPLE:
    
    good taste , but texture takes a little getting used to . <EOS> <PAD> <PAD> <PAD> 
    
    loss=2.1366303
    
    Epoch: 1 Iteration: 3001
    
    CHOSEN SAMPLE NO.: 7
    
    SAMPLE TEXT:
    these people are really fast . i just placed the order last night and it is in the mail this morning . the good service is really appreciated 
    
    
    PREDICTED SUMMARY OF THE SAMPLE:
    
    great <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS>
    
    ACTUAL SUMMARY OF THE SAMPLE:
    
    zowie ! <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=1.9405246
    
    Epoch: 1 Iteration: 3501
    
    CHOSEN SAMPLE NO.: 21
    
    SAMPLE TEXT:
    much smaller cut than other premium teas i have bought . taste is okay - you will need a fine mesh filter . 
    
    
    PREDICTED SUMMARY OF THE SAMPLE:
    
    not <EOS> like not <EOS> <EOS> a <EOS> of <EOS> <EOS> <EOS> <EOS>
    
    ACTUAL SUMMARY OF THE SAMPLE:
    
    okay taste but cut fine for a loose tea <EOS> <PAD> <PAD> <PAD> 
    
    loss=1.5413034
    
    Epoch: 1 Iteration: 4001
    
    CHOSEN SAMPLE NO.: 7
    
    SAMPLE TEXT:
    these spring roll wraps are delicious . easy to work with if you do n't leave them in the water too long when you soften them to roll . lol 
    
    
    PREDICTED SUMMARY OF THE SAMPLE:
    
    delicious! <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS>
    
    ACTUAL SUMMARY OF THE SAMPLE:
    
    delicious <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=1.5407957
    
    Epoch: 1 Iteration: 4501
    
    CHOSEN SAMPLE NO.: 11
    
    SAMPLE TEXT:
    these bags started off okay ... but once we got to the 3rd refill we began to notice that many of the bags either a ) did n't open at all or b ) both ends were open ! we figured maybe it was just one refill , but in four of these refills we have had the same problem . we had to throw away about 1/3 of the bags or try to use them even though they are defective . will not be buying these in the future . 
    
    
    PREDICTED SUMMARY OF THE SAMPLE:
    
    not <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS> <EOS>
    
    ACTUAL SUMMARY OF THE SAMPLE:
    
    disappointed <EOS> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> 
    
    loss=1.5031999

## This part is untested

```python
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
        
        batch_size = len(test_batches_x[i])

        sample_no = np.random.randint(0,batch_size)

        if i%display_step==0:
            print("\nEpoch: "+str(step+1)+" Iteration: "+str(i+1))
            print("\nCHOSEN SAMPLE NO.: "+str(sample_no))
            print("\nSAMPLE TEXT:")
            for vec in test_batches_x[i][sample_no]:
                print(str(idx2vocab[vec]),end=" ")
            print("\n")

        test_batch_x = np.asarray(test_batches_x[i],np.int32)
        test_batch_y = np.asarray(test_batches_y[i],np.int32)
        test_batch_in_lens = np.asarray(test_batches_in_lens[i],np.int32)
        test_batch_out_lens = np.asarray(test_batches_out_lens[i],np.int32)

        loss,out = sess.run([cost,logits],feed_dict={tf_texts: test_batch_x, 
                                                         tf_summaries: test_batch_y,
                                                         tf_text_lens: test_batch_in_lens,
                                                         tf_summary_lens: test_batch_out_lens,
                                                         tf_no_eval: False,
                                                         tf_train: False,
                                                         tf_teacher_forcing: False})

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

                    if idx2vocab.get(gold_idx, '<UNK>') == "<EOS>":
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

                if vocab[prediction_int] in string.punctuation or flag==0: 
                    print(str(vocab[prediction_int]),end='')
                else:
                    print(" "+str(vocab[prediction_int]),end='')
                flag=1
            print("\n")

            print("ACTUAL SUMMARY OF THE SAMPLE:\n")
            for vec in test_batches_y[i][sample_no]:
                print(str(idx2vocab[vec]),end=" ")
            print("\n")

            print("loss="+str(loss))
            print("BLEU-2=",batch_BLEU_argmax_list[sample_no])

    avg_BLEU = total_BLEU_argmax/total_len
    print("AVERAGE TEST BLEU:",avg_BLEU)

    
```


```python

```

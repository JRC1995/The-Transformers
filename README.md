# Attention - all that's needed?

The paper "[All you need is Attention](https://arxiv.org/abs/1706.03762)" proposes a novel approach to machine translation with the encoder decoder architecture, but solely with attention mechanisms i.e no complex layers having recurrence or convolutions.

The authors of the paper describes attention as such:

>An attention function can be described as mapping a query and a set of key-value pairs to an output,
>where the query, keys, values, and output are all vectors. The output is computed as a weighted sum
>of the values, where the weight assigned to each value is computed by a compatibility function of the
>query with the corresponding key.

This implementation is mostly based on the model as proposed in the paper, but trained for abstractive summarization (like [this](https://github.com/JRC1995/Abstractive-Summarization) and [this](https://github.com/JRC1995/Attention-Everywhere)) on [Amazon Fine-Food reviews dataset](https://www.kaggle.com/snap/amazon-fine-food-reviews/data).

Unlike the previous models ([this](https://github.com/JRC1995/Abstractive-Summarization) and [this](https://github.com/JRC1995/Attention-Everywhere)) I used for abstractive summarization, this one is set up for mini-batch training.

Note:

* I am using different hyperparameters. 
* I am using low dimensional GloVe embeddings, and less no. of layers.
* I haven't used regularization.
* I will not be completely training this model for now. I will not be validating or testing either. 

Overall, this is more or less a 'toy' implementation.

# Decoding

I haven't implemented the decoding process in the exact way as it was done in original model (truthfully, I do have some confusions about the exact details of the decoder in the original model - so I just made my own adjustments). 

I iterated the decoder layers for max_len no. of times when max_len is the total output sequence length. At each iteration, an output vector is predicted in the form batch_size x model_dimensions (later to be linearly transformed to express a probability distribution over the vocabularies). At the end of each iterations, the currently prediced output is concatenated with the previously predicted output vectors to get a sequence of output vectors in the shape batch_size x sequence_length x model_dimensions.

This newly formed sequence of vector of shape batch_size x timestep x model_dimensions (the timestep increases every iteration) is coverted to batch_size x max_len x model_dimensions (max_len = length of the output sequence) by padding. This is then fed to the decoder layer as input at the next iteration.  

Masking was used so that the network doesn't attend to the padded word vectors beyond the current timestep position in the sequence. 

The decoder layers are such that if the input is in the shape batch_size x sequence_length x model_dimensions then the output will be of the same shape.

But I needed a single output vector for each training data in the batch, that should represent the single next predicted word. That is, I needed a decoder output of shape batch_size x model_dimensions, but I get the shape: batch_size x max_len x model_dimensions. 

I reshaped the immediate decoder output to batch_size x (max_len * model_dimensions) and passed it through a simple fully connected feed forward network to get the desired outcome: batch_size x model_dimensions. Next, I applied positional encoding to the result. After all these, we reach the end of the iteration where the current output is concatenated with the previous ones for the next loop. 

The inital candidate output vector is initialized with ones for each data in the batch. This initial output vector is the first input to the first decoder sublayer of the first decoder layer of the first iteration.

At the end of the first iteration, the predicted output is not concatenated with the initial output vector; rather, the initial unit-initalized output vector is replaced by the newly predicted output. So, there are some exceptions to the rules (as stated before) for the first iteration.  

Finally all the predicted output vectors are converted into probability distributions from which predictions are made and losses are calculated for backpropagation. 

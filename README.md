# Attention - all that's needed?

The paper "[All you need is Attention](https://arxiv.org/abs/1706.03762)" proposes a novel approach to machine translation with the encoder decoder architecture, but solely with attention mechanisms i.e no complex layers having recurrence or convolutions.

The authors of the paper describes attention as such:

>An attention function can be described as mapping a query and a set of key-value pairs to an output,
>where the query, keys, values, and output are all vectors. The output is computed as a weighted sum
>of the values, where the weight assigned to each value is computed by a compatibility function of the
>query with the corresponding key.

This implementation is mostly based on the model as proposed in the paper, but trained for abstractive summarization (like [this](https://github.com/JRC1995/Abstractive-Summarization) and [this](https://github.com/JRC1995/Attention-Everywhere)) on Amazon Fine-Food reviews dataset.

Unlike the previous models ([this](https://github.com/JRC1995/Abstractive-Summarization) and [this](https://github.com/JRC1995/Attention-Everywhere)) I used for abstractive summarization, this one is set up for mini-batch training.

Note:

* I am using different hyperparameters. 
* I am using low dimensional GloVe embeddings, and less no. of layers.
* I haven't used regularization.
* I will not be completely training this model for now. I will not be validating or testing either. 

Overall, this is more or less a 'toy' implementation.

# Decoding

(under construction)

Code to be uploaded soon. 

# Attention - all that's needed?

The paper "[All you need is Attention](https://arxiv.org/abs/1706.03762)" proposes a novel approach to machine translation with the encoder decoder architecture, but solely with attention mechanisms i.e no complex layers having recurrence or convolutions.

The authors of the paper describes attention as:

>An attention function can be described as mapping a query and a set of key-value pairs to an output,
>where the query, keys, values, and output are all vectors. The output is computed as a weighted sum
>of the values, where the weight assigned to each value is computed by a compatibility function of the
>query with the corresponding key.

This implementation is mostly based on the model as proposed in the paper, but trained for abstractive summarization (like [this](https://github.com/JRC1995/Abstractive-Summarization) and [this](https://github.com/JRC1995/Attention-Everywhere)) on [Amazon Fine-Food reviews dataset](https://www.kaggle.com/snap/amazon-fine-food-reviews/data).

Unlike the previous models ([this](https://github.com/JRC1995/Abstractive-Summarization) and [this](https://github.com/JRC1995/Attention-Everywhere)) I used for abstractive summarization, this one is set up for mini-batch training.


## Tested with:

* Python 3.6
* Tensorflow 1.12


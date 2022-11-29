# Transformer
# SciFormer

Minimal multi-head self-attention transformer architecture implemented in PyTorch. Probably useful for rapid prototyping and education purposes.

# Transformer Network

## Self-attention

At its core, the self-attention (SA) mechanism is a sequence-to-sequence operation.

As an aside, self-attention is probably the critical component that allow transformer architecture to demonstrate the ability of "in-context" learning during inference. This means that transformer neural networks learn from the activations at runtime without having to update their weights. Aside end.

In a neural network, a self-attention module is a sequence-to-sequence layer mapping a sequence of input tokens $X = \{\mathbf{x}_1, \cdots, \mathbf{x}_n\}$ to a sequence of output tokens $Y = \{\mathbf{y_1}, \cdots, \mathbf{y}_n\}$. Here, $X$ and $Y$ represent matrices of the same dimension $n \times k$ where $n$ represents the sequence length and $k$ the token dimension.

A sequence consists of tokens. In case of a sentence, tokens can be, for example, the sentence's letters or words.

The SA mechanism processes token sequences in parallel and can learns dependencies between tokens over long time windows.

The most basic version of self-attention is a weighted sum over the input vectors $\{\mathbf{x}_1, \cdots, \mathbf{x}_n\}$ resulting in one self-attention output $\mathbf{y}_i$ for every input $\mathbf{x}_i$.

$$\mathbf{y}_i = \sum_{j=1}^n w_{ij} \mathbf{x}_j$$

The scalar weight $w_{ij}$ in the weighted sum above is a function derived from the input vectors $\mathbf{x}_i$ and $\mathbf{x}_j$. A simple way to compute the weight parameter is to use the dot-product

$$w'_{ij} = f(\mathbf{x}_i,\mathbf{x}_j) = \sum_{l=1}^k x_{il}x_{jl} = \mathbf{x}_i^\top\mathbf{x}_j$$

However, we are not finished yet. Modern transformer implementations apply the softmax operation row-wise to the computed weight matrix $W$. This leads to

$$w_{ij} = \frac{\exp(w'_{ij})}{\sum_{j=1}^k \exp(w'_{ij})}$$

> TODO: $$w_{ij} = \frac{w'_{ij} - \mu_{w[i,:]}}{\sigma_{w[i,:]}}$$
> ES inspired.

From how we compute the weights $w_{ij}$ for each input token, it is apparent, that the computational complexity of self-attention layers with this kind of weight computation is of the order of $\mathcal{O}(n^2)$. This already shows, that for transformer networks processing very long sequences, the self-attention operation is a primary bottle neck.

It should also be noticed, that, so far, no trainable parameters have been used in the operations outlined above.

The basic version of self-attention can also be expressed in matrix notation

$$Y = WX^\top = (X^\top X) X^\top$$

This shows, that self-attention is linear operation between the input tokens $X$ and output tokens $Y$ and a non-linear operation via $W$.


## ImageTransformer

This image transformer implementation uses a simple configuration of stacked transformer blocks without encoder-decoder structure and consists of three main blocks: 

The `ImageToSequence` module transforms image data of shape `(channels, height, width)` to a sequence of token embeddings of shape `(sequence_length, embedding_dim)`. The image to sequence transformation is implemented using a `Conv2d` operation as patch embedding with `kernel_size` = `stride`. Patch embedding downsamples the image and allows transformer architectures even for images of high resolution. The `ImageToSequence` module is purely linear.

A sequence of stacked `TransformerBlock`s consisting of a `MultiHeadSelfAttention` module followed by a standard fully connected neural network.

Finally, a `Classifier` module takes the output of the last `TransformerBlock` and applies a linear transformation to the network's final prediction.

# Weight Visualization

## Patch Embedding

## Positional Embedding

## Attention Mask 


# Random search

A simple random search method has been implemented to explore the model in more depth and to better understand the interplay of the model's various hyperparameters such as

- sequence length
- token embedding dimensions
- number of attention heads
- number of transformer blocks
- hidden expansion
- dropout probability
- use of bias in self-attention module
- learning rate

![Random search](./docs/images/hparams_random_search.png)

# Discussion

Running a few experiments showed that the model's test accuracy improves with an increased number of attention heads, head dimension, and sequence length (in that order). Adding more attention blocks always resulted in higher test accuracy.

# TODO:

- Add mask to self-attention
- Add option to have trainable embedding / mask
- Make attention fully "in-context"
    - Use activations to create bias terms
    - $b = W_x x$

# References

[1] *An Image is Worth 16x16 Words*: https://arxiv.org/pdf/2010.11929.pdf

# License

MIT
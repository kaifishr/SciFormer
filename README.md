# **SciFormer**

Minimal multi-head self-attention transformer architecture with experimental features implemented in PyTorch. Probably useful for rapid prototyping and educational purposes.

<p align="center">
    <img src="./docs/images/sciformer.jpeg" height="320">
</p>

# Transformer Network


## Transformer Neural Network

Transformer neural networks are sequence-based models that use at its core the self-attention mechanism to propagate information between the tokens present in the input sequence. Tokens represent basic units that can be words or letter in case of a language model, pixels or patch embeddings in case of an vision transformer to name just a few possibilities.


## Self-attention

The self-attention mechanism is the fundamental operations of transformer neural networks. At its core, the self-attention mechanism is a sequence-to-sequence operation.

As an aside, self-attention is probably the critical component that allows the transformer architecture to demonstrate the ability of "in-context" learning during inference. This means that transformer neural networks learn from the activations at runtime without having to update their weights. It is assumed that large transformer-based language models discover during their training to implicitly implement other models in their hidden activations. Prompting can therefore be a kind of fine-tuning that is running inside of a language model. Aside end.

In a neural network, a self-attention module is a sequence-to-sequence layer mapping a sequence of input tokens $X = \{\mathbf{x}_1, \cdots, \mathbf{x}_n\}$ to a sequence of output tokens $Y = \{\mathbf{y_1}, \cdots, \mathbf{y}_n\}$. Here, $X$ and $Y$ represent matrices of the same dimension $n \times k$ where $n$ represents the sequence length and $k$ the token dimension.

A sequence consists of tokens. In case of a sentence, tokens can be, for example, the sentence's letters or words.

The SA mechanism processes token sequences in parallel and can learns dependencies between tokens over long time windows.

The most basic version of self-attention is a weighted sum over the input vectors $\{\mathbf{x}_1, \cdots, \mathbf{x}_n\}$ resulting in one self-attention output $\mathbf{y}_i$ for every input $\mathbf{x}_i$.

$$\mathbf{y}_{i} = \sum_{j=1}^{n} w_{ij} \mathbf{x}_j$$

The scalar weight $w_{ij}$ in the weighted sum above is a function derived from the input vectors $\mathbf{x}_i$ and $\mathbf{x}_j$. A simple way to compute the weight parameter is to use the dot-product

$$w'_{ij} = f(\mathbf{x}_i,\mathbf{x}_j) = \sum_{l=1}^k x_{il}x_{jl} = \mathbf{x}_i^\top\mathbf{x}_j$$

After computing the dot product for all input tokens, we end up with a weight matrix $W$ of dimensions $n \times n$. This shows, that the computational complexity of self-attention layers with this kind of weight computation is of the order of $\mathcal{O}(n^2)$. Thus, the self-attention operation is a primary bottle neck for transformer networks processing very long sequences.

Modern transformer implementations apply the *Softmax* operation row-wise to the computed weight matrix $W$. As the *Softmax* operation is sensitive to large values leading to potentially small gradients of the softmax function, slowing down learning considerably, we normalize the dot-product by the square root of the input dimension $k$ to constraint the distribution of the attention weights $W$ to have a standard deviation of 1. Thus we compute

$$w'_{ij} = \frac{\mathbf{x}_i^\top\mathbf{x}_j}{\sqrt{k}}$$

This leads to

$$w_{ij} = \frac{\exp(w'_{ij})}{\sum_{j=1}^k \exp(w'_{ij})}$$

It should also be noticed, that, so far, no trainable parameters have been used in the operations outlined above.

The basic version of self-attention can also be expressed in matrix notation

$$Y = WX^\top = (\textcolor{red}{X^\top} \textcolor{green}{X}) \textcolor{blue}{X^\top}$$

This shows, that self-attention is linear operation between the input tokens $X$ and output tokens $Y$ and a non-linear operation via $W$. The three components on the right hand side are also known as the <span style="color:red">*query*</span>, <span style="color:green">*key*</span>, and <span style="color:blue">*value*</span>. In contrast to the basic attention mechanism, the *key*, *query*, and *value* come from the same set of input tokens $X$.

From the self-attention formulation above we can see, that the attention mechanism has no problems looking far back into the input sequence as every input token has the same distance to every output token. 

As the self-attention mechanism at its core is a weighted sum over the input vectors $\mathbf{x}_i$, the self-attention mechanism does not see the input tokens as a sequence but rather as a set. This means, that self-attention is permutation equivariant, as the position of the input token in the sequence does not change the output of the self-attention layer. Therefore it holds that

$$\text{permutation}(\text{selfAttention}(\mathbf{x})) = \text{selfAttention}(\text{permutation}(\mathbf{x}))$$

In which order the sequence is to be understood, can be achieved by encoding the sequential structure into the token embeddings using positional embeddings.


### Attention as a Soft Dictionary

To get a better understanding of the *query*, *key*, and *value* nomenclature in the context of self-attention, the attention mechanism can be seen as a soft form of a dictionary. Let's consider the following example with a Python dictionary:

$$\texttt{d = \{"a": 1, "b": 2\}}$$

Here, the dictionary consists of two key-value pairs, where $\texttt{a}$ and $\texttt{b}$ are the keys to the values $\texttt{1}$ and $\texttt{2}$. The request for accessing the data stored in the dictionary is called a query. We access the value of a dictionary using the key associated with that value. The following line shows a query requesting data using key $\texttt{a}$

$$\texttt{d["a"] = 1}$$

The example above show a "hard" dictionary, as the query gets you exactly the value that is associated with the key. The attention mechanism, on the other hand, allows for a key to match the query to some extend as determined by the respective dot product. Thus, a mixture of all values is returned (therefore soft dictionary) with softmax normalized dot products as mixture weights.

It should also be noted, that in general a query can either be a request for retrieving data or to perform an action on the data, or both.


### Trainable / Parameterized Self-Attention

As already mentioned, there are so far no trainable parameters in the self-attention operations outlined above. Introducing parameters that allow obtain trainable representations of *query* $\mathbf{q}$, *key* $\mathbf{k}$, and *value* $\mathbf{v}$, allows the attention mechanism to be more expressive and to learn more powerful representations of the input sequence.

To achieve this, we use weight matrices $W_Q$, $W_K$, and $W_V$ of dimension $k \times k$ for transforming every $\mathbf{x}_i$ linearly to get $\mathbf{q}_i$, $\mathbf{k}_i$, and $\mathbf{v}_i$

$$
\mathbf{q}_i = W_Q \mathbf{x}_i + \mathbf{b}_Q \\
\mathbf{k}_i = W_K \mathbf{x}_i + \mathbf{b}_K \\
\mathbf{v}_i = W_V \mathbf{x}_i + \mathbf{b}_V
$$


### Multi-head Self-Attention

The above formulation represents single-head self-attention, as the entire sequence $X$ is fed into the weight matrices $W_Q$, $W_K$, and $W_V$ to compute a new sequence $Y$. However, the computation can be split into different heads that perform the self-attention operation in parallel. This approach is intended to result in a network that models different relations between input tokens.

There are different flavors of how multi-head self-attention. One approach is to run through the entire scaled dot-product attention multiple times in parallel. Other approaches split the input sequence $X$ into $h$ chunks of the same size before running the attention on the subsequences in parallel. Running the attention on subsequences of the input, there a smaller weight matrices $W_Q$, $W_K$, and $W_V$ involved, allowing to run multi-head self-attention at about the same costs as single-head self-attention.

More concretely, this means that we split the input embedding dimension $n$ of $X$ into $h$ chunks of same size:

$$X = \{X_{[:,0:k/h]}, \cdots, X_{[:,k-k/h:k]}\}$$


### Masking

Autoregressive models predict the next token in a sequence such as, for example, a letter or a word. For this to work, the attention mechanism needs to be causal. This is achieved by removing the forward connections in the self-attention operations which is also known as masking. To remove the forward connections, we can add a mask of minus infinity to the entries above the diagonal. Thus, we add mask to the weight matrix $W'$ and get

$$W' =
\begin{pmatrix}
w_{11}  & \cdots & w_{1k} \\
\vdots & \ddots & \vdots \\
w_{k1} & \cdots & w_{kk}
\end{pmatrix}
+
\begin{pmatrix}
0 & -\infty & -\infty  \\
\vdots & \ddots & -\infty \\
0 & \cdots & 0 
\end{pmatrix}
=
\begin{pmatrix}
w_{11}  & -\infty & -\infty \\
\vdots & \ddots & -\infty \\
w_{k1} & \cdots & w_{kk}
\end{pmatrix}
$$


### Encoding Sequential Structure

The meaning of words often depends on their position in a sentence. As the attention mechanism is permutation equivariant, we need to tell the attention module about the structure of the input sequence. There are different techniques that allow the network to become aware of the words position. Two simple methods are

- **Position embedding** works by adding an embedding vector to every embedded token of the input sequence. However, this approach fixes the sequence length that can be processed by the model.

- **Position encoding** adds a predictable pattern to the embedded tokens. Here the idea is, that the model learns what the position encoding should look like for sequences longer than seen during training.


# Transformer Types



## TextTransformer

Transformer neural networks used for text generation are autoregressive models. However, autoregressive models are not limited to generating text but can also used to generate speech, music to name but a few. An autoregressive model receives a sequence of input tokens and predicts a probability distribution over the next index in the sequence.


## ImageTransformer

This image transformer implementation uses a simple configuration of stacked transformer blocks without encoder-decoder structure and consists of three main blocks: 

The `ImageToSequence` module transforms image data of shape `(channels, height, width)` to a sequence of token embeddings of shape `(sequence_length, embedding_dim)`. The image to sequence transformation is implemented using a `Conv2d` operation as patch embedding with `kernel_size` = `stride`. Patch embedding downsamples the image and allows transformer architectures even for images of high resolution. The `ImageToSequence` module is purely linear.

A sequence of stacked `TransformerBlock`s consisting of a `MultiHeadSelfAttention` module followed by a standard fully connected neural network.

Finally, a `Classifier` module takes the output of the last `TransformerBlock` and applies a linear transformation to the network's final prediction.


# Weight Visualization


## Patch Embedding


## Positional Embedding


## Attention Mask 


# TODO:

- Make attention fully "in-context"
    - Use activations to create bias terms
    - $b = W_x x$


# References

[1] [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)

[2] [*An Image is Worth 16x16 Words*](https://arxiv.org/pdf/2010.11929.pdf)

[3] [*minGPT*](https://github.com/karpathy/minGPT)

[4] [*Transformers from scratch*](https://peterbloem.nl/blog/transformers)


# License

MIT
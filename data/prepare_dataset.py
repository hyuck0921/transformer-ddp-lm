"""Prepare toy dataset for language modeling."""

import os
import argparse
from pathlib import Path


# Sample text corpus (can be replaced with any text)
SAMPLE_TEXT = """
The Transformer is a deep learning architecture that has revolutionized natural language processing.
It was introduced in the paper "Attention is All You Need" by Vaswani et al. in 2017.

The key innovation of the Transformer is the self-attention mechanism, which allows the model to weigh
the importance of different words in a sequence when processing each word. This enables the model to
capture long-range dependencies and contextual relationships more effectively than previous architectures
like recurrent neural networks (RNNs) and long short-term memory networks (LSTMs).

The Transformer architecture consists of an encoder and a decoder, each composed of multiple layers.
Each layer contains two main components: a multi-head self-attention mechanism and a position-wise
feed-forward network. The multi-head attention allows the model to attend to different aspects of
the input simultaneously, while the feed-forward network processes the attended information.

One of the major advantages of the Transformer is its ability to parallelize computation, as it does
not rely on sequential processing like RNNs. This makes it much faster to train on modern hardware,
especially GPUs. The architecture has become the foundation for many state-of-the-art models in NLP,
including BERT, GPT, and their variants.

The Transformer has also been successfully applied to other domains beyond NLP, such as computer vision
(Vision Transformer, or ViT) and reinforcement learning. Its flexibility and effectiveness have made it
one of the most important innovations in modern deep learning.

In a Transformer, positional encodings are added to the input embeddings to provide information about
the position of tokens in the sequence, since the self-attention mechanism itself is position-invariant.
These encodings can be learned or fixed, with sinusoidal functions being a common choice for fixed encodings.

The attention mechanism in the Transformer computes three vectors for each input token: query, key, and value.
The attention score between two tokens is computed as the dot product of the query vector of one token and
the key vector of another. These scores are normalized using a softmax function and used to compute a weighted
sum of the value vectors, producing the attention output.

Multi-head attention extends this idea by computing multiple sets of attention outputs in parallel, each with
different learned projection matrices. This allows the model to capture different types of relationships between
tokens. The outputs of all attention heads are concatenated and linearly transformed to produce the final output.

The feed-forward network in each Transformer layer typically consists of two linear transformations with a
non-linear activation function (usually ReLU or GELU) in between. This network is applied independently to
each position in the sequence, hence the name "position-wise" feed-forward network.

Layer normalization and residual connections are key components that help stabilize training and enable
the construction of very deep Transformer models. Residual connections allow gradients to flow directly
through the network, while layer normalization helps reduce internal covariate shift.

The decoder in the Transformer has an additional attention mechanism called cross-attention, which allows
it to attend to the encoder's output. This is crucial for sequence-to-sequence tasks like translation.
The decoder also uses masked self-attention to prevent attending to future tokens during training.

Training a Transformer typically involves techniques like learning rate warmup, where the learning rate
is gradually increased at the beginning of training, followed by decay. This helps stabilize training
and achieve better convergence. Label smoothing is often used to prevent the model from becoming
overconfident in its predictions.

The success of the Transformer has led to the development of many variants and improvements. Some models
focus on improving efficiency, such as Longformer and Reformer, which reduce the quadratic complexity of
self-attention. Others, like ELECTRA and ALBERT, aim to improve pre-training efficiency and effectiveness.

Despite its success, the Transformer has some limitations. The quadratic complexity of self-attention with
respect to sequence length makes it challenging to process very long sequences. The lack of built-in inductive
biases for certain tasks, such as understanding hierarchical structure, can also be a disadvantage compared
to architectures specifically designed for those tasks.

Research continues to address these limitations and explore new applications of the Transformer architecture.
Recent work has focused on making Transformers more efficient, interpretable, and capable of handling longer
contexts. The ongoing development of Transformer-based models promises to bring further advances in artificial
intelligence and machine learning.
"""


def create_toy_dataset(output_path: str, text: str = SAMPLE_TEXT, repeat: int = 100):
    """
    Create a toy text dataset.
    
    Args:
        output_path: Path to save the dataset
        text: Text corpus to use
        repeat: Number of times to repeat the text (to make dataset larger)
    """
    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Repeat text to make it larger
    full_text = text.strip() * repeat
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    # Print statistics
    num_chars = len(full_text)
    num_words = len(full_text.split())
    unique_chars = len(set(full_text))
    
    print(f"Created toy dataset at {output_path}")
    print(f"  - Total characters: {num_chars:,}")
    print(f"  - Total words: {num_words:,}")
    print(f"  - Unique characters: {unique_chars}")
    print(f"  - Text repeated: {repeat} times")


def main():
    parser = argparse.ArgumentParser(description='Prepare toy dataset')
    parser.add_argument('--output', type=str, default='data/toy_dataset.txt',
                        help='Output file path')
    parser.add_argument('--repeat', type=int, default=100,
                        help='Number of times to repeat the text')
    parser.add_argument('--custom-text', type=str, default=None,
                        help='Path to custom text file (optional)')
    
    args = parser.parse_args()
    
    # Load custom text if provided
    if args.custom_text is not None:
        with open(args.custom_text, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Loaded custom text from {args.custom_text}")
    else:
        text = SAMPLE_TEXT
        print("Using default sample text")
    
    # Create dataset
    create_toy_dataset(args.output, text, args.repeat)


if __name__ == '__main__':
    main()


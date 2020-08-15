# Image Captioning with RNN-based Attention

We introduce an attention based model that automatically learns to generate a caption for images. Our model consists of a novel attention module which includes an elegant modification of GRU architecture. We validate the use of our attention model on a benchmark dataset MS COCO (2017), and compare its performance with other sate-of-the-art models. Our proposed model has the BLEU-1 score of 74.0.

More details can be found in the following file:
> Image_Captioning_with_GRU_based_Attention.pdf



## Proposed Model
Our proposed attention model for image captioning, consisting of CNN, Attention Module, and LSTM. The attention module has two main components. A MLP to compute the attention weights and a attention GRU module which aims to provide a contextual representation that allows logical reasoning over interesting regions.

![GitHub Logo](/images/Proposed_model.png)


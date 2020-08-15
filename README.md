# Image Captioning with RNN-based Attention

We introduce an attention based model that automatically learns to generate a caption for images. Our model consists of a novel attention module which includes an elegant modification of GRU architecture. We validate the use of our attention model on a benchmark dataset MS COCO (2017), and compare its performance with other sate-of-the-art models. Our proposed model has the BLEU-1 score of 74.0.

More details can be found in the following file:
> Image_Captioning_with_GRU_based_Attention.pdf

<img src="/images/Example_result.png" width="600" height="600"/>

## Our Contributions and Proposed Model
Our proposed attention model for image captioning, consisting of CNN, Attention Module, and LSTM. The attention module has two main components. A MLP to compute the attention weights and a attention GRU module which aims to provide a contextual representation that allows logical reasoning over interesting regions.

<img src="/images/Proposed_model.png" width="900" height="500"/>

## Attention GRU Module
Inspired by [C. Xiong, S. Merity, and R. Socher. Dynamic memory networks for visual and textual question answering. In International conference on machine learning, pages 2397–2406, 2016.], we want the attention mechanism to take into account both position and ordering of the input regions. An RNN would be advantageous in this situation except they cannot make use of the attention wights.

In the figure below, you can ifnd the difference between (a) the traditional GRU, and (b) the proposed attention-based GRU model in this work:

<img src="/images/GRUs.png" width="600" height="250"/>

## Experiments

### Dataset
We performed experiments on MS COCO dataset, contains complex day-to-day scenes of common objects in their natural context. The dataset contains 82,783 training images, 40,504 validation images, and 40,775 test images. Each image is annotated with 5 sentences using Amazon Mechanical Turk. Since, there is an ongoing competition on this dataset, annotation for test dataset is not available. To train our model, we have used both training and validation sets. To
test the proposed model, we have hold out 5000 samples of the validation set. The same split is used for all the experiments. We did not use the test set for evaluation since there is a limited number of submissions available per day.

### Metric
We use METEOR and BLEU as evaluation metrics, which are popular in the machine translation literature and used in recent image caption generation papers. The BLEU score is based on n-gram precision of the generated caption with respect to the references. The METEOR is based on the harmonic mean of uni-gram precision and recall, and produces a good correlation with human judgment.

### Implementation Details
The models are implemented with Tensorflow and are trained using the RMSprop optimizer for 100 epochs with batch size 40 and learning rate 0:0001. For fairness, we re-implemented two baselines models in and trained them and our proposed model with the same setting. The CNN used in all the models is ResNet150 and we used the pre-trained model on ImageNet for initializing the weights.

## Results
BLEU-1,2,3,4/METEOR metrics compared to other methods on MS COCO dataset. Models with * are trained on both train set and validation set.

<img src="/images/scores.png" width="600" height="200"/>

## Conclusion
We have presented a new attention mechanism for image caption generation by introducing ATTN GRU (a modified version of traditional GRU). Unlike soft-attention mechanism, our attention model preserves the spatial information as well as the order of the regions in the image. Experimental results on MS COCO dataset shows the effectiveness of our model in image captioning task.



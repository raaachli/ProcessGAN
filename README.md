# ProcessGAN
Synthetic process data generator using GAN-based Transformer.

Implementation of the paper "Generating Privacy-Preserving Process Data with Deep Generative Models".

# Requirements
* Python 3
* Pytorch
* Matplotlib

# Training
Please run the run.py to start training the models.

You can choose to train the baseline models or the ProcessGAN variants inside the run.py file.

# Networks Details
* [Models](https://github.com/raaachli/ProcessGAN/tree/main/models) including basic models of the Transformer encoder, the RNN model, and the LSTM model.
* [Networks](https://github.com/raaachli/ProcessGAN/tree/main/nets) including the implementation of MLE-based training, and GAN-based training.

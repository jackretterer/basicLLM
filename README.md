Lyrics Generation LLM
This repository contains a PyTorch implementation of a Transformer-based model for generating lyrics. The model is trained on a dataset of Taylor Swift lyrics and "learns" (not all that well) to predict the next word in a sequence based on the provided context.

# Overview
The code in this repository demonstrates the following:

Preprocessing of lyrics data using the data_preprocessing.py script.
Implementation of a custom LyricsDataset class for loading and processing the lyrics data.
Definition of a SimpleTransformer model architecture using PyTorch.
Training of the Transformer model on the preprocessed lyrics dataset.
Evaluation of the trained model by generating lyrics based on input sequences.

# Dataset Preparation
To prepare the dataset for training, follow these steps:

1. Place your lyrics data in a text file named all_tswift_lyrics.txt, with each line representing a separate lyric.
2. Run the data_preprocessing.py script to preprocess the lyrics data:
```python data_preprocessing.py```
This script will remove empty lines and lowercase the text, saving the preprocessed lyrics to all_tswift_lyrics_cleaned.txt.
3. The preprocessed lyrics file will be used as input to the LyricsDataset class during training.

# Model Architecture
The SimpleTransformer model is a simplified implementation of the Transformer architecture for sequence prediction. It consists of the following components:

- An embedding layer to convert input word indices to dense vectors.
- Multiple Transformer encoder layers to capture the contextual information in the input sequence.
- A linear layer to map the encoded representations to the output vocabulary space.

The model is trained using the cross-entropy loss and optimized using the Adam optimizer.

# Training
To train the model, run the provided code in a Python environment with PyTorch installed. The code will:

1. Load the preprocessed lyrics dataset using the LyricsDataset class.
2. Instantiate the SimpleTransformer model with the specified hyperparameters.
3. Train the model for a specified number of epochs, updating the model's parameters based on the calculated gradients.
4. Print the loss value at the end of each epoch to monitor the training progress.

# Evaluation
After training, the code demonstrates how to use the trained model for generating lyrics. It provides two examples:

1. Generating the next word based on the input sequence "So shame on ".
2. Generating multiple predicted words based on the input sequence "I knew you were".

The generated lyrics, along with their corresponding probabilities, are printed to the console.

# Proposed Improvements
This code serves as a starting point for lyrics generation using a Transformer-based model. Here are some proposed improvements to enhance the model's performance and functionality:

1. Increase the dataset. Add lyrics of other artists so that the model learns English and songs better. Also learns the concept of sentence formation
2. Adjust the hyperparameters. More layers, more hidden dimensions, more heads, etc.
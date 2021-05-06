# Description
This app is designed to predict the location of the three complementarity determining regions (CDR) regions of an antibody. In particular, we consider an antibody that binds to the influenza virus receptor binding domain. Given an antibody protein, the app predicts the center of each CDR region along the protein as a scalar value from 0 to 1. This app uses the dataset provided in the [Zindi UmojaHack DeepChain Antibody Classification Challenge](https://zindi.africa/hackathons/umojahack-africa-2021-1-instadeep-challenge-advanced). Please see the competition page for more details.

![alt text](https://upload.wikimedia.org/wikipedia/commons/3/3c/Complementarity_determining_regions.PNG)

These CDR regions are useful when determining the binding energy of an antibody protein to an influenza protein. The goal of this app is to help the user determine the location of these regions. This model could also be used in combination with other models, such as a binding energy predictor, by extracting the CDR regions directly as features. These features could then be embedded with [Bio-Transformers](https://pypi.org/project/bio-transformers/) such as ProtBert, allowing for a very simple and robust model to predict the final binding energy trained on CDR region embeddings.

## The Data

The dataset consists of antibody 40184 proteins, each consisting of 221 amino acids. Three complementarity-determining regions (CDR) are also given. We predict the location of these CDR regions. Each amino acid sequence is labeled with a binding energy score, although this remains unused in our App. We hope that by predicting the location of the CDR regions, we could generate better estimates of the antibody binding energy. The dataset is available for download on the [Zindi competition page](https://zindi.africa/hackathons/umojahack-africa-2021-1-instadeep-challenge-advanced/data).

Each antibody protein has 221 amino acids, but we want our model to be sequence length agnostic. We therefore preprocess the sequence during training by cropping various lengths of the start and end of the amino acid sequence. We also mask random portions of the sequence to improve generalisation of the model.

## The Model

![alt text](https://raw.githubusercontent.com/KevinEloff/deep-chain-apps/feature/pytorch_app/example/pytorch_app/docs/model2.png)

We do not use any pre-trained embeddings for our scorer. Rather, we embed each individual amino acid. These embeddings are passed through a bi-directional LSTM block and a 1D convolution block. The output of both blocks is passed through a feed forward network to arrive at the final prediction for the three CDR regions.

# Tags

## Libraries

- PyTorch==1.5.0

## Author

- Kevin Eloff  
Stellenbosch University  
20801769@sun.ac.za


## Tasks

- cdr prediction
- LSTM
- convolution
- regressor

## Embeddings

- none

## Datasets

- [Zindi UmojaHack DeepChain Antibody Classification Challenge dataset](https://zindi.africa/hackathons/umojahack-africa-2021-1-instadeep-challenge-advanced/data)

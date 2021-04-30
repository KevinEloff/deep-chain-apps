# Description
This app is designed to estimate how effectively an antibody binds to the influenza virus receptor binding domain. Given an antibody protein, the app predicts the binding energy to the Influenza HA protein as seen below. This app uses a simplified version of the model used in the [Zindi UmojaHack DeepChain Antibody Classification Challenge](https://zindi.africa/hackathons/umojahack-africa-2021-1-instadeep-challenge-advanced). Please see the competition page for more details.

<p align="center" >
  <img src="https://raw.githubusercontent.com/KevinEloff/deep-chain-apps/feature/pytorch_app/example/pytorch_app/docs/influenza-infection2.png" style="width:50%"/>
</p>

## The Data

The dataset consists of 40184 proteins, each consisting of 221 amino acids. Three complementarity-determining regions (CDR) are also given. We ignore these CDR regions as we would like to train a model that is sequence length agnostic. Each amino acid sequence is labeled with a binding energy score. The dataset is available for download on the [Zindi competition page](https://zindi.africa/hackathons/umojahack-africa-2021-1-instadeep-challenge-advanced/data).

## The Model

<p align="center" >
  <img src="https://raw.githubusercontent.com/KevinEloff/deep-chain-apps/feature/pytorch_app/example/pytorch_app/docs/model.png" style="width:100%"/>
</p>

We do not use any pre-trained embeddings for our scorer. Rather, we embed each individual amino acid, along with amino acid bigrams and offset bigrams. These embeddings are passed through a bi-directional LSTM block and a 1D convolution block. The output of both blocks is passed through a feed forward network to arrive at the final binding energy prediction.

# Tags

## Libraries

- PyTorch==1.5.0

## Author

- Matthew Baas
- Kevin Eloff

## Tasks

- binding energy
- LSTM
- convolution
- regressor

## embeddings

- none

## datasets

- [Zindi UmojaHack DeepChain Antibody Classification Challenge dataset](https://zindi.africa/hackathons/umojahack-africa-2021-1-instadeep-challenge-advanced/data)
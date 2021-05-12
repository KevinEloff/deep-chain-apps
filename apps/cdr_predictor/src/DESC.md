# Description
This app is designed to predict the location of the three complementarity determining regions (CDR) regions of an antibody. In particular, we consider an antibody that binds to the influenza virus receptor binding domain. Given an antibody protein, the app predicts the center of each CDR region along the protein as a scalar value from 0 to 1. This app uses the dataset provided in the [Zindi UmojaHack DeepChain Antibody Classification Challenge](https://zindi.africa/hackathons/umojahack-africa-2021-1-instadeep-challenge-advanced). Please see the competition page for more details.

![alt text](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Complementarity_determining_regions.PNG/640px-Complementarity_determining_regions.PNG)

The CDR regions are useful when determining the binding energy of an antibody protein to an influenza protein. The goal of this app is to help the user determine the location of these regions. This model could also be used in combination with other models, such as a binding energy predictor, by extracting the CDR regions directly as features. These features could then be embedded with [bio-transformers](https://pypi.org/project/bio-transformers/) such as ProtBert, allowing for a very simple and robust model to predict the final binding energy trained on CDR region embeddings.


## The Data

The dataset consists of antibody 40184 proteins, each consisting of 221 amino acids. Three complementarity-determining regions (CDR) are also given. We predict the location of these CDR regions. Each amino acid sequence is labeled with a binding energy score, although this remains unused in our App. We hope that by predicting the location of the CDR regions, we could generate better estimates of the antibody binding energy. The dataset is available for download on the [Zindi competition page](https://zindi.africa/hackathons/umojahack-africa-2021-1-instadeep-challenge-advanced/data), or from the [bio-datasets](https://pypi.org/project/bio-datasets/) package as "antibodybinding".

Each antibody protein has 221 amino acids, but we want our model to be sequence length agnostic. We therefore preprocess the sequence during training by cropping various lengths of the start and end of the amino acid sequence. We also mask random portions of the sequence to improve generalisation of the model.

## The Model

![alt text](https://raw.githubusercontent.com/KevinEloff/deep-chain-apps/main/apps/cdr_predictor/docs/model.png)

We do not use any pre-trained embeddings for our scorer. Rather, we embed each individual amino acid. These embeddings are passed through a bi-directional LSTM block and a 1D convolution block. The output of both blocks is passed through a feed forward network to arrive at the final prediction for the three CDR regions.

## Results
We measure the accuracy of the model in terms of Root Mean Square Error (RMSE). The error is calculated between the predicted integer index location of each CDR region and the actual location for each of the three regions. The overall RMSE for the scalar prediction during training is 0.000293.

**RMSE**:
- CDR1: 0.5376026
- CDR2: 0.5678551
- CDR3: 1.0083260

## Example Usage 
The app itself takes in a sequence of antibody proteins as input and returns three float scalars for each protein. For example, if we input a sequence:

```python
sequences = [
  "QLKESGPGLVAPSQSLHITCTVSGFNLASNGVHWVRQPPGSGLEWLGVIWAGGNTNYNSALMSNVSIS
  KDNSKSQVFLKMKSLQTDDTAMYYCARDFYDTDVGYYAMDYWGQGTSVVVSSAKTTPPSVYPLAPGSAA
  FTNSMVTLPADPADPADPADKGYFPEPVTVTWNSGSLSSGVHTFPAVLQSDLYTLSSSVTVPSSTWPSE",
  "CTVSGFLLCSNGVHWVRQPPGKGLEWDGVIWAGGLTNYNSALMSRVSISKDNSKSQVFLKMKSLQTDD
  TALVYCARDFLDYDVGYYAMDYWGQGTSVTVSSAKTTPPYVDPLAPGSAAQTNSMVTLGCLVKGYFPEP
  VTVTWNSGSLSSGVHTFPAVLQSDLYTLSSSVTVPSSTWPSETVTCNVAHPASST",
]
app = App("cuda:0")
scores = app.compute_scores(sequences)
```
The app will return a set of three scalars for each input protein:

```python
> scores
[
  {
    'CDR1_position': 0.15440723299980164, 
    'CDR2_position': 0.32853004336357117, 
    'CDR3_position': 0.5693634152412415
  }, 
  {
    'CDR1_position': 0.11785992980003357, 
    'CDR2_position': 0.20338580012321472, 
    'CDR3_position': 0.4263269305229187
  }
]
```

These scalars each represent how far along the amino acid sequence the center of the CDR region is. To convert these back into a sequence index, we multiply by the original sequence length and round the result.

See the source code [here](https://github.com/KevinEloff/deep-chain-apps/tree/main/apps/cdr_predictor).
# Author

**Kevin Michael Eloff**  
Research Engineer Intern @ InstaDeep   
MEng @ Stellenbosch University  
20801769@sun.ac.za  

# Tags

## Libraries

- PyTorch==1.5.0

## Tasks

- cdr prediction
- regressor
- LSTM
- CNN

## Embeddings

- none

## Datasets

- antibodybinding dataset
- [Zindi UmojaHack DeepChain Antibody Classification Challenge dataset](https://zindi.africa/hackathons/umojahack-africa-2021-1-instadeep-challenge-advanced/data)

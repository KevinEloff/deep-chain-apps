# Description
This app is designed to estimate how effectively an antibody binds to the influenza virus receptor binding domain. Given an antibody protein, the app predicts the binding energy to the Influenza Hemagglutinin (HA) protein as seen below. This app uses a simplified version of the model used in the [Zindi UmojaHack DeepChain Antibody Classification Challenge](https://zindi.africa/hackathons/umojahack-africa-2021-1-instadeep-challenge-advanced). Please see the competition page for more details.

![alt text](https://raw.githubusercontent.com/KevinEloff/deep-chain-apps/main/apps/pytorch_app/docs/influenza-infection2.png)

The CDR regions are useful when determining the binding energy of an antibody protein to an influenza protein. The goal of this app is to predict the location of these CDR regions and then use them with ProtBert embeddings to create a simple, yet accurate, model that predicts the binding energy of antibody proteins.


## The Data

The dataset consists of antibody 40184 proteins, each consisting of 221 amino acids. Three complementarity-determining regions (CDR) are also given along with their ProtBert embeddings. Each amino acid sequence is also labeled with a binding energy score. For the purposes of our app, we will only use the embeddings of the three CDR regions (3x1024) and the binding energy labels. The dataset is available for download on the [Zindi competition page](https://zindi.africa/hackathons/umojahack-africa-2021-1-instadeep-challenge-advanced/data), or from the [bio-datasets](https://pypi.org/project/bio-datasets/) package as "antibodybinding".



## The Model

![alt text](https://raw.githubusercontent.com/KevinEloff/deep-chain-apps/main/apps/influenza_binding/docs/cdr_binding_prediction_720.png)

Normally, the CDR locations are unknown for a varying length antibody sequence. We therefore use our CDRPredictorApp to predict the location of the three CDR regions, which we then embed using ProtBert from the [bio-transformers](https://pypi.org/project/bio-transformers/) package. These embeddings are finally used in a simple three-layer MLP to predict the binding energy of the antibody. The architecture of the binding energy MLP is given below:

```Python
Model(
  (linear_cdr): Sequential(
    (0): Linear(in_features=3072, out_features=512, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.1, inplace=False)
    (3): Linear(in_features=512, out_features=256, bias=True)
    (4): ReLU()
    (5): Dropout(p=0.1, inplace=False)
  )
  (out_linear): Linear(in_features=256, out_features=1, bias=True)
)
```

To extract the CDR region embeddings, we first process the entire amino acid sequence through ProtBert which returns a full list of embeddings. We then index these embeddings at each of the three predicted CDR regions. The mean embedding of each location then becomes the embedding for that CDR region. Please see the CDRPredictorApp on the [DeepChain Hub](https://app.deepchain.bio/hub/apps) or [GitHub](https://github.com/KevinEloff/deep-chain-apps/tree/main/apps/cdr_predictor) for more detail on predicting the locations of the three CDR regions.

## Results
We measure the accuracy of the model in terms of Root Mean Square Error (RMSE). The error is calculated between the predicted binding energy and the actual binding energy on our validation and test sets.

**RMSE**:
- valid: 2.587984 
- test: 2.622088

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
The app will return a set of binding energies for each input protein:

```python
> scores
[
  {
    'binding_energy': -11.36922,
  }, 
  {
    'binding_energy': -14.23689,
  }
]
```

These values represent the predicted binding energies of the input antibody proteins to the influenza HA protein.

See the source code [here](https://github.com/KevinEloff/deep-chain-apps/tree/main/apps/influenza_binding).
# Author

**Kevin Eloff**  
Research Engineer Intern @ InstaDeep   
MEng @ Stellenbosch University  
20801769@sun.ac.za  

# Tags

## Libraries

- PyTorch==1.5.0

## Tasks

- binding energy
- regressor
- MLP

## Embeddings

- ProtBert (mean)

## Datasets

- antibodybinding dataset
- [Zindi UmojaHack DeepChain Antibody Classification Challenge dataset](https://zindi.africa/hackathons/umojahack-africa-2021-1-instadeep-challenge-advanced/data)

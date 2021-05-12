# Full Sequence Influenza Binding App
This app is designed to estimate how effectively an antibody binds to the influenza virus receptor binding domain. Given an antibody protein, the app predicts the binding energy to the Influenza HA protein as seen below. This app uses the full antibody sequence to estimate the binding energy. The app uses a simplified version of the model used in the [Zindi UmojaHack DeepChain Antibody Classification Challenge](https://zindi.africa/hackathons/umojahack-africa-2021-1-instadeep-challenge-advanced). Please see the competition page for more details.

![alt text](https://raw.githubusercontent.com/KevinEloff/deep-chain-apps/main/apps/pytorch_app/docs/influenza-infection2.png)

## The Data

The dataset consists of 40184 proteins, each consisting of 221 amino acids. Three complementarity-determining regions (CDR) are also given. We ignore these CDR regions as we would like to train a model that is sequence length agnostic. Each amino acid sequence is labeled with a binding energy score. The dataset is available for download on the [Zindi competition page](https://zindi.africa/hackathons/umojahack-africa-2021-1-instadeep-challenge-advanced/data).

## The Model

![alt text](https://raw.githubusercontent.com/KevinEloff/deep-chain-apps/feature/pytorch_app/example/pytorch_app/docs/model.png)

We do not use any pre-trained embeddings for our scorer. Rather, we embed each individual amino acid, along with amino acid bigrams and offset bigrams. These embeddings are passed through a bi-directional LSTM block and a 1D convolution block. The output of both blocks is passed through a feed forward network to arrive at the final binding energy prediction.

## Results
We measure the accuracy of the model in terms of Root Mean Square Error (RMSE). The error is calculated between the predicted binding energy and the actual binding energy on our validation and test sets.

**RMSE**:
- test: 2.64095712


## Example Usage 
The app itself takes in a sequence of antibody proteins as input and returns scalar float score for each protein. For example, if we input a sequence:

```python
sequences = [
  "QVQLKEHGPGLVNPSQSLSVTCSVSGFLLISNGVHWVRQPPGKGLEWLGVIWAGGNTNYNIALMSRVS
  ISKDNSKSQVFLKCKSLQTDDTAMYCCARDFYDYDNFTYAMAYWGQGTSVTVSSAKTTPPSVYPLAPGS
  AAQTNSMVTLGCLVKGYFPEPVTVTWNSGSLSSGVHTFPAVLQSDLYTLSSSVTVPSSTWPSETVTCNV
  AHPASSTKVDKKIVP",
  "YGPGLVAPSQSLSITCTVSGFLLISNGVHWVRQPPGKGLEWLGVIWAGGMTAYNSATMSRVSISKDNS
  KSQVFLKMKSLQTDDTAMYYCARDFYCYDVFYYAMDYWGQGTSVTVSSAYTTPPSVYPLAPGSAAQTNS
  MVTLGCLVKGYFPEPVTVTWNSGSLSSGVHTFPAVLQSDLYTLSSSVTVPSSTWPSETVTCNVAHPASS
  TKV",
]
app = App("cuda:0")
scores = app.compute_scores(sequences)
```
The app will return a set of binding energies for each input protein:

```python
> scores
[
  {
    'binding_energy': -9.6949663,
  }, 
  {
    'binding_energy': -10.2529935,
  }
]
```

These values represent the predicted binding energies of the input antibody proteins to the influenza HA protein.

See the source code [here](https://github.com/KevinEloff/deep-chain-apps/tree/main/apps/pytorch_app).
# Author

**Kevin Eloff**  
Research Engineer Intern @ InstaDeep   
MEng @ Stellenbosch University  
20801769@sun.ac.za  

**Matthew Baas**  
MEng @ Stellenbosch University  
20786379@sun.ac.za  

# Tags

## Libraries

- PyTorch==1.5.0

## Tasks

- binding energy
- LSTM
- CNN
- regressor

## Embeddings

- none

## Datasets

- antibodybinding dataset
- [Zindi UmojaHack DeepChain Antibody Classification Challenge dataset](https://zindi.africa/hackathons/umojahack-africa-2021-1-instadeep-challenge-advanced/data)

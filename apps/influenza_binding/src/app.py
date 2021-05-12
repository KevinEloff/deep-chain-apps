"""
PyTorch implementaton of binding energy scorer
The model scores the binding energy per position in the receptor binding domain of proteins.
By Kevin Eloff
"""

from typing import Dict, List, Optional, Tuple

from biotransformers import BioTransformers
from deepchain.components import DeepChainApp
from torch import load, nn
import torch

import numpy as np

Score = Dict[str, float]
ScoreList = List[Score]

class App(DeepChainApp):
    """CDR Region Predictor App:

    - Predicts the location of the 3 CDR regions given an antibody protein.
    - Embeds these regions and then calculates binding energy with simple MLP
    """

    def __init__(self, device: str = "cuda:0"):
        self._device = device
        self.transformer = BioTransformers(backend="protbert", device=device)

        # Make sure to put your checkpoint file in your_app/checkpoint folder
        self._checkpoint_filename_cdr: Optional[str] = 'model_cdr.pt'
        self._checkpoint_filename_eng: Optional[str] = 'model_eng.pt'

        # load_model - load for pytorch model
        self.cdr_model = CDRModel().to(device)
        if self._checkpoint_filename_cdr is not None:
            self._checkpoint_filename = self._checkpoint_filename_cdr
            self.cdr_model.load_state_dict(load(self.get_checkpoint_path(__file__)))
        if self._checkpoint_filename_eng is not None:
            self._checkpoint_filename = self._checkpoint_filename_eng
            self.energy_model = load(self.get_checkpoint_path(__file__))
        
        self.i2s = [
            'PAD', 'EOS', 'P', 'V', 'I', 'K', 'N', 'B', 'F', 'Y', 'E', 'W', 'R', 
            'D', 'X', 'S', 'C', 'U', 'Q', 'A', 'M', 'H', 'L', 'G', 'T'
        ]
        self.s2i = { k:v for k , v in zip(self.i2s, range(0,len(self.i2s)))}

    @staticmethod
    def score_names() -> List[str]:
        """
        Return a list of app score names
        """
        return ["binding_energy"]

    def compute_scores(self, sequences: List[str]) -> ScoreList:
        """Return a list of all proteins scores
        """

        cdr_locations = []
        for seq in sequences:
            # Full sequence converted to integer IDs
            full_seq = self._preprocess_seq(['EOS',] + list(seq) + ['EOS',])

            # Send full sequence and sequence length to model
            inp = (
                torch.LongTensor([full_seq]).to(self._device),
                torch.tensor([len(full_seq[0])], dtype=torch.float32).to(self._device),
            )
            # Predict CDR positions
            with torch.no_grad(): 
                pred = self.cdr_model(inp)[0]
                
            # Format as scores
            cdr_locations.append([
                np.clip(pred[i].cpu().item()/100, 0, 1)*len(full_seq[0])
                for i in range(3)
            ])

        # Extract CDR regions from ProtBert embedding
        cdr1_emb = []
        cdr2_emb = []
        cdr3_emb = []

        for i in range(len(sequences)):
            emb = self.transformer.compute_embeddings(
                [sequences[i]],
                pool_mode=["full"],
            )["full"][0]

            loc = np.floor(cdr_locations[i]).astype(int)
            
            cdr1_emb.append(emb[loc[0]-2:loc[0]+3].mean(axis=0))
            cdr2_emb.append(emb[loc[1]-3:loc[1]+4].mean(axis=0))
            cdr3_emb.append(emb[loc[2]-4:loc[2]+4].mean(axis=0))
        
        inp = torch.cat((
            torch.tensor(cdr1_emb, device=self._device), 
            torch.tensor(cdr2_emb, device=self._device), 
            torch.tensor(cdr3_emb, device=self._device)
        ), dim=-1)

        with torch.no_grad():
            pred = self.energy_model(inp)
        
        return [{self.score_names()[0]: p.item()} for p in pred]
    
    def _preprocess_seq(self, sequence: str) -> np.ndarray:
        full_cdr = list(sequence)
        full_cdr = [self.s2i[c] for c in full_cdr]
        
        inference_item = np.array([full_cdr])
        return inference_item

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear_cdr = nn.Sequential(
            nn.Linear(1024*3, 512), 
            nn.ReLU(),
            nn.Linear(512, 256), 
            nn.ReLU(), 
        )
        
        self.out_linear = nn.Linear(256, 1)

    def forward(self, x):
        x = self.linear_cdr(x)
        out = self.out_linear(x)
        
        return out

class CDRModel(nn.Module):
    def __init__(self):
        super(CDRModel, self).__init__()
        self.embed_plain = nn.Embedding(25, 128)
        
        self.BiLSTM = nn.LSTM(128, 256, num_layers=3, batch_first=True, bidirectional=True)
        
        self.conv_full = nn.Sequential(nn.Conv1d(128, 128, 3, stride=1),
                                       nn.SELU(), 
                                       nn.Conv1d(128, 128, 3, stride=1),
                                       nn.SELU(), 
                                       nn.Conv1d(128, 256, 3, stride=1),
                                       nn.SELU())
        
        self.input_linear = nn.Sequential(nn.Linear(1, 32), nn.SELU())
        
        self.hidden_linear = nn.Sequential(nn.Linear(256*2+256+32, 64), nn.SELU())
        self.out_linear = nn.Linear(64, 3)

    def forward(self, x):
        embed = self.embed_plain(x[0][:,0])
                                           
        xn, (hn, cn) = self.BiLSTM(embed)

        conv = torch.max(self.conv_full(embed.transpose(-1,-2)), dim=-1)[0]
        lin = self.input_linear(x[1].unsqueeze(1))

        idx = torch.stack((torch.arange(xn.shape[0], device=xn.device),x[1].long()-1),dim=1).long()
        conc = torch.cat((xn[idx[:,0],idx[:,1]], conv, lin), dim=1)
        out = self.out_linear(self.hidden_linear(conc))
        
        return out

if __name__ == "__main__":

    sequences = [
        "QVQLKEHGPGLVNPSQSLSVTCSVSGFLLISNGVHWVRQPPGKGLEWLGVIWAGGNTNYNIALMSRVSISKDNSKSQVFLKCKSLQTDDTAMYCCARDFYDYDNFTYAMAYWGQGTSVTVSSAKTTPPSVYPLAPGSAAQTNSMVTLGCLVKGYFPEPVTVTWNSGSLSSGVHTFPAVLQSDLYTLSSSVTVPSSTWPSETVTCNVAHPASSTKVDKKIVP",
        "YGPGLVAPSQSLSITCTVSGFLLISNGVHWVRQPPGKGLEWLGVIWAGGMTAYNSATMSRVSISKDNSKSQVFLKMKSLQTDDTAMYYCARDFYCYDVFYYAMDYWGQGTSVTVSSAYTTPPSVYPLAPGSAAQTNSMVTLGCLVKGYFPEPVTVTWNSGSLSSGVHTFPAVLQSDLYTLSSSVTVPSSTWPSETVTCNVAHPASSTKV",
    ]
    app = App("cuda:0")
    scores = app.compute_scores(sequences)
    print(scores)

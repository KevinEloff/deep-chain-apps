""" CDR Region Predictor App
Author: Kevin Eloff
"""

from typing import Dict, List, Optional

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
    - This is useful for
    """

    def __init__(self, device: str = "cuda:0"):
        self._device = device

        # Make sure to put your checkpoint file in your_app/checkpoint folder
        self._checkpoint_filename: Optional[str] = 'model.pt'

        # load_model - load for pytorch model
        self.model = Model().to(device)
        if self._checkpoint_filename is not None:
            self.model.load_state_dict(load(self.get_checkpoint_path(__file__)))
        
        self.i2s = [
            'PAD', 'EOS', 'P', 'V', 'I', 'K', 'N', 'B', 'F', 'Y', 'E', 'W', 'R', 
            'D', 'X', 'S', 'C', 'U', 'Q', 'A', 'M', 'H', 'L', 'G', 'T'
        ]
        self.s2i = { k:v for k , v in zip(self.i2s, range(0,len(self.i2s)))}

    @staticmethod
    def score_names() -> List[str]:
        """App Score Names. 
        CDR positions represented as a scalar from 0 to 1
        indicating how far along the protein sequence the
        predicted center of the CDR region is.
        """
        
        return [
            "CDR1_position",
            "CDR2_position",
            "CDR3_position",
            ]

    def compute_scores(self, sequences: List[str]) -> ScoreList:
        """Return a list of all proteins scores
        """
        scores = []

        for seq in sequences:
            # Full sequence converted to integer IDs
            full_seq = self._preprocess_seq(['EOS',] + list(seq) + ['EOS',])

            # Send full sequence and sequence length to model
            inp = (
                torch.LongTensor([full_seq]).to(self._device),
                torch.tensor([len(full_seq[0])], dtype=torch.float32).to(self._device),
            )
            # Predict CDR positions
            pred = self.model(inp)[0]
            
            # Format as scores
            score_names = self.score_names()
            scores.append({
                score_names[i]: np.clip(pred[i].item()/100, 0, 1)
                for i in range(3)
            })

        return scores
    
    def _preprocess_seq(self, sequence: str) -> np.ndarray:
        full_cdr = list(sequence)
        full_cdr = [self.s2i[c] for c in full_cdr]
        
        inference_item = np.array([full_cdr])
        return inference_item

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
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
        "CTVSGFLLCSNGVHWVRQPPGKGLEWDGVIWAGGLTNYNSALMSRVSISKDNSKSQVFLKMKSLQTDDTALVYCARDFLDYDVGYYAMDYWGQGTSVTVSSAKTTPPYVDPLAPGSAAQTNSMVTLGCLVKGYFPEPVTVTWNSGSLSSGVHTFPAVLQSDLYTLSSSVTVPSSTWPSETVTCNVAHPASST",
    ]
    app = App("cuda:0")
    scores = app.compute_scores(sequences)
    print(scores)

from typing import Dict

import torch
from torch.nn import Embedding

import torch.nn as nn
import torch.nn.functional as F


class SeqClassifier(torch.nn.Module):
    def __init__(
            self,
            embeddings: torch.tensor,
            hidden_size: int,
            num_layers: int,
            dropout: float,
            bidirectional: bool,
            num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class

        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bidirectional=self.bidirectional,
                            batch_first=self.bidirectional,
                            dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_size, self.num_class)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        x = self.embed(batch)
        x, (h_n, c_n) = self.lstm(x)
        output = h_n[-1, :, :]
        output = self.fc(output)
        output = F.log_softmax(output, dim=-1)
        return output
        raise NotImplementedError

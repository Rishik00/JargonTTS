## Taken inspo from this artice: https://medium.com/@tttzof351/build-text-to-speech-from-scratch-part-1-ba8b313a504f

import torch
import argparse
import torch.nn as nn

# local imports
from convnorm import ConvNormBlock
from test_yaml import load_yaml

class EncoderPreNet(nn.Module):
    
    def __init__(self):
        super(EncoderPreNet, self).__init__()

        self.embedding = nn.Embedding(100, 512)
        self.initial_proj = nn.Linear(512, 512)

        self.convblock = ConvNormBlock(
            embed_dim=512,
            kernel_size=3,
            stride=1,
            dropout=0.5
        )

        self.final_proj = nn.Linear(512, 512)

    def forward(self, text):
        x=self.embedding(text) # (32, 100, 512)
        x=self.initial_proj(x) # ()

        x=x.transpose(2,1) # (32, 512, 100)

        x = ConvNormBlock(x)
        x = x.transpose(1,2)

        x = self.final_proj(x)
        return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and print YAML file contents.")
    parser.add_argument("config_file_name", help="Path to the YAML configuration file")

    args = parser.parse_args()
    batch_size = load_yaml(args.config_file_name)

    print("Batch Size:", batch_size)
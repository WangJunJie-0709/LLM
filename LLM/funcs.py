import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import requests

if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.is_available())

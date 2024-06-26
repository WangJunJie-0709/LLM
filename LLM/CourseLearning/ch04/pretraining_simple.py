import argparse
import os
from pathlib import Path
import time
import tiktoken
import torch
from ch04 import create_dataloader_v1, GPTModel, generate_and_print_sample, calc_loss_batch, evaluate_model, plot_losses


def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text_data = file.read()
    return text_data






if __name__ == '__main__':

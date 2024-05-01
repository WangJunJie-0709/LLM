import torch

hyperparameters = {
    'batch_size': 4,
    'context_length': 64,
    'd_model': 64,
    'num_heads': 8,
    'num_blocks': 8,
    'learning_rate': 2e-4,
    'dropout': 0.1,
    'max_iters': 20000,
    'eval_interval': 50,
    'eval_iters': 20,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
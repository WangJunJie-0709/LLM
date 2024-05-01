import torch

hyperparameters = {
    'batch_size': 4,
    'context_length': 16,
    'd_model': 64,
    'num_heads': 4,
    'num_blocks': 8,
    'learning_rate': 1e-3,
    'dropout': 0.1,
    'max_iters': 50000,
    'eval_interval': 50,
    'eval_iters': 20,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
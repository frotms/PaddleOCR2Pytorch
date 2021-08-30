import os, sys
import numpy as np
import torch

if __name__ == '__main__':
    print('begin..')
    model_path = 'test.pt'

    np.random.seed(666)
    inputs = np.random.randn(1, 3, 32, 320).astype(np.float32)
    inp = torch.from_numpy(inputs)

    m = torch.jit.load(model_path)
    with torch.no_grad():
        out = m(inp).cpu().numpy()
    print('out:', np.sum(out), np.mean(out), np.max(out), np.min(out))

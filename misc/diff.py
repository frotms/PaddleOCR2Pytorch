import numpy as np

def print_cmp(inp, name=None):
    print('{}: shape-{}, sum: {}, mean: {}, max: {}, min: {}'.format(name, inp.shape,
                                                                     np.sum(inp), np.mean(inp),
                                                                     np.max(inp), np.min(inp)))

pp = np.load('pptmp.npy')
pt = np.load('pttmp.npy')



diff = pp-pt
print('diff:')
print(diff)
print('======')
print_cmp(pp, name='pp')
print_cmp(pt, name='pt')
print_cmp(diff, name='diff')

import sys, numpy as np, torch
sd = torch.load(sys.argv[1], map_location='cpu')
parts = []
for n in ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'out']:
    parts += [sd[f'{n}.weight'].numpy().ravel(), sd[f'{n}.bias'].numpy().ravel()]
a = np.concatenate(parts).astype('<f4'); a.tofile(sys.argv[2])
print('parametri', a.size, 'byte', a.nbytes)

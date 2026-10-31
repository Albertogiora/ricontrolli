# Riaddestra il lettore di cifre partendo direttamente da model.bin, con JAX su CPU (niente PyTorch).
# Stessa rete di seqtrain.py; mescola le correzioni reali (aumentate) con sequenze sintetiche MNIST.
# uso: pip install jax optax opencv-python-headless numpy
#      python3 finetune_jax.py ../model.bin nuovo.bin mnist.npz ricontrolli-correzioni-*.json [celle-foto.json]
#      HOLD=<src> tiene fuori i campioni con quella sorgente per misurare su dati mai visti; STEPS, LR regolabili.
import os, sys, json, base64, time, numpy as np, jax, jax.numpy as jnp, optax, cv2
src, dst, mnist, files = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
import types
for n in ('torch', 'torch.nn', 'torch.nn.functional'): sys.modules[n] = types.ModuleType(n)   # seqtrain importa torch solo per la rete: qui serve il generatore
sys.modules['torch'].nn = sys.modules['torch.nn']; sys.modules['torch.nn'].Module = object; sys.modules['torch.nn'].functional = sys.modules['torch.nn.functional']
sys.argv = ['x', mnist]; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import seqtrain as st
LAYERS = [('c1', 16, 1, 3, 3, 1, 1), ('c2', 32, 16, 3, 3, 1, 1), ('c3', 48, 32, 3, 3, 1, 1), ('c4', 64, 48, 3, 3, 1, 1),
          ('c5', 96, 64, 4, 3, 0, 1), ('c6', 96, 96, 1, 5, 0, 2), ('out', 12, 96, 1, 1, 0, 0)]
CH = '0123456789-'
def load(fn):
    f = np.fromfile(fn, np.float32); o = 0; p = {}
    for n, oc, ic, kh, kw, _, _ in LAYERS:
        p[n] = {'w': jnp.array(f[o:o + oc * ic * kh * kw].reshape(oc, ic, kh, kw)), 'b': jnp.array(f[o + oc * ic * kh * kw:o + oc * ic * kh * kw + oc])}; o += oc * ic * kh * kw + oc
    assert o == len(f); return p
def save(p, fn):
    np.concatenate([np.concatenate([np.asarray(p[n]['w']).ravel(), np.asarray(p[n]['b']).ravel()]) for n, *_ in LAYERS]).astype(np.float32).tofile(fn)
PADS = {n: ((ph, ph), (pw, pw)) for n, _, _, _, _, ph, pw in LAYERS}
def conv(x, l, n): return jax.lax.conv_general_dilated(x, l['w'], (1, 1), PADS[n], dimension_numbers=('NCHW', 'OIHW', 'NCHW')) + l['b'][None, :, None, None]
def mp(x, a, b): return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 1, a, b), (1, 1, a, b), 'VALID')
def fwd(p, x):                               # x: N x 1 x 32 x W  ->  N x T x 12
    x = mp(jax.nn.relu(conv(x, p['c1'], 'c1')), 2, 2); x = mp(jax.nn.relu(conv(x, p['c2'], 'c2')), 2, 2); x = mp(jax.nn.relu(conv(x, p['c3'], 'c3')), 2, 1)
    x = jax.nn.relu(conv(x, p['c4'], 'c4')); x = jax.nn.relu(conv(x, p['c5'], 'c5')); x = jax.nn.relu(conv(x, p['c6'], 'c6'))
    return conv(x, p['out'], 'out')[:, :, 0, :].transpose(0, 2, 1)
def unpack(s):
    bits = np.unpackbits(np.frombuffer(base64.b64decode(s['b']), np.uint8), bitorder='little')
    return bits[:32 * s['w']].reshape(32, s['w']).astype(np.float32)
HU_OK = lambda l: len(l) == 9 and l.startswith('190')
real, test = [], []
hold = os.environ.get('HOLD', '')
for fn in files:
    for s in json.load(open(fn))['samples']:
        if s['f'] not in ('hu', 'contr', 'ril', 'scart', 'data'): continue
        if s.get('src') is None:                                   # dall'app: solo etichette affidabili
            if s['p'] == s['l']:
                if s['t'] < int(os.environ.get('FROM_T', 0)): continue  # export vecchi con valori non controllati
                if s['f'] == 'hu' and not HU_OK(s['l']): continue
        lab = [CH.index(c) + 1 for c in s['l'] if c in CH]
        if not lab: continue
        (test if hold and s.get('src') == hold else real).append((unpack(s), lab, s['f'], s['l']))
print('reali', len(real), 'test', len(test), flush=True)
WMAX, B_REAL, B_SYN = 256, 24, 40
rng = np.random.default_rng(0)
def aug(img):
    h, w = img.shape; g = (img * 255).astype(np.uint8)
    # taglia il bordo e rimetti a destra/sinistra, inclina, scala in altezza, sposta
    sh = rng.uniform(-.25, .25); sy = rng.uniform(.85, 1.08); sx = rng.uniform(.85, 1.15); dy = rng.uniform(-2, 2)
    M = np.float32([[sx, sh, -sh * h / 2], [0, sy, (1 - sy) * h / 2 + dy]])
    W2 = int(w * sx + 6); g = cv2.warpAffine(g, M, (W2, h))
    r = rng.random()
    if r < .25: g = cv2.dilate(g, np.ones((2, 2), np.uint8))
    elif r < .4: g = cv2.erode(g, np.ones((2, 1), np.uint8))
    if rng.random() < .2: y = int(rng.integers(h - 5, h)); cv2.line(g, (0, y), (W2, y + int(rng.integers(-1, 2))), 255, 1)
    out = (g > 100).astype(np.float32)
    xs = np.nonzero(out.sum(0))[0]
    if len(xs): out = out[:, max(0, xs[0] - int(rng.integers(2, 6))):xs[-1] + int(rng.integers(3, 7))]
    return out
def batch():
    items = []
    for i in rng.integers(0, len(real), B_REAL): items.append((aug(real[i][0]), real[i][1]))
    for _ in range(B_SYN): a, l, x = st.sample(); items.append((a[:, :max(8, x + 4)], l))
    X = np.zeros((len(items), 1, 32, WMAX), np.float32); LP = np.ones((len(items), WMAX // 4), np.float32)
    LB = np.zeros((len(items), 16), np.int32); LBP = np.ones((len(items), 16), np.float32)
    for j, (a, l) in enumerate(items):
        a = a[:, :WMAX]; X[j, 0, :, :a.shape[1]] = a; LP[j, :a.shape[1] // 4] = 0
        l = l[:16]; LB[j, :len(l)] = l; LBP[j, :len(l)] = 0
    return X, LP, LB, LBP
def loss_fn(p, X, LP, LB, LBP): return optax.ctc_loss(fwd(p, X), LP, LB, LBP, blank_id=0).mean()
params = load(src)
STEPS = int(os.environ.get('STEPS', 1500))
opt = optax.adam(optax.cosine_decay_schedule(float(os.environ.get('LR', 3e-4)), STEPS)); state = opt.init(params)
@jax.jit
def step(p, s, X, LP, LB, LBP):
    l, g = jax.value_and_grad(loss_fn)(p, X, LP, LB, LBP); u, s = opt.update(g, s, p); return optax.apply_updates(p, u), s, l
fj = jax.jit(fwd)
def greedy(p, img):
    W = img.shape[1] + (-img.shape[1]) % 4; x = np.zeros((1, 1, 32, W), np.float32); x[0, 0, :, :img.shape[1]] = img
    o = np.asarray(fj(p, x))[0].argmax(1); return ''.join(CH[k - 1] for j, k in enumerate(o) if k and (j == 0 or k != o[j - 1]))
def report(p, tag):
    for name, data in (('train', real), ('test', test)):
        if not data: continue
        ok = sum(greedy(p, a) == l for a, _, _, l in data); hu = [(greedy(p, a), l) for a, _, f, l in data if f == 'hu']
        print(f'{tag} {name}: esatti {ok}/{len(data)}', ('HU ' + ' '.join(f'{g}/{l}' for g, l in hu)) if name == 'test' else '', flush=True)
report(params, 'prima')
t0 = time.time()
for i in range(STEPS):
    params, state, l = step(params, state, *batch())
    if i % 250 == 0: print('passo', i, 'loss', round(float(l), 3), round(time.time() - t0), 's', flush=True)
report(params, 'dopo')
save(params, dst); print('salvato', dst)

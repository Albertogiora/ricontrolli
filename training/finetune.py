# Riaddestra il modello con le correzioni esportate dall'app (ricontrolli-correzioni-*.json),
# mescolandole a sequenze sintetiche MNIST per non dimenticare le altre cifre.
# Uso: python3 finetune.py mnist.npz modello.pt nuovo.pt correzioni1.json [correzioni2.json ...]
import os, re, sys, json, base64, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.argv, args = sys.argv[:2], sys.argv[1:]
import seqtrain as st                     # generatore sintetico e rete
mnist, src, dst, files = args[0], args[1], args[2], args[3:]
CH = '0123456789-'

HU_OK = re.compile(r'^190\d{6}$')
def trusted(s):
    # corretto a mano: è un'etichetta vera
    if s['p'] != s['l']: return True
    if os.environ.get('ONLY_FIXED'): return False
    # non toccato: lo teniamo solo se è plausibile (un HU valido); gli altri campi non si possono verificare
    if s['f'] == 'hu': return bool(HU_OK.match(s['l']))
    return True

real, skipped = [], 0
for fn in files:
    for s in json.load(open(fn))['samples']:
        if not trusted(s): skipped += 1; continue
        bits = np.frombuffer(base64.b64decode(s['b']), np.uint8)
        img = np.unpackbits(bits, bitorder='little')[:32 * s['w']].reshape(32, s['w']).astype(np.float32)
        lab = [CH.index(c) + 1 for c in s['l'] if c in CH]
        if lab: real.append((img, lab))
print('scartati (letture non toccate e non plausibili)', skipped)
if not real: sys.exit('Nessun campione affidabile: correggi qualche lettura nell\'app prima di esportare.')
print('campioni reali', len(real), '· corretti a mano', sum(1 for fn in files for s in json.load(open(fn))['samples'] if s['p'] != s['l']))

def pad(imgs):
    W = max(i.shape[1] for i in imgs); W += (-W) % 4
    return torch.tensor(np.stack([np.pad(i, ((0, 0), (0, W - i.shape[1]))) for i in imgs])).unsqueeze(1)

dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
m = st.Seq(); m.load_state_dict(torch.load(src, map_location='cpu')); m.to(dev)
opt = torch.optim.Adam(m.parameters(), 3e-4); ctc = nn.CTCLoss(blank=0, zero_infinity=True)
rng = np.random.default_rng(1)
for step in range(int(os.environ.get('STEPS', 3000))):
    batch = [real[i] for i in rng.integers(0, len(real), 48)] + [(a[:, :max(8, x)], l) for a, l, x in (st.sample() for _ in range(80))]
    x = pad([b[0] for b in batch]).to(dev)
    lp = F.log_softmax(m(x), 1).permute(2, 0, 1).cpu()
    tg = torch.tensor([c for _, l in batch for c in l]); tl = torch.tensor([len(l) for _, l in batch])
    loss = ctc(lp, tg, torch.full((len(batch),), lp.shape[0], dtype=torch.long), tl)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0: print('passo', step, 'loss', round(loss.item(), 3), flush=True)
m.eval(); ok = 0
with torch.no_grad():
    for img, lab in real:
        o = m(torch.tensor(img)[None, None].to(dev)).argmax(1)[0].cpu().tolist()
        ok += [k for j, k in enumerate(o) if k and (j == 0 or k != o[j - 1])] == lab
print(f'letti giusti sui campioni reali: {ok}/{len(real)} (misura ottimistica: sono gli stessi usati per addestrare)')
torch.save(m.state_dict(), dst)

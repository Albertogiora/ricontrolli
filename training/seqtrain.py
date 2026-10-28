# Modello che legge una riga di cifre intera (CTC): 0 = vuoto, 1..10 = cifre 0-9, 11 = separatore (. o -)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2, sys, time
rng = np.random.default_rng(0)
d = np.load(sys.argv[1], allow_pickle=False)
X = np.concatenate([d['x_train'], d['x_test']]); Y = np.concatenate([d['y_train'], d['y_test']])
crops = []
for g in X:
    ys, xs = np.nonzero(g > 30); crops.append(g[ys.min():ys.max()+1, xs.min():xs.max()+1])
byd = [np.nonzero(Y == k)[0] for k in range(10)]
HH, WW = 32, 320

def sample():
    kind = rng.random()
    if kind < .35: s = [1, 9, 0] + list(rng.integers(0, 10, 6)) if rng.random() < .7 else list(rng.integers(0, 10, 9))
    elif kind < .5: s = list(rng.integers(0, 10, 2)) + [10] + list(rng.integers(0, 10, 2)) + ([10] + list(rng.integers(0, 10, 2)) if rng.random() < .6 else [])
    else: s = list(rng.integers(0, 10, rng.integers(1, 5)))
    h = rng.integers(17, 27); canvas = np.zeros((HH, WW + 60), np.float32); x = rng.integers(0, 12)
    shear = rng.uniform(-.35, .35); placed = []
    for c in s:
        x = max(0, x)
        if c == 10:
            if rng.random() < .5: cv2.circle(canvas, (int(x + 3), int(HH/2 + h/2 - 3 + rng.integers(-2, 3))), int(rng.integers(1, 3)), 255, -1); x += rng.integers(6, 12); placed.append(c)
            else:
                y = int(HH/2 + rng.integers(-3, 4)); L = int(rng.integers(6, 13)); cv2.line(canvas, (int(x), y), (int(x + L), y + int(rng.integers(-2, 3))), 255, int(rng.integers(1, 3))); x += L + rng.integers(2, 7); placed.append(c)
            continue
        g = crops[rng.choice(byd[c])].astype(np.float32)
        gh = int(h * rng.uniform(.9, 1.1)) if c != 1 else int(h * rng.uniform(.95, 1.15))
        gw = max(2, int(g.shape[1] * gh / g.shape[0] * rng.uniform(.8, 1.15)))
        g = cv2.resize(g, (gw, gh))
        y0 = int((HH - gh) / 2 + rng.integers(-3, 4)); y0 = min(max(y0, 0), HH - gh)
        if x + gw >= WW: break
        canvas[y0:y0+gh, int(x):int(x)+gw] = np.maximum(canvas[y0:y0+gh, int(x):int(x)+gw], g)
        x += gw + rng.integers(-5, 6); placed.append(c)
    s = placed or [0]
    canvas = canvas[:, :WW]
    M = np.float32([[1, shear, -shear * HH / 2], [0, 1, 0]]); canvas = cv2.warpAffine(canvas, M, (WW, HH))
    r = rng.random()
    if r < .3: canvas = cv2.dilate(canvas, np.ones((2, 2), np.uint8))
    elif r < .45: canvas = cv2.erode(canvas, np.ones((2, 2), np.uint8))
    if rng.random() < .35: y = int(rng.integers(HH - 6, HH)); cv2.line(canvas, (0, y), (int(min(x, WW)), y + int(rng.integers(-2, 3))), 255, 1)   # sottolineatura/riga
    if rng.random() < .3: xx = int(rng.integers(0, 4)); cv2.line(canvas, (xx, 0), (xx, HH), 255, 1)                                            # resto di riga verticale
    for _ in range(rng.integers(0, 4)): cv2.circle(canvas, (int(rng.integers(0, WW)), int(rng.integers(0, HH))), 1, 255, -1)                # puntini
    canvas = (canvas > 90).astype(np.float32)
    return canvas, [c + 1 for c in s], int(min(x, WW))

class Seq(nn.Module):
    def __init__(s):
        super().__init__()
        s.c1 = nn.Conv2d(1, 32, 3, padding=1); s.c2 = nn.Conv2d(32, 64, 3, padding=1); s.c3 = nn.Conv2d(64, 96, 3, padding=1)
        s.c4 = nn.Conv2d(96, 96, 3, padding=1); s.c5 = nn.Conv2d(96, 128, (4, 3), padding=(0, 1)); s.c6 = nn.Conv2d(128, 128, (1, 5), padding=(0, 2)); s.out = nn.Conv2d(128, 12, 1)
    def forward(s, x):                       # x: N x 1 x 32 x W
        x = F.max_pool2d(F.relu(s.c1(x)), 2)  # 16 x W/2
        x = F.max_pool2d(F.relu(s.c2(x)), 2)  # 8 x W/4
        x = F.max_pool2d(F.relu(s.c3(x)), (2, 1))  # 4 x W/4
        x = F.relu(s.c4(x)); x = F.relu(s.c5(x)); x = F.relu(s.c6(x))  # 1 x W/4
        return s.out(x).squeeze(2)           # N x 12 x W/4

if __name__ == '__main__':
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 120000
    EP = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    t0 = time.time(); data = [sample() for _ in range(N)]; print('generati', N, 'in', round(time.time() - t0), 's', flush=True)
    imgs = torch.tensor(np.stack([a for a, _, _ in data])).unsqueeze(1)
    labs = [l for _, l, _ in data]
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    m = Seq().to(dev); opt = torch.optim.Adam(m.parameters(), 1e-3); ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=EP * (N // 128 + 1))
    for ep in range(EP):
        p = torch.randperm(N); tl = 0
        for i in range(0, N, 128):
            b = p[i:i+128]; x = imgs[b].to(dev)
            lp = F.log_softmax(m(x), 1).permute(2, 0, 1).cpu()
            tg = torch.tensor([c for j in b.tolist() for c in labs[j]]); tlen = torch.tensor([len(labs[j]) for j in b.tolist()])
            loss = ctc(lp, tg, torch.full((len(b),), lp.shape[0], dtype=torch.long), tlen)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item()
        # accuratezza su stringhe nuove
        m.eval(); ok = 0; tst = [sample() for _ in range(1000)]
        with torch.no_grad():
            o = m(torch.tensor(np.stack([a for a, _, _ in tst])).unsqueeze(1).to(dev)).argmax(1).cpu().numpy()
        for (a, l, _), seq in zip(tst, o):
            dec = [k for j, k in enumerate(seq) if k and (j == 0 or k != seq[j-1])]
            ok += dec == l
        m.train(); print('epoch', ep, 'loss', round(tl / (N // 128), 3), 'stringhe esatte', ok / 1000, flush=True)
    torch.save(m.state_dict(), sys.argv[2])

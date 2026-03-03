# Orbits Systems

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time

print("Project Synapse - Orbits Systems")
print("="*50)

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

BOTTLENECK = 64
N_CELLS    = 32
N_STEPS    = 8
LR         = 1e-3
BATCH      = 64
EPOCHS     = 300
PATIENCE   = 40  # early stopping

TRAIN_NAMES = ["blorp","ziff","mook","fral","snee","wump",
               "grix","plonk","veem","quaz","trid","yoof"]

TEST_NAMES  = ["gliff","tazzle","primble","zunked","bleeple","groff",
               "vorpal","mimsy","borogrove","jubjub","fleem","gruntle"]

# Difficulty tiers
EASY = [
    ("All {a}s are {b}s. X is a {a}. Is X a {b}?", 1),
    ("All {a}s are {b}s. X is not a {b}. Is X a {a}?", 0),
    ("No {a} is a {b}. X is a {a}. Is X a {b}?", 0),
]

MEDIUM = [
    ("All {a}s are {b}s. All {b}s are {c}s. X is a {a}. Is X a {c}?", 1),
    ("All {a}s are {b}s. All {b}s are {c}s. X is a {c}. Is X an {a}?", 0),
    ("{a} is bigger than {b}. {b} is bigger than {c}. Is {a} bigger than {c}?", 1),
    ("{a} is bigger than {b}. {b} is bigger than {c}. Is {c} bigger than {a}?", 0),
    ("All {a}s are {b}s. No {b} is a {c}. X is a {a}. Is X a {c}?", 0),
]

HARD = [
    # 4-step chains
    ("All {a}s are {b}s. All {b}s are {c}s. X is not a {c}. Is X a {a}?", 0),
    ("All {a}s are {b}s. All {b}s are {c}s. All {c}s are {d}s. X is a {a}. Is X a {d}?", 1),
    ("All {a}s are {b}s. All {b}s are {c}s. All {c}s are {d}s. X is a {d}. Is X an {a}?", 0),
    # Contrapositive
    ("All {a}s are {b}s. All {b}s are {c}s. X is not a {c}. Is X an {a}?", 0),
    # Mixed
    ("{a} is heavier than {b}. {b} is heavier than {c}. {c} is heavier than {d}. Is {a} heavier than {d}?", 1),
    ("{a} is heavier than {b}. {b} is heavier than {c}. {c} is heavier than {d}. Is {d} heavier than {a}?", 0),
    ("All {a}s are {b}s. Some {b}s are {c}s. X is a {a}. Is X definitely a {c}?", 0),
]

def make_puzzles(names, n=5000):
    out = []
    extra = names + ["W","X","Y","Z"]  # for 4-variable puzzles
    for _ in range(n):
        tier = random.choices([EASY, MEDIUM, HARD], weights=[0.2, 0.4, 0.4])[0]
        tmpl, label = random.choice(tier)
        needed = tmpl.count("{")
        vars_needed = len(set(c for c in "abcd" if "{"+c+"}" in tmpl))
        sample = random.sample(names, min(vars_needed, len(names)))
        a = sample[0] if len(sample) > 0 else "x"
        b = sample[1] if len(sample) > 1 else "y"
        c = sample[2] if len(sample) > 2 else "z"
        d = sample[3] if len(sample) > 3 else "w"
        text = tmpl.format(a=a, b=b, c=c, d=d)
        out.append((text, label))
    random.shuffle(out)
    return out

print("Generating puzzles (easy/medium/hard mix)...")
train_puzzles = make_puzzles(TRAIN_NAMES, 5000)
test_puzzles  = make_puzzles(TEST_NAMES,  1000)
print(f"Train: {len(train_puzzles)} | Test (novel names): {len(test_puzzles)}")
print(f"Hard example: {next(t for t,_ in train_puzzles if 'heavier' in t or 'All' in t and 'All' in t[10:])}")

# Load GPT-2
print("\nLoading GPT-2 small...")
from transformers import GPT2Tokenizer, GPT2Model

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
gpt2 = GPT2Model.from_pretrained("gpt2")
gpt2.eval()
for p in gpt2.parameters():
    p.requires_grad = False
GPT2_DIM = 768
print(f"Loaded. {sum(p.numel() for p in gpt2.parameters()):,} params frozen.")

def cache_hidden(puzzles, batch_size=32):
    texts  = [p[0] for p in puzzles]
    labels = torch.tensor([p[1] for p in puzzles], dtype=torch.long)
    hidden = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc   = tokenizer(batch, return_tensors="pt", padding=True,
                         truncation=True, max_length=80)
        with torch.no_grad():
            out = gpt2(**enc)
        hidden.append(out.last_hidden_state.mean(1))
        print(f"  {min(i+batch_size,len(texts))}/{len(texts)}", end="\r")
    print()
    return torch.cat(hidden, dim=0), labels

print("\nCaching hidden states...")
t0 = time.time()
train_H, train_Y = cache_hidden(train_puzzles)
test_H,  test_Y  = cache_hidden(test_puzzles)
print(f"Done in {time.time()-t0:.1f}s")

# Models

class BaselineProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(GPT2_DIM, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, 64),       nn.GELU(), nn.LayerNorm(64),
            nn.Linear(64, 2)
        )
    def forward(self, x): return self.net(x)


class TernaryLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.5)
        self.bias   = nn.Parameter(torch.zeros(out_f))
    def forward(self, x):
        q     = torch.sign(self.weight) * (self.weight.abs() > 0.1).float()
        w_ste = self.weight + (q - self.weight).detach()
        return F.linear(x, w_ste, self.bias)
    def stats(self):
        q = torch.sign(self.weight) * (self.weight.abs() > 0.1).float()
        t = q.numel()
        return {k:(q==v).sum().item()/t for k,v in [("-1",-1),("0",0),("1",1)]}


class ReasoningCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(GPT2_DIM, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, BOTTLENECK), nn.LayerNorm(BOTTLENECK),
        )
        self.read_addr  = TernaryLinear(BOTTLENECK, N_CELLS)
        self.write_addr = TernaryLinear(BOTTLENECK, N_CELLS)
        self.write_val  = TernaryLinear(BOTTLENECK, BOTTLENECK)
        self.mem_mix    = nn.Linear(BOTTLENECK*2, BOTTLENECK)
        self.mem_norm   = nn.LayerNorm(BOTTLENECK)
        self.Wr         = nn.Linear(BOTTLENECK*3, BOTTLENECK)
        self.Wz         = nn.Linear(BOTTLENECK*3, BOTTLENECK)
        self.Wn         = nn.Linear(BOTTLENECK*3, BOTTLENECK)
        self.hnorm      = nn.LayerNorm(BOTTLENECK)
        self.gate_emb   = nn.Embedding(N_STEPS, BOTTLENECK)
        self.gate_fc    = nn.Linear(BOTTLENECK*2, 1)
        nn.init.constant_(self.gate_fc.bias, -2.0)
        self.decoder = nn.Sequential(
            nn.Linear(BOTTLENECK, 64), nn.GELU(), nn.LayerNorm(64),
            nn.Linear(64, 2)
        )
        self.last_steps = 0
        self.last_gates = []

    def forward(self, x, record=False):
        B     = x.size(0)
        z     = self.encoder(x)
        cells = torch.zeros(N_CELLS, BOTTLENECK)
        h     = torch.zeros(B, BOTTLENECK)
        addr  = torch.softmax(self.write_addr(z), dim=-1)
        cells = cells + addr.mean(0).unsqueeze(-1) * torch.tanh(self.write_val(z)).mean(0).unsqueeze(0)
        self.last_gates = []

        for s in range(N_STEPS):
            ra      = torch.softmax(self.read_addr(h), dim=-1)
            readout = ra @ cells
            mem_out = self.mem_norm(torch.tanh(self.mem_mix(torch.cat([h, readout], dim=-1))))
            inp = torch.cat([z, mem_out], dim=-1)
            xh  = torch.cat([inp, h], dim=-1)
            r   = torch.sigmoid(self.Wr(xh))
            gz  = torch.sigmoid(self.Wz(xh))
            n   = torch.tanh(self.Wn(torch.cat([inp, r*h], dim=-1)))
            h   = self.hnorm((1-gz)*h + gz*n)
            wa  = torch.softmax(self.write_addr(h), dim=-1)
            wv  = torch.tanh(self.write_val(h))
            cells = cells.detach() + wa.mean(0).unsqueeze(-1)*wv.mean(0).unsqueeze(0)
            cells = cells / cells.norm(dim=-1, keepdim=True).clamp(min=1.0)
            se   = self.gate_emb(torch.tensor(s))
            conf = torch.sigmoid(self.gate_fc(torch.cat([h+se, h], dim=-1))).squeeze(-1)
            self.last_steps = s+1
            if record: self.last_gates.append(conf.mean().item())
            if not self.training and conf.mean() > 0.8: break

        return self.decoder(h)

    def n_params(self): return sum(p.numel() for p in self.parameters())
    def ternary_stats(self):
        return {n:m.stats() for n,m in self.named_modules() if isinstance(m, TernaryLinear)}


def train(model, name, epochs=EPOCHS):
    opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = 0.0
    no_improve = 0
    t0   = time.time()
    for epoch in range(epochs):
        model.train()
        perm   = torch.randperm(len(train_H))
        Hp, Yp = train_H[perm], train_Y[perm]
        ls=c=t=0
        for i in range(0, len(Hp), BATCH):
            xb,yb = Hp[i:i+BATCH], Yp[i:i+BATCH]
            opt.zero_grad()
            logits = model(xb)
            loss   = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ls += loss.item()
            c  += (logits.detach().argmax(-1)==yb).sum().item()
            t  += yb.size(0)
        sch.step()
        if (epoch+1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                te_acc = (model(test_H).argmax(-1)==test_Y).float().mean().item()
            print(f"  [{name}] Ep {epoch+1:3d} | {(time.time()-t0)/60:.1f}m | "
                  f"Train: {c/t*100:.1f}% | Test: {te_acc*100:.1f}%")
            if te_acc > best:
                best = te_acc
                no_improve = 0
            else:
                no_improve += 20
            if no_improve >= PATIENCE:
                print(f"  [{name}] Early stop at epoch {epoch+1}")
                break
    return best

print("\n" + "="*50)
print("Baseline MLP")
print("="*50)
baseline = BaselineProbe()
print(f"Params: {sum(p.numel() for p in baseline.parameters()):,}")
b_acc = train(baseline, "Baseline")

print("\n" + "="*50)
print("Reasoning Core")
print("="*50)
core = ReasoningCore()
print(f"Params: {core.n_params():,}")
c_acc = train(core, "Synapse ")

print("\n" + "="*50)
print("RESULTS")
print("="*50)
print(f"Baseline:        {b_acc*100:.1f}%")
print(f"Reasoning core:  {c_acc*100:.1f}%")
print(f"Improvement:     +{(c_acc-b_acc)*100:.1f}%")
print()

# Example trace on a hard puzzle
print("\nHard puzzle example:")
core.eval()
hard = [(i,p) for i,p in enumerate(test_puzzles) if "heavier" in p[0] or
        (p[0].count("All") >= 2)]
if hard:
    idx, (text, ans) = random.choice(hard[:20])
    with torch.no_grad():
        logits = core(test_H[idx].unsqueeze(0), record=True)
    pred = logits.argmax(-1).item()
    print(f"  {text}")
    print(f"  Label: {'Yes' if ans else 'No'} | Pred: {'Yes' if pred else 'No'} {'✓' if pred==ans else '✗'}")
    print(f"  Steps used: {core.last_steps}/{N_STEPS}")
    for s,g in enumerate(core.last_gates):
        print(f"  Step {s+1} [{'█'*int(g*20)}{'░'*(20-int(g*20))}] {g:.3f}")

st = core.ternary_stats()
if st:
    s = next(iter(st.values()))
    print(f"\nTernary: -1:{s['-1']*100:.0f}% 0:{s['0']*100:.0f}% 1:{s['1']*100:.0f}%")

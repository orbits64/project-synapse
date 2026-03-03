# Project Synapse

**Orbits Systems** — 2026

A controlled experiment testing whether a dedicated reasoning component outperforms a standard neural network baseline on multi-step logic problems — specifically problems requiring generalization to entities never seen during training.

---

## The Question

Does architectural specialization matter, or is a bigger standard model always the answer?

We isolated a ternary-weight reasoning module and compared it against an MLP baseline of comparable size. Both were trained on top of a frozen GPT-2 backbone, so the only variable was the classifier architecture.

---

## Results

| Run | Baseline (MLP) | Reasoning Component | Gap |
|-----|---------------|---------------------|-----|
| 1   | 75.1%         | 87.3%               | +12.2% |
| 2   | 68.5%         | 88.1%               | +19.6% |

Test set used entirely different entity names than training. The reasoning component generalized. The baseline overfit and stalled.

---

## How It Works

- **GPT-2 (frozen)** converts puzzle text to numerical representations. Neither model trains GPT-2 — it is a shared, fixed input layer.
- **Baseline** is a 3-layer MLP (214K parameters) — fully connected layers, no memory, no iteration.
- **Reasoning Component** (272K parameters) runs iterative steps over explicit memory cells using ternary weights {-1, 0, +1}, with a confidence gate that exits early on easier problems and uses all steps on harder ones.

---

## Run It

```bash
pip install torch transformers
python synapse.py
```

Reproduces the full experiment. Results print at the end. Takes about 2-3 minutes on GPU, longer on CPU.

---

## Paper

*Not All Weights Are Equal: A heterogeneous design for a reasoning-first model*
Beniven, Orbits Systems, 2026

Available on arXiv.

---

## License

MIT — use it, build on it, cite it if it helps you.

© 2026 Orbits Systems

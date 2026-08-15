# FermiAI — tiny neural networks on a 2011 Fermi GPU

FermiAI is a playground of **tiny neural networks that actually run inference on a
retro NVIDIA GTX 550 Ti (Fermi, 2011, OpenCL 1.1, <1 GB VRAM)** — trained on a
modern RTX 5060 Ti and served from the Fermi to your browser.

> Concept: **Fermi trains & thinks, GitHub distributes.** The Fermi is the "brain"
> (it runs the models); this repo is the "body" (code, weights, and a web UI that
> talks to it).

## What's inside

| Toy | What it is | How it runs |
|-----|-----------|-------------|
| **LLM Chat** | 6.98M-param word-level transformer, trained on UltraChat | OpenCL on Fermi; has a built-in calculator (`2+2 → 4`) |
| **Number Oracle** | tiny transformer trained on number sequences | OpenCL on Fermi (`3 6 9 12 → 15 18 21 24`) |
| **Baby Lab** | `a` / `gpu` / `rare` — proof a Fermi can learn arbitrary tiny languages | OpenCL on Fermi |
| **Digit Eye** | MNIST digit MLP | OpenCL on Fermi |

All models share one C/OpenCL engine (`tinyllm.c`) — a from-scratch transformer
(matmul + LayerNorm + GQA attention + GeGLU FFN) compiled for the Fermi.

## Live demo

Open the GitHub Pages site (link in repo About). It talks to the Fermi over the
network; the Fermi must be online for the live demo to answer.

## Run it yourself (local)

**1. Train (needs a CUDA GPU + PyTorch):**
```bash
python3 train_word.py      # -> tinyllm.bin   (the chatbot)
python3 train_number.py    # -> tinyllm_num.bin (number oracle)
python3 train_a.py         # -> tinyllm_a.bin
python3 train_2word.py     # -> tinyllm_2w.bin
python3 train_rare.py      # -> tinyllm_rare.bin
```

**2. Build & run the engine on an OpenCL device (e.g. the Fermi):**
```bash
gcc -O2 tinyllm.c -o tinyllm -lOpenCL -lm
./tinyllm --bin tinyllm.bin "hello"        # one-shot reply
./tinyllm --bin tinyllm.bin --server 9001  # TCP server on :9001
```

Binary format (magic `0x4C4C4D31`, ver 2):
`<magic><ver><vocab><D><NH><NL><BLOCK><FFN_MULT>` then per-token utf8 bytes,
then float32 weights.

**3. Web UI (runs on any PC, proxies to the Fermi):**
```bash
cd playground
python3 ui_server.py        # http://localhost:8090
```
Point `BACKEND` in `playground/index.html` at your Fermi host
(defaults to `http://localhost:8090` when served by `ui_server.py`).

## Files

- `tinyllm.c` — OpenCL transformer engine + TCP server + calculator
- `train_*.py` — trainers (PyTorch, from scratch)
- `word_ref.py`, `numpy_ref.py` — reference validators
- `cam_client.py` — webcam hand-tracking client that talks to the Fermi
- `playground/` — the web UI (HTML + Python proxy)
- `draw_diagram*.py` — the FermiAI architecture diagrams

## License

MIT — do whatever you want with a 2011 GPU.

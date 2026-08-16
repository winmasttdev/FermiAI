import os, glob, time, torch
from transformers import GPT2TokenizerFast
import train

DEVICE = "cpu"
OUT = "/home/winmastt/neko-llm/chat450/out"
SYS = "You are a friendly, helpful assistant who speaks like a real person.\n"

def load_latest_ckpt():
    fs = [f for f in glob.glob(os.path.join(OUT, "ckpt_*.pt")) if "final" not in f]
    if not fs:
        return None, None
    fs.sort(key=lambda f: int(os.path.basename(f).split("_")[1].split(".")[0]))
    return fs[-1], fs

tok = GPT2TokenizerFast.from_pretrained("gpt2")
EOS = tok.eos_token_id

def gen(m, prompt, max_new=120, temp=0.9, top_p=0.92, rep=1.15):
    s = SYS + "User: " + prompt + "\nAssistant: "
    ids = tok.encode(s)
    base = len(ids)
    for _ in range(max_new):
        x = torch.tensor(ids[-1024:]).unsqueeze(0)
        with torch.no_grad():
            logits, _ = m(x)
        lg = logits[0, -1] / temp
        seen = set(ids[-64:])
        for t in seen: lg[t] /= rep
        sl, si = torch.sort(lg, descending=True)
        cum = torch.cumsum(torch.softmax(sl, dim=-1), dim=-1)
        keep = cum < top_p; keep[0] = True
        mask = torch.zeros_like(lg, dtype=torch.bool)
        mask[si[:keep.sum()]] = True
        lg[~mask] = -1e9
        n = torch.multinomial(torch.softmax(lg, dim=-1), 1).item()
        if n == EOS: break
        ids.append(n)
    return tok.decode(ids[base:])

prompts = ["Hello! Who are you?", "What is 2+2? Explain simply."]
last = None
while True:
    ck, allf = load_latest_ckpt()
    if ck is None or ck == last:
        time.sleep(30); continue
    last = ck
    step = int(os.path.basename(ck).split("_")[1].split(".")[0])
    sd = torch.load(ck, map_location=DEVICE)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    m = train.GPT().to(DEVICE); m.eval(); m.load_state_dict(sd)
    lines = [f"\n##### step {step} #####"]
    for p in prompts:
        try: lines.append(f"USER: {p}\nASST: {gen(m, p)}")
        except Exception as e: lines.append(f"USER: {p}\nERR {e}")
    with open("/home/winmastt/neko-llm/chat450/cpu_history.txt", "a") as f:
        f.write("\n".join(lines) + "\n")
    print("checked", step, flush=True)
    del m; torch.cuda.empty_cache()
    time.sleep(30)

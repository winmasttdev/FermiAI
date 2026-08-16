import os, torch
from transformers import GPT2TokenizerFast
import train

torch.manual_seed(0)
DEVICE = "cpu"
sd = torch.load("out/ckpt_900.pt", map_location=DEVICE)
sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
m = train.GPT().to(DEVICE)
m.eval()
m.load_state_dict(sd)
tok = GPT2TokenizerFast.from_pretrained("gpt2")
EOS = tok.eos_token_id
SYS = "You are a friendly, helpful assistant who speaks like a real person.\n"

def gen(prompt, max_new=130, temp=0.9, top_p=0.92, rep=1.15):
    s = SYS + "User: " + prompt + "\nAssistant: "
    ids = tok.encode(s)
    base = len(ids)
    for _ in range(max_new):
        x = torch.tensor(ids[-1024:]).unsqueeze(0)
        with torch.no_grad():
            logits, _ = m(x)
        lg = logits[0, -1] / temp
        seen = set(ids[-64:])
        for t in seen:
            lg[t] /= rep
        sorted_logits, sorted_idx = torch.sort(lg, descending=True)
        cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        keep = cum < top_p; keep[0] = True
        mask = torch.zeros_like(lg, dtype=torch.bool)
        mask[sorted_idx[:keep.sum()]] = True
        lg[~mask] = -1e9
        probs = torch.softmax(lg, dim=-1)
        n = torch.multinomial(probs, 1).item()
        if n == EOS: break
        ids.append(n)
    return tok.decode(ids[base:])

prompts = [
    "Hello! Who are you?",
    "What is 2+2? And explain why.",
    "Write a short Python function that reverses a string.",
    "What is a fun fact about space?",
]
out = []
for p in prompts:
    try:
        r = gen(p)
    except Exception as e:
        r = "ERR " + str(e)
    out.append(f"USER: {p}\nASSISTANT: {r}\n")
with open("cpu_test.txt", "w") as f:
    f.write("\n".join(out))
print("done", flush=True)
for o in out: print(o, flush=True)

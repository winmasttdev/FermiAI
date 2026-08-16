import os, glob, torch
from transformers import GPT2TokenizerFast
import train
DEVICE="cpu"
OUT="/home/winmastt/neko-llm/chat450/out"
SYS="You are a friendly, helpful assistant who speaks like a real person.\n"
fs=[f for f in glob.glob(OUT+"/ckpt_*.pt") if "final" not in f]
fs.sort(key=lambda f:int(os.path.basename(f).split("_")[1].split(".")[0]))
ck=fs[-1]
print("using",os.path.basename(ck),flush=True)
sd=torch.load(ck,map_location=DEVICE); sd={k.replace("_orig_mod.",""):v for k,v in sd.items()}
m=train.GPT().to(DEVICE); m.eval(); m.load_state_dict(sd)
tok=GPT2TokenizerFast.from_pretrained("gpt2"); EOS=tok.eos_token_id
def gen(prompt,max_new=80,temp=0.0):
    s=SYS+"User: "+prompt+"\nAssistant: "; ids=tok.encode(s); base=len(ids)
    for _ in range(max_new):
        x=torch.tensor(ids[-1024:]).unsqueeze(0)
        with torch.no_grad():
            logits,_=m(x)
        if temp==0:
            n=torch.argmax(logits[0,-1]).item()
        else:
            lg=logits[0,-1]/temp
            n=torch.multinomial(torch.softmax(lg,-1),1).item()
        if n==EOS: break
        ids.append(n)
    return tok.decode(ids[base:])
for p in ["Hello! Who are you?","What is 2+2?","Write a Python function to reverse a string."]:
    print("USER:",p,flush=True)
    print("  greedy:",gen(p)[:260],flush=True)

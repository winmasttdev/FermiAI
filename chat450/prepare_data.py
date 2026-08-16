import os, sys, time, numpy as np
from datasets import load_dataset
from transformers import GPT2TokenizerFast

OUT = "/home/winmastt/neko-llm/chat450/data"
os.makedirs(OUT, exist_ok=True)
tok = GPT2TokenizerFast.from_pretrained("gpt2")
EOS = tok.eos_token_id
SYS = "You are a friendly, helpful assistant who speaks like a real person.\n"

# token budgets (approx)
CAP = {
    "tiny": 140_000_000,
    "fineweb": 120_000_000,
    "ultra": 70_000_000,
    "dolly": 8_000_000,
    "oasst": 30_000_000,
    "share": 30_000_000,
    "gsm": 6_000_000,
    "code": 25_000_000,
}

buf = np.zeros(1_000_000, dtype=np.int32)
bufn = 0
total = 0
fout = open(os.path.join(OUT, "train.bin"), "wb")
t0 = time.time()

def flush():
    global buf, bufn
    if bufn:
        fout.write(buf[:bufn].tobytes())
        bufn = 0

def add_ids(ids):
    global bufn, total
    for i in ids:
        buf[bufn] = i
        bufn += 1
        if bufn == buf.shape[0]:
            flush()
    total += len(ids)

def add_text(s, cap_key=None):
    ids = tok.encode(s, add_special_tokens=False)
    if cap_key:
        room = CAP[cap_key] - add_text.seen.get(cap_key, 0)
        if room <= 0:
            return False
        if len(ids) > room:
            ids = ids[:room]
        add_text.seen[cap_key] = add_text.seen.get(cap_key, 0) + len(ids)
    add_ids(ids + [EOS])
    return True

add_text.seen = {}

def fmt(turns):
    s = SYS
    for role, text in turns:
        s += role + ": " + text + "\n"
    return s

def log(src):
    print(f"[{time.time()-t0:6.0f}s] {src}: total={total/1e6:.1f}M seen={add_text.seen}", flush=True)

# ---------- TinyStories ----------
print("== TinyStories ==")
ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
for ex in ds:
    if not add_text(ex["text"], "tiny"): break
log("tiny")

# ---------- fineweb-edu ----------
print("== fineweb-edu ==")
ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
for ex in ds:
    if not add_text(ex["text"], "fineweb"): break
log("fineweb")

# ---------- ultrachat ----------
print("== ultrachat ==")
try:
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    for ex in ds:
        msgs = ex["messages"]
        turns = []
        for m in msgs:
            r = m["role"]
            if r == "user": turns.append(("User", m["content"]))
            elif r == "assistant": turns.append(("Assistant", m["content"]))
        if turns:
            if not add_text(fmt(turns), "ultra"): break
    log("ultra")
except Exception as e:
    print("ultrachat skip:", e)

# ---------- dolly ----------
print("== dolly ==")
try:
    ds = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True)
    for ex in ds:
        p = ex["instruction"]
        if ex.get("context"): p = ex["context"] + "\n" + p
        turns = [("User", p), ("Assistant", ex["response"])]
        if not add_text(fmt(turns), "dolly"): break
    log("dolly")
except Exception as e:
    print("dolly skip:", e)

# ---------- oasst1 ----------
print("== oasst1 ==")
try:
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    by_id = {ex["message_id"]: ex for ex in ds}
    for ex in ds:
        if ex["role"] != "assistant": continue
        chain = []
        cur = ex
        while cur is not None:
            chain.append(cur)
            pid = cur.get("parent_id")
            cur = by_id.get(pid) if pid else None
        chain.reverse()
        turns = []
        for m in chain:
            role = "User" if m["role"] == "prompter" else "Assistant"
            turns.append((role, m["text"]))
        if turns:
            if not add_text(fmt(turns), "oasst"): break
    log("oasst")
except Exception as e:
    print("oasst1 skip:", e)

# ---------- ShareGPT English ----------
print("== sharegpt ==")
try:
    ds = load_dataset("Aeala/ShareGPT_2022_English", split="train", streaming=True)
    for ex in ds:
        convs = ex.get("conversations")
        if not convs: continue
        turns = []
        for m in convs:
            f = m.get("from")
            if f == "human": turns.append(("User", m["value"]))
            elif f == "gpt": turns.append(("Assistant", m["value"]))
        if turns:
            if not add_text(fmt(turns), "share"): break
    log("share")
except Exception as e:
    print("sharegpt skip:", e)

# ---------- gsm8k (math) ----------
print("== gsm8k ==")
try:
    ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
    for ex in ds:
        turns = [("User", "Solve this step by step:\n" + ex["question"]), ("Assistant", ex["answer"])]
        if not add_text(fmt(turns), "gsm"): break
    log("gsm")
except Exception as e:
    print("gsm8k skip:", e)

# ---------- CodeAlpaca (code) ----------
print("== codealpaca ==")
try:
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train", streaming=True)
    for ex in ds:
        p = ex["instruction"]
        if ex.get("input"): p = p + "\n" + ex["input"]
        turns = [("User", p), ("Assistant", ex["output"])]
        if not add_text(fmt(turns), "code"): break
    log("code")
except Exception as e:
    print("codealpaca skip:", e)

flush()
fout.close()
print(f"DONE total={total} ({total*4/1e9:.2f} GB)", flush=True)

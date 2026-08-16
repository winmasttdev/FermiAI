import os, time, json, glob, urllib.request

OUT = "/home/winmastt/neko-llm/chat450/out"

def ckpts():
    fs = glob.glob(os.path.join(OUT, "ckpt_*.pt"))
    return [f for f in fs if "final" not in f]

print("monitor waiting for first checkpoint...", flush=True)
while not ckpts():
    time.sleep(20)
print("checkpoint found, letting chat server load...", flush=True)
time.sleep(40)

tests = [
    ("hello", "Hello! Who are you?"),
    ("math", "Solve step by step: if a train travels 60 km in 1.5 hours, what is its speed?"),
    ("code", "Write a short Python function that reverses a string."),
    ("chat", "What's a fun fact about space?"),
]
res = {}
for name, p in tests:
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:9001/api/chat",
            data=json.dumps({"prompt": p}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=120) as r:
            res[name] = json.loads(r.read().decode()).get("reply", "")
    except Exception as e:
        res[name] = "ERR " + str(e)
    print(f"test {name} done", flush=True)

with open("/home/winmastt/neko-llm/chat450/chat_test.txt", "w") as f:
    for k, v in res.items():
        f.write(f"=== {k} ===\n{v}\n\n")
print("chat_test.txt written", flush=True)

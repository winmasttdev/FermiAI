import sys, torch, numpy as np, importlib.util
spec=importlib.util.spec_from_file_location('r','ref_tinyllm.py')
r=importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
m,vocab=r.load('tinyllm.bin')
m.eval()
def gen(prompt, n=120, temp=0.8):
    ctx=f"You: {prompt}\nFriend: "
    ids=[vocab.index(c) if c in vocab else 0 for c in ctx]
    out=[]
    with torch.no_grad():
        for _ in range(n):
            idx=torch.tensor([ids[-256:]])
            lg=m(idx)[0,-1]
            if temp<=0:
                nxt=int(lg.argmax())
            else:
                p=torch.softmax(lg/temp,-1).numpy()
                nxt=int(np.random.choice(len(p),p=p))
            c=vocab[nxt]
            if c=='\n': break
            out.append(c); ids.append(nxt)
    return ''.join(out)
for p in ["what is gravity","how do i stay calm","tell me about the moon","i am sad"]:
    print(f"YOU: {p}\nFRIEND: {gen(p)}\n")

import train, torch, time
torch.manual_seed(0)
device="cuda"
m=train.GPT().to(device); m=torch.compile(m)
opt=torch.optim.AdamW(m.parameters(),lr=3e-4)
tr=train.DataLoader("train")
batch,accum,block=4,4,train.C.block_size
for _ in range(3):
    opt.zero_grad()
    for mi in range(accum):
        idx=tr.next(batch,block).to(device)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            _,loss=m(idx[:,:-1],idx[:,1:])
        (loss/accum).backward()
    opt.step()
torch.cuda.synchronize()
t=time.time(); N=20
for _ in range(N):
    opt.zero_grad()
    for mi in range(accum):
        idx=tr.next(batch,block).to(device)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            _,loss=m(idx[:,:-1],idx[:,1:])
        (loss/accum).backward()
    opt.step()
torch.cuda.synchronize()
dt=time.time()-t; toks=N*batch*accum*block
print("tok/s", round(toks/dt), "loss", round(float(loss),3), "memGB", round(torch.cuda.memory_allocated()/1e9,1))

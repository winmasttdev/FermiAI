import struct, numpy as np, sys

def load(path):
    with open(path,'rb') as f:
        magic,ver,VOCAB,D,NH,NL,BLOCK,FFN_MULT = struct.unpack("<iiiiiiii", f.read(32))
        FFN = D*FFN_MULT
        vocab = f.read(VOCAB).decode('latin-1')
        def rd(n): return np.frombuffer(f.read(n*4), dtype=np.float32)
        tok = rd(VOCAB*D).reshape(VOCAB,D)
        pos = rd(BLOCK*D).reshape(BLOCK,D)
        W={}; DH=D//NH
        for L in range(NL):
            W[f'l{L}_ln1g']=rd(D); W[f'l{L}_ln1b']=rd(D)
            wq=rd(D*D).reshape(D,D); W[f'l{L}_Wq']=wq.T
            W[f'l{L}_bq']=rd(D)
            wk=rd(D*D).reshape(D,D); W[f'l{L}_Wk']=wk.T
            W[f'l{L}_bk']=rd(D)
            wv=rd(D*D).reshape(D,D); W[f'l{L}_Wv']=wv.T
            W[f'l{L}_bv']=rd(D)
            wo=rd(D*D).reshape(D,D); W[f'l{L}_Wo']=wo.T
            W[f'l{L}_bo']=rd(D)
            W[f'l{L}_ln2g']=rd(D); W[f'l{L}_ln2b']=rd(D)
            w1=rd(FFN*D).reshape(FFN,D); W[f'l{L}_W1']=w1.T
            W[f'l{L}_b1']=rd(FFN)
            w2=rd(D*FFN).reshape(D,FFN); W[f'l{L}_W2']=w2.T
            W[f'l{L}_b2']=rd(D)
        lnfg=rd(D); lnfb=rd(D)
        wlm=rd(VOCAB*D).reshape(VOCAB,D); W['Wlm']=wlm.T
        blm=rd(VOCAB)
    return dict(VOCAB=VOCAB,D=D,NH=NH,NL=NL,BLOCK=BLOCK,FFN=FFN,DH=DH,
                tok=tok,pos=pos,W=W,lnfg=lnfg,lnfb=lnfb,blm=blm,vocab=vocab)

def ln(x,g,b,eps=1e-5):
    mu=x.mean(-1,keepdims=True); var=x.var(-1,keepdims=True)
    return (x-mu)/np.sqrt(var+eps)*g+b

def forward(m, ids):
    D=m['D']; NH=m['NH']; NL=m['NL']; FFN=m['FFN']; DH=m['DH']; BLOCK=m['BLOCK']
    W=m['W']; tok=m['tok']; pos=m['pos']
    T=len(ids)
    x = tok[ids].astype(np.float32) + pos[np.arange(T)%BLOCK]
    print("numpy x0=%.4f rms=%.4f"%(x[0,0], np.sqrt((x**2).mean())))
    for L in range(NL):
        h=ln(x,W[f'l{L}_ln1g'],W[f'l{L}_ln1b'])
        Q=h@W[f'l{L}_Wq']+W[f'l{L}_bq']
        K=h@W[f'l{L}_Wk']+W[f'l{L}_bk']
        V=h@W[f'l{L}_Wv']+W[f'l{L}_bv']
        ao=np.zeros((T,D),dtype=np.float32)
        for hidx in range(NH):
            off=hidx*DH
            Qh=Q[:,off:off+DH]; Kh=K[:,off:off+DH]; Vh=V[:,off:off+DH]
            sc=(Qh@Kh.T)/np.sqrt(DH)
            mask=np.triu(np.ones((T,T),dtype=bool),1)
            sc=np.where(mask,-1e30,sc)
            sc=sc-sc.max(-1,keepdims=True)
            e=np.exp(sc); a=e/e.sum(-1,keepdims=True)
            ctx=a@Vh
            ao[:,off:off+DH]=ctx
        tmp=ao@W[f'l{L}_Wo']+W[f'l{L}_bo']
        x=x+tmp
        h=ln(x,W[f'l{L}_ln2g'],W[f'l{L}_ln2b'])
        tmp=h@W[f'l{L}_W1']+W[f'l{L}_b1']
        def gelu(x): return 0.5*x*(1+np.tanh(0.7978845608*(x+0.044715*x*x*x)))
        tmp=gelu(tmp)
        h=tmp@W[f'l{L}_W2']+W[f'l{L}_b2']
        x=x+h
        print("numpy after layer %d x0=%.4f rms=%.4f"%(L,x[0,0],np.sqrt((x**2).mean())))
    h=ln(x,m['lnfg'],m['lnfb'])
    print("numpy lnf0=%.4f rms=%.4f"%(h[0,0],np.sqrt((h**2).mean())))
    logits=h@W['Wlm']+m['blm']
    return logits

m=load(sys.argv[1])
txt=sys.argv[2]
ids=[m['vocab'].index(c) if c in m['vocab'] else 0 for c in txt]
lg=forward(m,ids)
last=lg[-1]
print("numpy logits last[0..7]=",np.round(last[:8],4))
print("numpy argmax= %r"%(m['vocab'][int(last.argmax())]))

import torch, random, sys, time
from transformers import AutoModelForCausalLM, AutoTokenizer

random.seed(1234)
M='Qwen/Qwen2.5-0.5B-Instruct'
tok=AutoTokenizer.from_pretrained(M)
tok.padding_side='left'
model=AutoModelForCausalLM.from_pretrained(M, dtype=torch.float16, device_map='auto')
model.eval()

SYSTEM=("You are Friend, a warm, helpful, and knowledgeable companion who texts like a close human friend. "
        "Reply to the user in 1-2 short, natural sentences. Be genuinely useful: give real info, empathy, or a good take. "
        "Never say you are an AI, a chatbot, or an assistant. Never repeat the user's words. Just write your reply.")

TOPICS=["space","dogs","cooking","music","movies","books","school","work","friendship","love",
"health","sleep","dreams","travel","money","video games","art","nature","science","history",
"the future","coffee","rain","the beach","sports","music festivals","learning","anxiety",
"family","coding","robots","the ocean","mountains","food","exercise","weather","plants","cats"]

EMOS=["happy","sad","tired","lonely","stressed","excited","angry","grateful","bored","nervous","hopeful","down"]
REASONS=["work was rough","i got good news","everything feels heavy","i aced my exam","my friend ghosted me",
"i didnt sleep well","something nice happened","i feel stuck","its a beautiful day","i miss someone"]

GREET=["hey","yo","hi","hello","sup","whats good","morning","evening","howdy"]
DAYPART=["morning","afternoon","evening","night","weekend","day"]
ACTIONS=["quit my job","apologize to my friend","start exercising","learn to code","travel alone",
"tell them i like them","take a break","change my career","adopt a cat","wake up earlier"]
OPINIONS=["pineapple on pizza","morning vs night","cats vs dogs","books vs movies","city vs countryside",
"coffee vs tea","smartphones","social media","space exploration","AI"]

def user_prompts(n):
    out=[]
    while len(out)<n:
        r=random.random()
        if r<0.18:
            out.append(f"i am feeling {random.choice(EMOS)} because {random.choice(REASONS)}")
        elif r<0.32:
            out.append(f"{random.choice(GREET)}, hows your {random.choice(DAYPART)}")
        elif r<0.46:
            out.append(f"what do you think about {random.choice(TOPICS)}")
        elif r<0.56:
            out.append(f"tell me something interesting about {random.choice(TOPICS)}")
        elif r<0.64:
            out.append(f"should i {random.choice(ACTIONS)}")
        elif r<0.72:
            out.append(f"whats your take on {random.choice(OPINIONS)}")
        elif r<0.80:
            out.append(f"i need advice about {random.choice(TOPICS)}")
        elif r<0.88:
            out.append(f"can you explain {random.choice(TOPICS)} in simple terms")
        else:
            out.append(f"im {random.choice(['bored','curious','tired','happy'])} talk to me about {random.choice(TOPICS)}")
    return out[:n]

def make_messages(u):
    return [{'role':'system','content':SYSTEM},{'role':'user','content':u}]

def gen_batch(users, max_new=30):
    texts=[tok.apply_chat_template(make_messages(u), tokenize=False, add_generation_prompt=True) for u in users]
    inps=tok(texts, return_tensors='pt', padding=True)
    ids=inps['input_ids'].to(model.device)
    mask=inps['attention_mask'].to(model.device)
    with torch.no_grad():
        out=model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new,
                           do_sample=True, temperature=0.95, top_p=0.92, repetition_penalty=1.25)
    res=[]
    for i,u in enumerate(users):
        plen=int(mask[i].sum())
        txt=tok.decode(out[i][plen:], skip_special_tokens=True)
        txt=txt.split('\n')[0]
        if 'User:' in txt: txt=txt.split('User:')[0]
        if 'Friend:' in txt: txt=txt.split('Friend:')[0]
        txt=txt.strip().strip('"').strip()
        if txt: res.append((u,txt))
    return res

def main():
    n=int(sys.argv[1]) if len(sys.argv)>1 else 8000
    out_path='chat_big.txt'
    f=open(out_path,'w')
    written=0
    bs=24
    t0=time.time()
    while written<n:
        remaining=n-written
        users=user_prompts(min(bs,remaining))
        pairs=gen_batch(users)
        for u,r in pairs:
            f.write(f"You: {u}\nFriend: {r}\n\n")
            written+=1
        f.flush()
        if written%200==0 or written==len(pairs):
            rate=written/(time.time()-t0+1e-9)
            print(f"  {written}/{n}  ({rate:.1f}/s)", flush=True)
    f.close()
    print("DONE", written, "examples ->", out_path)

if __name__=='__main__':
    main()

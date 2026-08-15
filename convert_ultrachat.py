import sys, unicodedata
from datasets import load_dataset

OUT = "chat.txt"          # trainer reads this
MAX_CHARS = 120_000_000   # corpus size budget (chars)

# unicode punctuation -> ascii
REPL = {
    '\u2018':"'", '\u2019':"'", '\u201c':'"', '\u201d':'"', '\u2013':'-',
    '\u2014':'-', '\u2026':'...', '\u00b7':'-', '\u2022':'-', '\u00a0':' ',
    '\u200b':'', '\u2009':' ', '\ufeff':'', '\u2122':'(tm)', '\u00ae':'(r)',
    '\u00b0':' deg', '\u2032':"'", '\u2033':'"',
}
def clean(s):
    s = s.replace('\r',' ').replace('\t',' ')
    for k,v in REPL.items(): s = s.replace(k,v)
    # normalize remaining unicode to ascii, drop non-ascii
    s = unicodedata.normalize('NFKD', s)
    out = []
    for ch in s:
        o = ord(ch)
        if o < 128: out.append(ch)
        elif o == 10 or o == 13: pass
        else: out.append('?')
    s = ''.join(out)
    # collapse whitespace, flatten internal newlines to spaces
    s = s.replace('\n',' ').replace('\r',' ')
    while '  ' in s: s = s.replace('  ',' ')
    return s.strip()

def main():
    f = open(OUT,'w')
    total = 0
    splits = [('train_sft', 90000), ('test_sft', 8000)]
    for split, limit in splits:
        print("streaming", split, "limit", limit)
        ds = load_dataset('HuggingFaceH4/ultrachat_200k', split=split, streaming=True)
        cnt = 0
        for ex in ds:
            cnt += 1
            if cnt > limit: break
            msgs = ex.get('messages') or []
            # extract adjacent user->assistant pairs
            for i in range(0, len(msgs)-1, 1):
                a = msgs[i]; b = msgs[i+1]
                if a['role']!='user' or b['role']!='assistant': continue
                u = clean(a['content']); r = clean(b['content'])
                if not u or not r: continue
                if len(u) > 150: u = u[:150]
                block = f"You: {u}\nFriend: {r}\n\n"
                # keep each full turn inside the context window so the model
                # learns the You:/Friend: format (not just mid-answer fragments)
                if len(block) > 470:
                    cut = block[:470].rfind(' ')
                    if cut < 12: cut = 470
                    block = block[:cut].rstrip() + "\n\n"
                f.write(block)
                total += len(block)
            f.flush()
            if total >= MAX_CHARS:
                print("reached char budget"); break
        print("  done", split, "examples", cnt)
        if total >= MAX_CHARS: break
    f.close()
    print("WROTE", total, "chars ->", OUT)

if __name__=='__main__':
    main()

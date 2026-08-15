#!/usr/bin/env python3
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont

fp = fm.findfont(fm.FontProperties(family="DejaVu Sans"))
def F(sz): return ImageFont.truetype(fp, sz)

W, H = 1200, 1320
BG=(14,17,22); PANEL=(22,27,34); BORD=(48,54,61)
BLUE=(63,111,235); GREEN=(63,185,80); PURPLE=(163,89,230); ACCENT=(88,166,255)
TXT=(230,237,243); MUT=(139,148,158); GOLD=(240,200,90)

img = Image.new("RGB",(W,H),BG)
d = ImageDraw.Draw(img)

def rr(box, r, fill=PANEL, ol=BORD, w=2):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=ol, width=w)
def ct(cx, cy, s, font, fill=TXT, anchor="mm"):
    d.text((cx,cy), s, font=font, fill=fill, anchor=anchor)
def wrap(cx, cy, s, font, fill=TXT, lh=22, maxw=330):
    words=s.split(); lines=[]; cur=""
    for w in words:
        if d.textlength(cur+" "+w, font=font) > maxw and cur:
            lines.append(cur); cur=w
        else: cur=(cur+" "+w).strip()
    lines.append(cur)
    for i,l in enumerate(lines):
        d.text((cx, cy+(i-len(lines)/2)*lh), l, font=font, fill=fill, anchor="mm")
def arrow(x0,y0,x1,y1,col=MUT):
    d.line([(x0,y0),(x1,y1)], fill=col, width=3)
    import math
    ang=math.atan2(y1-y0,x1-x0); L=11
    d.polygon([(x1,y1),
               (x1-L*math.cos(ang-0.4), y1-L*math.sin(ang-0.4)),
               (x1-L*math.cos(ang+0.4), y1-L*math.sin(ang+0.4))], fill=col)

# ---------- title ----------
ct(W/2, 44, "THE FERMI AI", F(42), fill=TXT)
ct(W/2, 90, "как это всё работает  ·  a tiny neural net on a 2011 GPU", F(18), fill=MUT)
ct(W/2, 116, "нейросеть — это буквально математика", F(16), fill=GOLD)

# ---------- pipeline ----------
ct(60, 158, "PIPELINE", F(18), fill=ACCENT, anchor="lm")
PB=[
 (60,185,250,300,"Qwen2.5 0.5B","teacher\nwrites stories\n(on RTX)",PURPLE),
 (330,185,520,300,"TRAINING","RTX 5060 Ti\nword-level LLM\nUltraChat 120M",BLUE),
 (600,185,800,300,"FERMI","GTX 550 Ti\nOpenCL 1.1\n= the AI brain",GREEN),
 (870,185,1060,300,"YOUR PC","web UI :8000\n+ webcam eyes",BLUE),
]
for x0,y0,x1,y1,name,desc,col in PB:
    rr((x0,y0,x1,y1),14,fill=PANEL,ol=col,w=3)
    ct((x0+x1)/2,y0+32,name,F(19),fill=col)
    wrap((x0+x1)/2,(y0+y1)/2+16,desc,F(15),fill=TXT,lh=22,maxw=(x1-x0)-24)
for i in range(len(PB)-1):
    arrow(PB[i][2], PB[i][3], PB[i+1][0]-8, PB[i][3])
# YOU below YOUR PC
rr((935,320,1060,390),14,fill=PANEL,ol=ACCENT,w=3)
ct(997,355,"YOU 👋",F(18),fill=ACCENT)
arrow(960,PB[3][3],960,320)

# ---------- baby family ----------
ct(60, 430, "THE FERMI BABY FAMILY", F(18), fill=ACCENT, anchor="lm")
ct(60, 452, "same tiny transformer — different training", F(14), fill=MUT, anchor="lm")
cols=[(60,"FILE"),(420,"VOCABULARY"),(720,"BEHAVIOR")]
rows=[
 ("tinyllm.bin","12,000 words","smart chatbot  →  \"gravity is a force…\""),
 ("tinyllm_a.bin","1  ['a']","says only  →  \"a a a a a a…\""),
 ("tinyllm_2w.bin","2  ['a','gpu']","alternates  →  \"gpu a gpu a gpu a…\""),
 ("tinyllm_rare.bin","2, 'gpu'×1","saw 'gpu' ONCE → forgot → \"a a a…\""),
]
y=480; rowh=46
# header
for cx,lab in cols: ct(cx,y,lab,F(15),fill=GOLD,anchor="lm")
d.line([(60,y+22),(1130,y+22)],fill=BORD,width=1)
for r in rows:
    y+=rowh
    ct(60,y,r[0],F(16),fill=TXT,anchor="lm")
    ct(420,y,r[1],F(16),fill=GREEN,anchor="lm")
    ct(720,y,r[2],F(16),fill=TXT,anchor="lm")
    d.line([(60,y+18),(1130,y+18)],fill=BORD,width=1)

# ---------- experiments ----------
ct(60, 720, "WHAT THE EXPERIMENTS PROVED", F(18), fill=ACCENT, anchor="lm")
ex=[
 "1 letter 'a' trained  →  it ONLY knows 'a'   (твой \"ИИ только с буквой А\")",
 "2 words, alternating  →  learns the pattern, ping-pongs forever.",
 "2 words, 'gpu' once   →  CAN say 'gpu' but ~1 in 6,600 tokens (we measured!).",
 "128 'a' tokens fed in →  window caps at 128, can't overflow / \"die\".",
]
y=752
for e in ex:
    d.text((70,y), "•", font=F(16), fill=GOLD)
    wrap(92,y,e,F(15),fill=TXT,lh=21,maxw=1020)
    y+=44

# ---------- truth banner ----------
rr((60,960,1140,1110),16,fill=(20,24,16),ol=GOLD,w=3)
ct(W/2, 1010, "IT IS ALL JUST MATH", F(34), fill=GOLD)
ct(W/2, 1052, "token → number → attention → next-token odds", F(18), fill=TXT)
ct(W/2, 1082, "a billion-param AI runs the EXACT same loop — just bigger & with more words", F(15), fill=MUT)

# ---------- footer ----------
ct(W/2, 1160, "big AI = same loop, just scaled up  ·  our baby proves it fits on a 2011 GPU", F(15), fill=MUT)
ct(W/2, 1192, "winmavics & flaxss — welcome to the Fermi family 💚", F(15), fill=ACCENT)

img.save("/home/winmastt/neko-llm/fermi_diagram.png")
print("saved fermi_diagram.png", img.size)

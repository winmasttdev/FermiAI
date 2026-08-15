#!/usr/bin/env python3
import math
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont

fp = fm.findfont(fm.FontProperties(family="DejaVu Sans"))
def F(sz): return ImageFont.truetype(fp, sz)

W, H = 1200, 1540
BG=(14,17,22); PANEL=(22,27,34); BORD=(48,54,61)
BLUE=(63,111,235); GREEN=(63,185,80); PURPLE=(163,89,230); ACCENT=(88,166,255); GITHUB=(240,200,90)
TXT=(230,237,243); MUT=(139,148,158); GOLD=(240,200,90)

img = Image.new("RGB",(W,H),BG)
d = ImageDraw.Draw(img)

def rr(box, r, fill=PANEL, ol=BORD, w=2):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=ol, width=w)
def ct(cx, cy, s, font, fill=TXT, anchor="mm"):
    d.text((cx,cy), s, font=font, fill=fill, anchor=anchor)
def wrap(cx, cy, s, font, fill=TXT, lh=21, maxw=200):
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
    ang=math.atan2(y1-y0,x1-x0); L=11
    d.polygon([(x1,y1),(x1-L*math.cos(ang-0.4),y1-L*math.sin(ang-0.4)),
               (x1-L*math.cos(ang+0.4),y1-L*math.sin(ang+0.4))], fill=col)

# ---------- title ----------
ct(W/2, 44, "ИИ НА ВИДЕОКАРТЕ FERMI", F(38), fill=TXT)
ct(W/2, 86, "FermiAI отдаёт данные на GitHub, а GitHub хостит чатбот для людей", F(16), fill=MUT)
ct(W/2, 112, "нейросеть — это буквально математика", F(16), fill=GOLD)

# ---------- pipeline ----------
ct(60, 156, "ПУТЬ ОТ МОЗГА К ЛЮДЯМ", F(18), fill=ACCENT, anchor="lm")
PB=[
 (60,182,258,300,"Qwen2.5 0.5B","учитель\nпишет тексты\n(на RTX)",PURPLE),
 (288,182,486,300,"ОБУЧЕНИЕ","RTX 5060 Ti\nпословная LLM\nUltraChat 120M",BLUE),
 (516,182,714,300,"FermiAI","GTX 550 Ti\nOpenCL 1.1\n= мозг ИИ\n(Neko сервер)",GREEN),
 (744,182,942,300,"GitHub","хостит чатбот\n(как GitHub\nхостит код)",GITHUB),
 (972,182,1170,300,"ПОЛЬЗОВАТЕЛИ","общаются\nчерез GitHub",ACCENT),
]
for x0,y0,x1,y1,name,desc,col in PB:
    rr((x0,y0,x1,y1),14,fill=PANEL,ol=col,w=3)
    ct((x0+x1)/2,y0+30,name,F(17),fill=col)
    wrap((x0+x1)/2,(y0+y1)/2+14,desc,F(13),fill=TXT,lh=20,maxw=(x1-x0)-20)
for i in range(len(PB)-1):
    arrow(PB[i][2], PB[i][3], PB[i+1][0]-7, PB[i][3])
ct(W/2, 332, "личный ПК (git настроен) пушит данные FermiAI на GitHub", F(14), fill=MUT)

# ---------- baby family ----------
ct(60, 372, "СЕМЕЙСТВО FermiAI-МЛАДЕНЦЕВ", F(18), fill=ACCENT, anchor="lm")
ct(60, 394, "один и тот же крошечный трансформер — разное обучение", F(14), fill=MUT, anchor="lm")
cols=[(60,"ФАЙЛ"),(430,"СЛОВАРЬ"),(730,"ПОВЕДЕНИЕ")]
y=422; rowh=46
for cx,lab in cols: ct(cx,y,lab,F(15),fill=GOLD,anchor="lm")
d.line([(60,y+22),(1130,y+22)],fill=BORD,width=1)
rows=[
 ("tinyllm.bin","12 000 слов","умный чат-бот  →  «гравитация — это сила…»"),
 ("tinyllm_a.bin","1  ['а']","только  →  «а а а а а а…»"),
 ("tinyllm_2w.bin","2  ['а','gpu']","чередует  →  «gpu а gpu а gpu а…»"),
 ("tinyllm_rare.bin","2, 'gpu'×1","видел 'gpu' ОДИН раз → забыл → «а а а…»"),
]
for r in rows:
    y+=rowh
    ct(60,y,r[0],F(16),fill=TXT,anchor="lm")
    ct(430,y,r[1],F(16),fill=GREEN,anchor="lm")
    ct(730,y,r[2],F(16),fill=TXT,anchor="lm")
    d.line([(60,y+18),(1130,y+18)],fill=BORD,width=1)

# ---------- experiments ----------
ct(60, 636, "ЧТО ДОКАЗАЛИ ЭКСПЕРИМЕНТЫ", F(18), fill=ACCENT, anchor="lm")
ex=[
 "1 буква 'а' в обучении → знает только 'а'  (твой «ИИ только с буквой А»)",
 "2 слова, по очереди → выучил паттерн, повторяет его вечно",
 "2 слова, 'gpu' один раз → МОЖЕТ сказать 'gpu', но ~1 из 6600 (замерили!)",
 "128 букв 'а' на входе → окно = 128, не может переполниться / «умереть»",
]
y=668
for e in ex:
    d.text((70,y),"•",font=F(16),fill=GOLD)
    wrap(92,y,e,F(15),fill=TXT,lh=21,maxw=1010)
    y+=44

# ---------- plain explanation ----------
ct(60, 874, "ОБЪЯСНЕНИЕ ПО-ПРОСТОМУ", F(18), fill=ACCENT, anchor="lm")
rr((60,898,1140,1116),14,fill=(18,22,30),ol=BLUE,w=2)
pl=[
 "Нейросеть — это просто математика и алгоритм.",
 "У неё есть «мозг» — набор слов (датасет), которому её научили.",
 "Она берёт твои слова и подбирает СЛЕДУЮЩЕЕ подходящее слово.",
 "Её так хорошо натренировали, что она говорит почти как человек.",
 "FermiAI на видеокарте 2011 года делает то же самое — пока из одной буквы «а».",
]
yy=928
for p in pl:
    d.text((78,yy),"▸",font=F(16),fill=GOLD)
    wrap(102,yy,p,F(16),fill=TXT,lh=23,maxw=1000)
    yy+=42

# ---------- truth banner ----------
rr((60,1142,1140,1286),16,fill=(20,24,16),ol=GOLD,w=3)
ct(W/2, 1192, "ЭТО ВСЯ МАТЕМАТИКА", F(34), fill=GOLD)
ct(W/2, 1234, "слово → число → внимание → вероятность следующего слова", F(17), fill=TXT)
ct(W/2, 1264, "огромный ИИ делает ТОТ ЖЕ цикл — просто больше и с бóльшими словами", F(14), fill=MUT)

# ---------- footer ----------
ct(W/2, 1322, "FermiAI генерирует → GitHub хостит → люди общаются · всё на видеокарте 2011 года", F(14), fill=MUT)
ct(W/2, 1354, "winmavics и flaxss — добро пожаловать в семью FermiAI 💚", F(15), fill=ACCENT)

img.save("/home/winmastt/neko-llm/fermi_diagram_ru.png")
print("saved fermi_diagram_ru.png", img.size)

import random, sys

random.seed(7)

NAME = ["buddy", "friend", "pal", "dude", "mate", "bro"]
TOPIC = ["the weather", "music", "games", "food", "movies", "books", "coffee", "walking outside",
         "sleep", "weekends", "friends", "dreams", "the gym", "coding", "art", "cats", "travel"]
HOWQ = ["how are you", "how've you been", "how's it going", "how are things", "what's going on with you"]

def f(s): return s.format(t=random.choice(TOPIC), n=random.choice(NAME))

# ---- user line generators (return (category, text)) ----
def u_greet():
    g = random.choice(["hey", "hi", "hello", "yo", "sup", "good morning", "good evening", "hey there"])
    return ("GREET", g + (", " + random.choice(NAME) if random.random() < 0.5 else ""))
def u_how():
    return ("HOW", random.choice(HOWQ) + " " + random.choice(NAME) + "?")
def u_feel_good():
    return ("FEEL", random.choice(["i'm doing great", "i'm good thanks", "i'm pretty good",
                                   "feeling awesome", "i'm wonderful", "really good actually", "i'm happy"]))
def u_feel_bad():
    return ("FEEL", random.choice(["i'm a bit tired", "i'm kinda down", "i've been stressed",
                                   "i'm feeling off today", "been better tbh", "a little lonely"]))
def u_qtopic():
    return ("QTOPIC", random.choice(["what do you think about {t}", "do you like {t}",
                                     "ever get into {t}", "what's your take on {t}", "are you into {t}"]).format(t=random.choice(TOPIC)) + "?")
def u_like():
    return ("LIKE", random.choice(["i love {t}", "i'm really into {t}", "{t} is my favorite",
                                   "i'm obsessed with {t}", "i enjoy {t} a lot"]).format(t=random.choice(TOPIC)))
def u_chat():
    return ("CHAT", random.choice(["anyway, how's your day been", "so what are you up to", "tell me something fun",
                                   "what made you smile today", "you're fun to chat with", "this is nice",
                                   "wanna hang out", "i'm glad we're talking"]))

U_GEN = [u_greet, u_greet, u_how, u_feel_good, u_feel_bad, u_qtopic, u_like, u_chat]

# ---- friend replies conditioned on category ----
def friend_reply(cat, you_text):
    if cat == "GREET":
        return f(random.choice(["hey {n}!", "hi {n}!", "hello {n}!", "yo {n}!", "hey, good to see you"])) + " " + random.choice(
            ["how are you?", "how've you been?", "how's it going?"])
    if cat == "HOW":
        if random.random() < 0.6:
            return f(random.choice(["i'm doing great {n}", "i'm good thanks", "pretty good actually", "feeling awesome", "can't complain"]))
        return f(random.choice(["i've been better tbh", "kinda tired today", "a little stressed", "been a weird day"]))
    if cat == "FEEL":
        good = any(w in you_text for w in ["great", "good", "awesome", "wonderful", "happy", "fine"])
        if good:
            return f(random.choice(["that's awesome!", "love that for you", "so happy to hear it", "amazing!", "you deserve it"]))
        return f(random.choice(["aww i'm sorry to hear that", "that sucks, hope it gets better", "i'm here for you",
                                "sending good vibes", "take it easy ok?", "want to talk about it?"]))
    if cat == "QTOPIC":
        t = random.choice(TOPIC)
        return f(random.choice(["yeah i really like {t}", "not much but it's cool", "{t} is underrated honestly",
                                "i do enjoy {t} from time to time", "oh totally, {t} is great", "i'm super into {t}"])).format(t=t)
    if cat == "LIKE":
        return f(random.choice(["nice! i'm into {t} too", "same here, {t} is the best", "oh {t} is awesome",
                                "right? {t} is so fun", "haha i love {t} as well"]))
    if cat == "CHAT":
        return f(random.choice(["i'm good {n}, just chilling", "not much, enjoying our chat", "you made me smile tbh",
                                "same to you {n}!", "haha thanks, you're sweet", "for sure, let's hang"]))
    return f(random.choice(["haha true", "nice", "for real", "i feel that", "agreed"]))

def dialogue():
    lines = []
    cat, txt = u_greet()
    lines.append("You: " + txt)
    lines.append("Friend: " + friend_reply(cat, txt))
    turns = random.randint(1, 4)
    for _ in range(turns):
        gen = random.choice(U_GEN)
        cat, txt = gen()
        lines.append("You: " + txt)
        lines.append("Friend: " + friend_reply(cat, txt))
    if random.random() < 0.4:
        lines.append("You: " + u_chat()[1])
        lines.append("Friend: " + friend_reply("CHAT", ""))
    return "\n".join(lines)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
out = []
for _ in range(N):
    out.append(dialogue())
    out.append("")
text = "\n".join(out)
for _ in range(N // 5):
    text += "\n" + f(random.choice(["anyway, how's your day been", "you're fun to chat with", "this is nice"])) + " " + \
            f(random.choice(["i love {t}", "i'm really into {t}"])).format(t=random.choice(TOPIC)) + "."

with open("chat.txt", "w", encoding="utf-8") as fo:
    fo.write(text)
print("wrote chat.txt chars=", len(text))

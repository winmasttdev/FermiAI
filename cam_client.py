#!/usr/bin/env python3
"""
Hand-tracking client for the Fermi chatbot.
- Your PC is the "eyes": OpenCV grabs the webcam, finds the hand, classifies a
  coarse gesture, and sends that state to the Fermi (192.168.0.100:9001).
- The Fermi runs the AI brain (tinyllm): it reacts to your hand in natural language.
Run with a webcam connected:  python3 cam_client.py
"""
import cv2, numpy as np, socket, time, sys

FERMI = ("192.168.0.100", 9001)
WINDOW = "Fermi Hand Cam"

def fermi_reply(ctx, timeout=180):
    s = socket.create_connection(FERMI, timeout=timeout)
    s.sendall((ctx + "\n").encode("utf-8"))
    data = b""
    while not data.endswith(b"\n"):
        c = s.recv(4096)
        if not c:
            break
        data += c
    s.close()
    return data.decode("utf-8").strip()

def detect_hand(frame):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # skin mask (two HSV ranges to cover varying skin tones / lighting)
    lower1 = np.array([0, 25, 50], np.uint8)
    upper1 = np.array([20, 150, 255], np.uint8)
    lower2 = np.array([160, 25, 50], np.uint8)
    upper2 = np.array([180, 150, 255], np.uint8)
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 4000:
        return None
    x, y, bw, bh = cv2.boundingRect(c)
    cx, cy = x + bw / 2, y + bh / 2
    ratio = area / float(bw * bh)  # open palm spreads -> higher
    if ratio > 0.45:
        gesture = "open palm"
    elif ratio > 0.25:
        gesture = "relaxed hand"
    else:
        gesture = "fist"
    side = "left" if cx < w * 0.45 else ("right" if cx > w * 0.55 else "center")
    return {"x": int(cx), "y": int(cy), "bw": int(bw), "bh": int(bh),
            "gesture": gesture, "side": side, "area": int(area)}

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open webcam. Connect the camera and retry.")
        sys.exit(1)
    print("webcam open. Press 'q' to quit.")
    last_state = ""
    last_send = 0
    reply = "(say hi to the Fermi once it sees your hand)"
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        hand = detect_hand(frame)
        # overlay
        if hand:
            x, y, bw, bh = hand["x"] - hand["bw"] // 2, hand["y"] - hand["bh"] // 2, hand["bw"], hand["bh"]
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (63, 111, 235), 2)
            label = f'{hand["gesture"]} ({hand["side"]})'
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (63, 111, 235), 2)
            state = f'{hand["gesture"]} on the {hand["side"]}'
        else:
            state = "no hand visible"
        cv2.putText(frame, "FRIEND: " + reply, (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 237, 243), 2)
        cv2.imshow(WINDOW, frame)

        now = time.time()
        if state != last_state and (now - last_send) > 4:
            last_state = state
            last_send = now
            ctx = f"you : i am showing you my hand, it is a {state} friend :"
            try:
                reply = fermi_reply(ctx)
            except Exception as e:
                reply = f"(fermi error: {e})"
            print("hand:", state, "->", reply[:90])

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

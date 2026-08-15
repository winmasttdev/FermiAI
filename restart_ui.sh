#!/bin/bash
pids=$(pgrep -f "ui_server.py")
[ -n "$pids" ] && kill $pids 2>/dev/null
sleep 1
cd /home/winmastt/neko-llm/playground
setsid python3 ui_server.py > /home/winmastt/neko-llm/playground/ui.log 2>&1 < /dev/null &
echo restarted pid $!

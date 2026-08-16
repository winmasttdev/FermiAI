#!/bin/bash
ROOT=/home/winmastt/neko-llm
while true; do
  "$ROOT/start_tunnel.sh" > /tmp/tunnel.log 2>&1 &
  TPID=$!
  sleep 8
  SUB=$(grep -oP 'https://\K[a-z0-9-]+(?=\.serveousercontent\.com)' /tmp/tunnel.log | head -1)
  if [ -n "$SUB" ]; then
    FULL="https://$SUB.serveousercontent.com"
    for f in "$ROOT/index.html" "$ROOT/playground/index.html"; do
      python3 - "$f" "$FULL" <<'PY'
import sys,re
f,full=sys.argv[1],sys.argv[2]
s=open(f).read()
s=re.sub(r'const TUNNEL = "[^"]*"','const TUNNEL = "%s"'%full,s)
open(f,'w').write(s)
PY
    done
    cd "$ROOT" && git add -A index.html playground/index.html && git commit -q -m "auto: tunnel $SUB" && git push -q 2>/dev/null
    echo "$(date) tunnel up: $FULL" >> /tmp/tunnel_watchdog.log
  fi
  while kill -0 $TPID 2>/dev/null; do sleep 20; done
  echo "$(date) tunnel died, restarting" >> /tmp/tunnel_watchdog.log
  sleep 2
done

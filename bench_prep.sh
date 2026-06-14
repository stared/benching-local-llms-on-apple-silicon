#!/usr/bin/env bash
# Quiet the machine before benchmarking. Gracefully quits heavy apps, kills stray
# model servers, and prints the current state. Nothing destructive — apps just close.
set -u

APPS=("Beeper Desktop" "Slack" "Signal" "Dropbox" "Cursor" "Claude" "cmux"
      "Google Chrome" "Arc" "Spotify")

echo "== quitting heavy apps =="
for app in "${APPS[@]}"; do
  if pgrep -fi "$app" >/dev/null 2>&1; then
    osascript -e "tell application \"$app\" to quit" 2>/dev/null \
      && echo "  quit: $app" || echo "  (couldn't quit $app — close manually)"
  fi
done

echo "== killing stray model servers =="
pkill -f llama-server 2>/dev/null && echo "  killed llama-server" || true
pkill -f mlx_lm.server 2>/dev/null && echo "  killed mlx_lm.server" || true

echo "== state =="
echo -n "power: "; pmset -g batt | sed -n '2p'
echo -n "low power mode: "; pmset -g | awk '/lowpowermode/{print $2}'
echo "system used memory:"; vm_stat | awk '/Pages (active|wired|occupied)/'
echo "load average:"; uptime | sed 's/.*load/load/'

echo
echo "Optional, most invasive (re-enable after!):"
echo "  sudo mdutil -a -i off    # pause Spotlight indexing"
echo "  sudo mdutil -a -i on     # turn it back on when done"
echo
echo "Now run:  python3 bench.py"

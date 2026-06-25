#!/usr/bin/env sh
# One-command setup + launch for Conjure on macOS / Linux.
# Run it with:  sh start.sh    (or ./start.sh if you've made it executable)
# It just finds your Python and hands off to run.py, which does everything.
cd "$(dirname "$0")" || exit 1

if command -v python3 >/dev/null 2>&1; then
    exec python3 run.py "$@"
elif command -v python >/dev/null 2>&1; then
    exec python run.py "$@"
fi

echo "Python 3 was not found." >&2
echo "Install Python 3.10 or newer from https://python.org and run this again." >&2
exit 1

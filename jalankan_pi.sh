#!/bin/sh
# Jalankan Threck di Raspberry Pi.
#   ./jalankan_pi.sh --800x480     layar sentuh 800x480 (tata letak ringkas, layar penuh)
#   ./jalankan_pi.sh --fullscreen  layar penuh untuk monitor besar
cd "$(dirname "$0")"
if [ -d .venv ]; then
    . .venv/bin/activate
elif [ -d "$HOME/threck-venv" ]; then
    . "$HOME/threck-venv/bin/activate"
fi
exec python3 main.py "$@"

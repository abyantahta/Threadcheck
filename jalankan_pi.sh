#!/bin/sh
# Jalankan Threck di Raspberry Pi. Tambahkan --fullscreen untuk mode HMI layar penuh.
cd "$(dirname "$0")"
if [ -d .venv ]; then
    . .venv/bin/activate
elif [ -d "$HOME/threck-venv" ]; then
    . "$HOME/threck-venv/bin/activate"
fi
exec python3 main.py "$@"

#!/usr/bin/env bash
# Serve the 3D scene viewer (fetch() needs http://, not file://). Then open http://127.0.0.1:8765/
cd "$(dirname "$0")" && python3 -m http.server 8765 --bind 127.0.0.1

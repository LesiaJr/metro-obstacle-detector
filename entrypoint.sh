#!/bin/bash
set -e
source /opt/ros/humble/setup.bash

if [ $# -eq 0 ]; then
    exec python3 /workspace/src/gate_tube.py --help
fi

case "$1" in
    bash|sh|python3|python)
        exec "$@"
        ;;
esac

if [ -d "$1" ]; then
    DB=$(ls "$1"/*.db3 2>/dev/null | head -1)
    if [ -z "$DB" ]; then
        echo "Ошибка: в папке '$1' нет .db3 файла"
        exit 1
    fi
    shift
    set -- "$DB" "$@"
fi

exec python3 /workspace/src/gate_tube.py "$@"
#!/bin/sh
# Scientific parameters and seeds belong to the supplied study configuration.
set -eu
if [ "$#" -ne 1 ]; then
    echo "usage: $0 STUDY.yaml (optional WORKERS environment variable)" >&2
    exit 2
fi
exec uv run --locked smartsom batch "$1" --workers "${WORKERS:-2}"

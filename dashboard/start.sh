#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
docker build -t pluto-dashboard . && \
docker run --rm -it \
    --mount "src=$SCRIPT_DIR/notebooks,dst=/workspace/notebooks,type=bind" \
    --mount "src=$HOME/.cache/GPT/linguistic_analysis,dst=/workspace/linguistic_analysis,type=bind" \
    -p 7777:7777 pluto-dashboard "$@"

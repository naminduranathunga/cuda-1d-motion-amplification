#!/bin/bash

createVenv() {
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
}


cleanDir() {
    rm -rf processed
    rm -rf uploads
    mkdir -p processed
    mkdir -p uploads
}


# Check venv
if [ ! -d "venv" ]; then
    createVenv
fi

# If --clear flag is passed, clean the processed and uploads directories
if [ "$1" == "--clear" ]; then
    cleanDir
    echo "Directories cleaned successfully"
    exit 0
fi

# Build
mkdir -p build && cd build && cmake .. && make -j
# Run (ensure venv is active)
source venv/bin/activate
python3 python/main.py
#!/bin/sh
set -e

BUILD_DIR="build"

cmake -S . -B "$BUILD_DIR" \
    -G "Ninja" \
    -DCMAKE_BUILD_TYPE=Debug

cmake --build "$BUILD_DIR" --parallel

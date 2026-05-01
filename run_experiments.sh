#!/bin/bash

set -e

mkdir -p logs outputs

for cfg in $(find configs -name "*.yaml" | sort); do
  name=$(echo "$cfg" | sed 's|configs/||; s|\.yaml$||; s|/|_|g')

  echo "=============================="
  echo "Running $cfg"
  echo "Log: logs/${name}.log"
  echo "=============================="

  python train.py --config "$cfg" \
    2>&1 | tee "logs/${name}.log"
done
#!/bin/bash

mkdir -p logs outputs

for cfg in configs/*.yaml configs/*/*.yaml; do
  name=$(basename "$cfg" .yaml)

  echo "=============================="
  echo "Running $name"
  echo "=============================="

  python train.py --config "$cfg" \
    --run_name "$name" \
    2>&1 | tee "logs/${name}.log"

done
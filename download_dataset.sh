#!/bin/bash
# Download the dataset

set -euo pipefail
mkdir -p dataset
curl -L "https://cdn.intra.42.fr/document/document/33101/data.csv" -o dataset/data.csv


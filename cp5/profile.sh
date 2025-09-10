#!/bin/bash
set -eo pipefail

make -B cp
ncu --set full -k ssyrk_tn_128x128_padded_device -c 1 -o cp-%i.ncu-rep cp benchmarks/4b.txt
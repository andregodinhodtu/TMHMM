#!/bin/bash

for seqtype in GLOBULAR SIGNAL "SP+TM" TM; do
    echo "Going through $seqtype sequences"
    python3 align_hobohm.py -sq "$seqtype"
done

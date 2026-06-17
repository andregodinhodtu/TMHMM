#!/bin/bash

for seqtype in BETA GLOBULAR SIGNAL "SP+TM" TM; do
    echo "Going through $seqtype sequences"
    python3 align_hobohm.py -sq "$seqtype"
done

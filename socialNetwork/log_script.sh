#!/bin/bash

outfile="top_logs_$(date +%Y%m%d_%H%M%S).txt"
end=$((SECONDS+300))

while [ $SECONDS -lt $end ]; do
    top -b -n 1 >> "$outfile"
    sleep 10
done


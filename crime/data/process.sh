#!/usr/bin/env bash
column -t -s $'\t' -o ',' < crime.txt > crime.csv

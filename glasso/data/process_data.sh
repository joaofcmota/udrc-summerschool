#!/usr/bin/env bash

cut -d ',' -f2,4 dow_jones_index.data | tr -d '$' | tail -n+2 | sort > data_processed.csv

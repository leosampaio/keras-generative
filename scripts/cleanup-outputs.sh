#! /bin/bash

output_dir=$1
for experiment in $(ls $output_dir) 
do
    for epoch_ending in 1 2 3 4 5 6 7 8 9
    do
        rm -r $output_dir$experiment/weights/*$epoch_ending
    done
done
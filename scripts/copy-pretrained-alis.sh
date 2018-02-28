#! /bin/bash

d1=$1
d2=$2
destination=$3

mkdir -p $destination

cp $d1/f_D.hdf5 $destination/d1_f_D.hdf5
cp $d1/f_Gx.hdf5 $destination/d1_f_Gx.hdf5
cp $d1/f_Gz.hdf5 $destination/d1_f_Gz.hdf5

cp $d2/f_D.hdf5 $destination/d2_f_D.hdf5
cp $d2/f_Gx.hdf5 $destination/d2_f_Gx.hdf5
cp $d2/f_Gz.hdf5 $destination/d2_f_Gz.hdf5

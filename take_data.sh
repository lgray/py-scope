#!/bin/bash

config=$1
runname=$2
save_dir=$3

fname=${runname}.hdf5
outdir=${save_dir}/${runname}
outdir_plots=${outdir}/plots

mkdir -p ${outdir}
mkdir -p ${outdir_plots}

python3 data_plotter.py ${outdir} ${outdir_plots} >& data_plotter.log &

python3 py-scope.py ${config} ${fname} ${outdir}

trap "kill -9 `jobs -p`" EXIT

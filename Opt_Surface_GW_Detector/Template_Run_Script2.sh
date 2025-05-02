#!/bin/bash
# wn-205-10-22-03-a.cr.cnaf.infn.it
export TMPDIR='/tmp'
MYTEMPDIR=$(mktemp -p $TMPDIR -d IntegralXXXXXXX)


if [ ! -d $MYTEMPDIR ]; then
	echo "error creating directory"
	exit
fi

hostname
STARTINGDIR=$PWD
echo "Starting dir is $STARTINGDIR"

echo "Changing dir"
pwd
ls
cd $MYTEMPDIR
if [[ $PWD != $MYTEMPDIR ]] ;then
	echo "Error changing dir"
	exit
fi
LANDEDDIR=$PWD
echo "Landed dir is $LANDEDDIR"

echo "Setting trap for exit signal"
trap "echo 'exiting with trap and rm';rm -rf ${MYTEMPDIR};exit" EXIT

echo "Setting new pip cache"
mkdir ${LANDEDDIR}/.cache
export XDG_CACHE_HOME=${LANDEDDIR}/.cache

echo "OLD PATH: $PATH"
echo "activate environment"
#source /storage/gpfs_small_files/VIRGO/users/lucarei/env/cwb/bin/activate
#echo "PATH ALTERED BY CONDA: $PATH"
#export PATH=/storage/gpfs_small_files/VIRGO/users/lucarei/python/install/bin/:$PATH
#echo "NEW PATH: $PATH"
#which python3
#
## setting home and installing missing dep
#export HOME=$PWD
#pip3 install --user numpy scipy pyKriging pyswarms pyyaml


export OPENBLAS_NUM_THREADS=1
echo "Quadrants 0 0"
/storage/gpfs_small_files/VIRGO/users/fbadaracco/miniconda3/bin/python3  $STARTINGDIR/Script2.py 0 0 0 0
echo "python executed ... forse"


echo "python executed ... forse"
cp *.obj /storage/gpfs_small_files/VIRGO/users/fbadaracco/Reg_out10
echo "We are in dir: $PWD"

echo "THIS IS THE END"


#echo "cleaning $TMPDIR"
#rm -rf ${MYTEMPDIR}

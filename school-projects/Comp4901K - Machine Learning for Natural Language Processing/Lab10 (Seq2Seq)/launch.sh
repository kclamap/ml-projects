#!/bin/bash
#PBS -N lab10
#PBS -e ./error_log.txt
#PBS -o ./outptu_log.txt

cd ~/lab10
echo Start of calculation
python lab10_skeleton.py
echo End of calculation
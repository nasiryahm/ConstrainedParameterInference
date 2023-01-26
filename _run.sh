#!/bin/bash
pyth=YOURPYTHONPATHHERE

#export CUDA_VISIBLE_DEVICES=9

for s in 1 2 3 4 5;
do 
  #options="--dataset=CIFAR10 --seed=${s}"
  options="--dataset=MNIST --seed=${s}"
  
  ${pyth} run.py ${options} --config=copi_bp_boosted
  ${pyth} run.py ${options} --config=copi_fa_boosted
  ${pyth} run.py ${options} --config=bp_adam
  ${pyth} run.py ${options} --config=bp_decorr_boosted
  ${pyth} run.py ${options} --config=copi_bio_bp_boosted
  

done

##!/bin/bash
device=0
pyth=/YOURPYTHONPATHHERE/python

# MNIST
for s in 1 2 3 4 5;
do
  options="--device=${device} --seed=${s} --dataset=MNIST --nb_hid_layers=4 --nb_hid_units=500"
  lr="0.001"
  ${pyth} runner.py --algo=bp --momentum --learning_rate=${lr} ${options} # BP (Adam)
  lr="0.01"
  ${pyth} runner.py --algo=bp --decorr --learning_rate=${lr} ${options} # BP (decorr)
  ${pyth} runner.py --algo=copi --decorr --learning_rate=${lr} ${options} # COPI (SGD)
  ${pyth} runner.py --algo=copi --decorr --fa --learning_rate=${lr} ${options} # COPI (FA)
done

# CIFAR
for s in 1 2 3 4 5;
do
  options="--device=${device} --seed=${s} --dataset=CIFAR10 --nb_hid_layers=2 --nb_hid_units=1000"
  lr="0.0001"
  ${pyth} runner.py --algo=bp --momentum --learning_rate=${lr} ${options} # BP (Adam)
  lr="0.001"
  ${pyth} runner.py --algo=bp --decorr --learning_rate=${lr} ${options} # BP (decorr)
  ${pyth} runner.py --algo=copi --decorr --learning_rate=${lr} ${options} # COPI (SGD)
  ${pyth} runner.py --algo=copi --decorr --fa --learning_rate=${lr} ${options} # COPI (FA)
done

cd results
${pyth} PlotPerformanceCurves.py --MNIST
${pyth} PlotPerformanceCurves.py --CIFAR10


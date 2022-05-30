# COPI

This code accompanies the submission titled **Constrained Parameter Inference as a Principle for Learning**.

Enclosed are a few key files/folders, below find brief descriptions:
- *pip_requirements.txt*/*conda_requirements.txt*: The set of packages required for a python/conda environment to execute the models enclosed
- *runner.py*: A python script which loads and executes a feed-forward, multi-layer, dnn model with a number of optional arguments to specify training/network etc
- *_generate_results.sh*: A bash-script capable of producing all data and plots required to verify the results shown in, Figure 2. This can be examined to see examples of possible commandline options for _runner.py_
- *copi/*: A folder containing a module of code python for the COPI and BP trained deep neural networks
- *results/*: A folder into-which results and network architectures are automatically placed following the completion of simulations
- *results/PlotPerformanceCurves.py*: A python script (automatically executed by *_generate_results.sh* which produces a range of plots of performance for either the MNIST or CIFAR datasets (depending upon provided commandline option)
- *results/_plots/*: A folder into-which all performance plots are automatically stored
- *results/RFandNetworkCompression.ipynb*: An ipython notebook which takes stored, trained networks and reproduces all subplots of Figure 3.


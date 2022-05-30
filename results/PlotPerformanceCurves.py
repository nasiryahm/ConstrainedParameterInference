import json
import numpy as np
import matplotlib.pyplot as plt
import getopt

# Setting up
dataset = 'MNIST'
ylims = [0.75, 1.02]
# Setting up the options for simulation
opts, remaining = getopt.getopt(
    sys.argv[1:],
    '',
    ['dataset='])
for opt, arg in opts:
    if opt == '--dataset':
        dataset = str(arg)
assert dataset == 'MNIST' or dataset =='CIFAR10'

if dataset == 'MNIST':
  ylims = [0.75, 1.02]
outpath = './' + dataset + '/'

sim_paths = ['bp_adam/', 'bp_decorr/', 'copi_decorr/', 'copi_decorr_fa/']
labels_paths = ['bp (adam)', 'bp (decorr)', 'copi (bp)', 'copi (fa)']
seeds = [1, 2, 3, 4, 5]
nb_epochs = 100
tick_interval = 10
set_to_plot = ['train', 'test']
measures = ['acc', 'loss']
me_labels = ['Accuracy', 'Loss']

data = [{m: {s: [] for s in set_to_plot} for m in measures} for s in sim_paths]
for indx, sp in enumerate(sim_paths):
  path = outpath + sp
  for seed in seeds:
    s_path = path + str(seed) + '/'
    with open(s_path + '_measures.json') as json_file:
       d = json.load(json_file)
       for me in measures:
         for se in set_to_plot:
           data[indx][me][se].append(d[me][se])

cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig = plt.figure(figsize=(5*len(measures),3), dpi=200)
for j, me in enumerate(measures):  
  plt.subplot(1,len(measures),j+1)
  ax = plt.gca()
  plt.title(dataset)
  plt.ylabel(me_labels[j])
  plt.xlabel('Epochs')
  for i, se in enumerate(set_to_plot):
    style = '-'
    alpha = 1.0
    if se == 'train':
      alpha = 0.25

    for indx, d in enumerate(data):
      proc_data = np.asarray(d[me][se])
      # The first index is assumed to be the adam
      if indx > 0:
        std = np.std(proc_data, axis=0); meanval = np.mean(proc_data, axis=0)
        plt.plot(meanval, ls=style, color=cycle[indx], alpha=alpha)
        ax.fill_between(range(len(data[0][me][se][0])), meanval - std, meanval + std, alpha=0.1, color=cycle[indx])
      else:
        proc_data = proc_data[:,::12]
        meanval = np.mean(proc_data, axis=0)
        if me == 'acc':
          val = np.max(meanval)
        else:
          val = np.min(meanval)
        plt.scatter((12*len(meanval)), (val), marker='*', color=cycle[indx], alpha=alpha)
    if me == 'acc':
      plt.ylim(ylims)


    plt.xticks(np.arange(len(data[0][me][se][0]) + 1,step=tick_interval*(int(len(data[0][me][se][0])/nb_epochs))), tick_interval*np.arange(nb_epochs/tick_interval + 1).astype(int))
    plt.xlabel('Epochs')

plt.show()
fig.savefig('./_plots/' + dataset + '.png', bbox_inches="tight")
plt.clf()

fig = plt.figure(figsize=(5*len(measures),3), dpi=200)
for indx in range(len(data)):
  plt.plot([0],[0], color=cycle[indx], label=labels_paths[indx])
plt.legend()
fig.savefig('./_plots/color_legend.png', bbox_inches="tight")
plt.clf()

fig = plt.figure(figsize=(5*len(measures),3), dpi=200)
plt.plot([0],[0], color='k', ls='-', label='train', alpha=0.25)
plt.plot([0],[0], color='k', ls='-', label='test')
plt.legend()
fig.savefig('./_plots/tt_legend.png', bbox_inches="tight")
plt.clf()

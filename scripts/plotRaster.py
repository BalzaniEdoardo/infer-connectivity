import matplotlib as mlp
import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys

#def main(start,end):
mlp.style.use('classic')
inputs = sys.argv[1:]
start = 2500
end =3000
NE = 300
NI = 100
f, axarr = plt.subplots(1, figsize=(12,3))

spikes = pickle.load(open('spikes1.pckl', 'rb'))

xsr = []
ysr = []
for i in range(NE+NI):
  spikesForNeur = spikes[i]
  for time in spikesForNeur:
    if time >= start and time <= end:
      xsr.append(time/1000)
      ysr.append(i)
colors = np.where(np.asarray(ysr)<NE, 'r', 'b')
axarr.scatter(xsr, ysr, color = colors, s=3)
#  axarr.set_ylabel('Neuron #', fontsize = 15)
axarr.set_ylim(0,400)
axarr.set_xlim(start/1000,end/1000)
axarr.set_xlabel('seconds', fontsize=15)
axarr.set_ylabel('neuron number', fontsize=15)
axarr.tick_params(axis='both', labelsize = 15)
plt.savefig('Raster.pdf', format = 'pdf', bbox_inches = 'tight')
plt.show()
 
#  binsize = 5

#  xsE = []
#  xsI = []
#  for i in range(400):
#    neurSpikes = spikes[i]
#    if i <300:
#      xsE += neurSpikes
#    else:
#      xsI += neurSpikes

#  (nE, binsE, patchesE) = ax.hist(xsE,np.arange(2000,3000+binsize,binsize))
#  percE = [ z/3 for z in nE ]
#  axarr[1].bar(binsE[:len(binsE)-1], percE, binsize, color='r')
#  axarr[1].set_ylabel('% of E neurons', fontsize = 15)
#  axarr[1].set_ylim(0,30)
#  axarr[1].tick_params(axis='y', labelsize=15)

#  (nI, binsI, patchesI)= ax.hist(xsI,np.arange(2000,3000+binsize,binsize))
#  percI = [ z for z in nI ]
#  axarr[2].bar(binsI[:len(binsI)-1], percI, binsize, color='b')
#  axarr[2].set_ylim(0,100)
#  axarr[2].set_xlabel('ms', fontsize = 15)
#  axarr[2].set_ylabel('% of I neurons', fontsize = 15)
#  axarr[2].tick_params(axis='both', labelsize=15)
#  axarr[2].set_xlim(2000,3000)
 
#  f.subplots_adjust(hspace=.12) 
 

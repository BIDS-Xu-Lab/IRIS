import numpy as np
from scipy.special import rel_entr
import os

from .Utils import layout

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

import sys
#sys.stdout = Unbuffered(sys.stdout)

def cost(time, zeta, rho, bins=100):
	p=np.array([(2.*zeta/bins + (((x+1.)/bins)**2-(x/bins)**2)*(1-zeta))/(1.+zeta) for x in range(bins)])
	q = np.histogram(zeta+(1-zeta)*np.power(time, np.exp(rho)), bins=100, density=True)[0]+1
	return np.sum(rel_entr(p, q/np.sum(q)))

def get_rho(t, zeta=0.1, bins=100):
	tests = np.arange(-5, 5, 0.01)
	t = (t - np.min(t)) / (np.max(t) - np.min(t))
	return tests[np.argmin([cost(t, zeta, x, bins) for x in tests])]

def logit(x):
	return np.log(x / (1 - x))

def fit_transform(data, time, **kwargs):
	if time.shape[0] != data.shape[0]:
		raise ValueError("Time array must have the same number of rows as data.")
	
	data = data.astype(np.float32)
	kwargs['time'] = time.astype(np.float32)
	
	if 'sample_time' in kwargs:
		# is sample_time a scalar?
		if isinstance(kwargs['sample_time'], (int, float)):
			kwargs['time'] += np.random.uniform(0, kwargs['sample_time'], size=kwargs['time'].shape[0])
		elif kwargs['sample_time'] == 'hetero':
			time2range = {}
			uniq = np.unique(kwargs['time'])
			avg_range = 0
			for i, time in enumerate(uniq):
				time2range[time]=(uniq[i+1]-time) if (i < len(uniq)-1) else avg_range / (len(uniq)-1)
				avg_range += time2range[time]
			kwargs['time'] += np.array([np.random.uniform(0, time2range[time]) for time in kwargs['time']])
		del kwargs['sample_time']
	#normalize time to [0, 1]
	kwargs['time'] = (kwargs['time'] - np.min(kwargs['time'])) / (np.max(kwargs['time']) - np.min(kwargs['time']))
	
	if 'zeta' not in kwargs:
		kwargs['zeta'] = 0.1
	if 'alpha' not in kwargs:
		kwargs['alpha'] = .1
	if 'beta' not in kwargs:
		kwargs['beta'] = 0.95
	if 'gamma' not in kwargs:
		kwargs['gamma'] = 128
	if 'n_neighbors' not in kwargs:
		kwargs['n_neighbors'] = 32
	if 'n_trees' not in kwargs:
		kwargs['n_trees'] = 32
	if 'n_propagations' not in kwargs:
		kwargs['n_propagations'] = 3
	if 'n_negatives' not in kwargs:
		kwargs['n_negatives'] = 5
	if 'normalize' not in kwargs:
		kwargs['normalize'] = False
	if 'n_iterations' not in kwargs:
		kwargs['n_iterations'] = int(data.shape[0]/100)

	if 'rho' not in kwargs and 'time' in kwargs:
		kwargs['rho'] = get_rho(kwargs['time'])
		print("Optimal rho: %.2f" % kwargs['rho'])

	return_polar = False
	if 'return_polar' in kwargs and kwargs['return_polar']:
		return_polar = True
		del kwargs['return_polar']

	polar = layout(data, **kwargs)

	if polar is None:
		return None
	
	# delete annoy_index_file if it exists
	if os.path.exists('annoy_index_file'):
		os.remove('annoy_index_file')

	if return_polar:
		return polar
	else:
		# convert to cartesian coordinates and scale to [-100, 100]
		cartesian = np.zeros_like(polar)
		cartesian[:, 0] = polar[:, 0] * np.cos(polar[:, 1])
		cartesian[:, 1] = polar[:, 0] * np.sin(polar[:, 1])
		return cartesian * 100
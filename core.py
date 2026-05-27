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
	"""
	Find the optimal rho value for the given time points.

	Parameters
	----------
	t : array-like
		The time points to find the optimal rho value for. Should be a 1D numpy array with shape (n_samples,).
	zeta : float, optional
		The ratio of inner diameter to outer diameter. Should be in [0, 1]. Defaults to 0.1.
	bins : int, optional
		The number of bins to use for computing KL divergence. Defaults to 100.

	Returns
	-------
	rho : float
		The optimal rho value for the given time points.
	"""

	tests = np.arange(-5, 5, 0.01)
	t = (t - np.min(t)) / (np.max(t) - np.min(t))
	return tests[np.argmin([cost(t, zeta, x, bins) for x in tests])]

def logit(x):
	return np.log(x / (1 - x))

def fit_transform(data, time, **kwargs):
    """
    Perform time-structure manifold projection.

    Parameters
    ----------
    data : array-like
        The high-dimensional data points to project. Should be a 2D numpy array with shape (n_samples, n_features).
    time : array-like
        Timestamps for each sample. Should be a 1D numpy array with shape (n_samples,).
    **kwargs :
		n_iterations : int, optional
			The number of stochastic gradient descent steps to perform, in millions. Defaults to `n_samples // 100`.
        sample_time : float, str, optional
            If a scalar, resample each time point `t_i` uniformly within `[t_i, t_i + sample_time).
            If 'hetero', resample each time point `t_i` uniformly within `[t_i, t_i + (t_i+1 - t_i) / 2].
			If None (default), no resampling is performed.
		return_polar : bool, optional
			If True, return the layout in polar coordinates (radius, angle). Defaults to False.
        zeta : float, optional
            The ratio of inner diameter to outer diameter. Should be in [0, 1]. Defaults to 0.1.
		rho : float, optional
			The exponential parameter for computing radii from [0, 1]-normalized time values, with 0 being direct mapping. Defaults to the optimal value for the given time points. Use values below 0 for left-skewed distributions and values above 0 for right-skewed distributions. Optimal values typically lie within [-4, 4].
        alpha : float, optional
            The learning rate. Should be in [0, 1]. Defaults to 0.1.
        beta : float, optional
            The weight of the polar component of loss. Should be in [0, 1]. Defaults to 0.95. Higher values allow less overloading of classes within different time ranges of the same sector, resulting in tighter, more radial clusters.
        gamma : int, optional
            The weights assigned to negative edges. Defaults to 128. Higher values assign more weight to negative edges, resulting in more repulsion between points.
        n_neighbors : int, optional
            The number of neighbors to consider for each point. Defaults to 32.
        n_trees : int, optional
            The number of trees to build for the Annoy index. Defaults to 32.
        n_propagations : int, optional
            The number of propagations to perform. Defaults to 3.
        n_negatives : int, optional
            The number of negative samples to use for each positive sample. Defaults to 5.
        normalize : bool, optional
            Whether to normalize the high-dimensional data. Defaults to False.

    Returns
    -------
    layout : ndarray
        The layout of the data points, shape (n_samples, 2). If return_polar is True, the layout is in polar coordinates (radius, angle). Otherwise, the layout is in Cartesian coordinates (x, y).
    """
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
            for i, uniq_time in enumerate(uniq):
                time2range[uniq_time] = (uniq[i+1] - uniq_time) if (i < len(uniq)-1) else avg_range / (len(uniq)-1)
                avg_range += time2range[uniq_time]
            kwargs['time'] += np.array([np.random.uniform(0, time2range[t]) for t in kwargs['time']])
        del kwargs['sample_time']

    # normalize time to [0, 1]
    kwargs['time'] = (kwargs['time'] - np.min(kwargs['time'])) / (np.max(kwargs['time']) - np.min(kwargs['time']))

    if 'zeta' not in kwargs:
        kwargs['zeta'] = 0.1
    if 'alpha' not in kwargs:
        kwargs['alpha'] = 0.1
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
        kwargs['n_iterations'] = int(data.shape[0] / 100)

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
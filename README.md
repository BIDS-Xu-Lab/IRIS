# IRIS: Time-structured Manifold Projections

IRIS performs nonlinear dimension reduction (similar to UMAP, t-SNE, or LargeVis), but
incorporates timestamps of data points to stucture the layout, with earlier points near
the center and later points near the perimeter.

    fit_transform(data, time, **kwargs)
        Perform time-structured manifold projection.
        
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

    get_rho(t, zeta=0.1, bins=100)
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

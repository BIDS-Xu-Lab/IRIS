// Based on https://raw.githubusercontent.com/lferry007/LargeVis/refs/heads/master/Linux/LargeVismodule.cpp
// Modified by Brian Ondov on 2026-05-21

#include "Python.h"
#include "LargeVis.h"
#include "numpy/arrayobject.h" 

static PyObject *layout(PyObject *self, PyObject *args, PyObject *kwargs)
{
	LargeVis model(8);

	PyObject *v, *t; v = t = NULL;
	long long n_vertices;
	long long n_dim;

	// kwargs
	long long out_dim = -1;
	long long n_samples = -1;
	long long n_threads = -1;
	long long n_negatives = -1;
	long long n_neighbors = -1;
	long long n_trees = -1;
	long long n_propagation = -1;
	real alpha, beta, gamma, perplexity, rho, zeta;
	double alpha_f = -1, beta_f = -1, gamma_f = -1, perplexity_f = -1, rho_f = 0, zeta_f = -2;
	bool normalize = true;

	static char* kwlist[] = {"data", "time", "out_dim", "n_threads", "n_iterations", "n_trees", "n_propagations", "alpha", "beta", "rho", "zeta", "n_negatives", "n_neighbors", "gamma", "perplexity", "normalize", NULL};  // Keyword argument names
	
	//Py_Initialize();
	//PyEval_InitThreads();
	
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OLLLLLddddLLddb", kwlist, &v, &t, &out_dim, &n_threads, &n_samples, &n_trees, &n_propagation, &alpha_f, &beta_f, &rho_f, &zeta_f, &n_negatives, &n_neighbors, &gamma_f, &perplexity_f, &normalize))
	{
		printf("Input error!\n");
		return Py_None;  // Error occurred
	}

	alpha = alpha_f;
	beta = beta_f;
	gamma = gamma_f;
	perplexity = perplexity_f;
	rho = rho_f;
	zeta = zeta_f;

	PyArrayObject* array = (PyArrayObject*)PyArray_FROM_OTF(v, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);
	if (array == NULL)
	{
		printf("ERROR: Could not interpret data as NumPy array.\n");
		return Py_None; // Error handling
	}

	PyArrayObject* arraytime = NULL;

	n_vertices = (int)PyArray_DIM(array, 0);
	n_dim = (int)PyArray_DIM(array, 1);
	if ( PyArray_NDIM(array) != 2) 
	{
		printf("ERROR: Input data must have 2 dimensions (got %lu).\n", PyArray_NDIM(array));
		return Py_None; // Error handling
	}
	//PySys_WriteStdout("Received data of shape (%lu, %lu)\n", n_vertices, n_dim);
	Py_INCREF(array);
	real* data = (real*)PyArray_DATA(array);
	//PySys_WriteStdout("data: %f\t%f\t%f...\n", data[0], data[1], data[2]);

	real* time = NULL;
	if (t != NULL)
	{
		arraytime = (PyArrayObject*)PyArray_FROM_OTF(t, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);

		if (arraytime == NULL)
		{
			printf("ERROR: Could not interpret time as NumPy array.\n");
			return Py_None; // Error handling
		}

		if ((int)PyArray_DIM(arraytime, 0) != n_vertices)
		{
			printf("ERROR: Time array must have the same number of rows as data.\n");
			return Py_None; // Error handling
		}

		Py_INCREF(arraytime);
		time = (real*)PyArray_DATA(arraytime);
		//PySys_WriteStdout("time: %f\t%f\t%f...\n", time[0], time[1], time[2]);
	}
	//PySys_WriteStdout("trees: %lld\n", n_trees);
	bool res = false;
	model.load_from_data(data, n_vertices, n_dim, normalize, time);
	Py_BEGIN_ALLOW_THREADS
	
	res = model.run(out_dim, n_threads, n_samples, n_trees, n_propagation, alpha, beta, rho, zeta, n_negatives, n_neighbors, gamma, perplexity);
	Py_END_ALLOW_THREADS
	Py_DECREF(array);
	if (arraytime != NULL)
	{
		Py_DECREF(arraytime);
	}
	PyObject* ret = Py_None;
	if (res)
	{
	    npy_intp dims[2] = {n_vertices, model.get_out_dim()};
    	ret = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, model.get_ans());
	}
	//Py_Finalize();
	return ret;
}

static PyMethodDef module_methods[] = {
	{ "layout", (PyCFunction)layout, METH_VARARGS | METH_KEYWORDS, "layout(numpy array)\nFit & transform array" },
	{ NULL, NULL, 0, NULL }
};

static struct PyModuleDef Utils =
{
    PyModuleDef_HEAD_INIT,
    "IRIS.Utils", /* name of module */
    NULL,
    -1,   /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    module_methods
};

PyMODINIT_FUNC PyInit_Utils(void)
{
	import_array();
    return PyModule_Create(&Utils);
}
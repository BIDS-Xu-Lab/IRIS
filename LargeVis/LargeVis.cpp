#include "LargeVis.h"
#include <map>
#include <float.h>
#include <queue>
#include <utility>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <thread>
#include <time.h>

using std::pair;
using std::vector;

#include "Python.h"

#if PYTHON
/* Must hold the GIL: layout() releases it around model.run(); logout/logoutp are
 * also used from pthread workers. Route through sys.stdout so Jupyter sees output. */
static inline void iris_flush_stdout(void)
{
	PyObject *out = PySys_GetObject("stdout");
	if (out == NULL || out == Py_None)
		return;
	PyObject *res = PyObject_CallMethod(out, "flush", NULL);
	Py_XDECREF(res);
	if (PyErr_Occurred())
		PyErr_Clear();
}

#define logout(...) do { \
	PyGILState_STATE _gil_log = PyGILState_Ensure(); \
	PySys_WriteStdout(__VA_ARGS__); \
	iris_flush_stdout(); \
	PyGILState_Release(_gil_log); \
} while (0)

#define logoutp(...) do { \
	PyGILState_STATE _gil_log = PyGILState_Ensure(); \
	PySys_WriteStdout(__VA_ARGS__); \
	iris_flush_stdout(); \
	PyGILState_Release(_gil_log); \
} while (0)
#else
	#define logout(...) {printf(__VA_ARGS__);fflush(stdout);}
	#define logoutp(...) {printf(__VA_ARGS__);fflush(stdout);}
#endif

struct args {
    int joined;
    pthread_t td;
    pthread_mutex_t mtx;
    pthread_cond_t cond;
    void **res;
};

static void *waiter(void *ap)
{
    struct args *args = (struct args *)ap;
    pthread_join(args->td, args->res);
    pthread_mutex_lock(&args->mtx);
    args->joined = 1;
    pthread_mutex_unlock(&args->mtx);
    pthread_cond_signal(&args->cond);
    return 0;
}

int pthread_timedjoin_np(pthread_t td, void **res, struct timespec *ts)
{
    pthread_t tmp;
    int ret;
    struct args args = { .td = td, .res = res };

    pthread_mutex_init(&args.mtx, 0);
    pthread_cond_init(&args.cond, 0);
    pthread_mutex_lock(&args.mtx);
	//printf("Waiting until %d %d\n", ts->tv_sec, ts->tv_nsec);
    ret = pthread_create(&tmp, 0, waiter, &args);
    if (!ret)
        do ret = pthread_cond_timedwait(&args.cond, &args.mtx, ts);
        while (!args.joined && ret != ETIMEDOUT);
	//if (args.joined) {printf("Joined\n");pthread_cancel(td);}
	//printf("Reached       %d %d\n", ts->tv_sec, ts->tv_nsec);

    pthread_mutex_unlock(&args.mtx);

    pthread_cancel(tmp);
    pthread_join(tmp, 0);

    pthread_cond_destroy(&args.cond);
    pthread_mutex_destroy(&args.mtx);

    return args.joined ? 0 : ret;
}

void LargeVis::wait_for_threads(pthread_t *pt, int n_threads)
{
	bool done = false;
	int result;
	struct timespec ts;

	while (!done)
	{
#if PYTHON
//Py_BEGIN_ALLOW_THREADS
PyGILState_STATE gstate;
gstate = PyGILState_Ensure();
		if (PyErr_CheckSignals() != 0)
		{
			printf("Interrupted\n");
			interrupt = true;
		}
		PyGILState_Release(gstate);
//Py_END_ALLOW_THREADS
#endif
		done = true;
		for (int j = 0; j < n_threads; ++j)
		{
			if (pt[j] == 0) continue;
			if (clock_gettime(CLOCK_REALTIME, &ts) == -1) {
				printf("Error getting time\n");
				exit(1);
			}
			ts.tv_nsec += 10000000;
			if (ts.tv_nsec > 1000000000) {
				ts.tv_sec += 1;
				ts.tv_nsec -= 1000000000;
			}
			result = pthread_timedjoin_np(pt[j], nullptr, &ts);
			//result = pthread_join(pt[j], NULL);
			if (result == 0) {
				pt[j] = 0;
			} else if (result == ETIMEDOUT) {
				done = false;
			}
		}
	}
	delete[] pt;
}

const real pi = 3.1415926;
const real twopi = 2 * 3.1415926;

LargeVis::LargeVis(long long n_thre)
{
	vec = time = vis = prob = NULL;
	knn_vec = old_knn_vec = NULL;
	annoy_index = NULL;
	head = alias = NULL;
    neg_table = NULL;
    out_dim = -1;
    n_old_vertices = -1;
    resume = false;
	update_old_points = false;
	n_threads = n_thre < 0 ? 8 : n_thre;
}

void LargeVis::clean_model()
{
	//if (vis) {delete[] vis; vis = 0;}
	if (prob) delete[] prob;
	if (knn_vec) delete[] knn_vec;
	if (old_knn_vec) delete[] old_knn_vec;
	//if (faiss_index) faiss_Index_free(faiss_index);
	if (neg_table) delete[] neg_table;
	if (alias) delete[] alias;
	prob = NULL;
	knn_vec = old_knn_vec = NULL;
	//faiss_index = NULL;
    neg_table = NULL;
    alias = NULL;

	edge_count_actual = 0;
	neg_size = 1e8;
}

void LargeVis::clean_graph()
{
	if (head) { delete[] head; head = NULL; }

	n_edge = 0;
	next.clear(); edge_from.clear(); edge_to.clear(); reverse.clear(); edge_weight.clear(); names.clear();
}

void LargeVis::clean_data()
{
	if (vec) { delete[] vec; vec = NULL; }
	if (time) { delete[] time; time = NULL; }
	clean_graph();
}

void LargeVis::load_from_file(char *infile, char *timefile, bool normalize, bool append, long long n_init_neighbors)
{
    if (n_init_neighbors == -1) n_init_neighbors = 150;
	n_neighbors = n_init_neighbors;
	if (!append) clean_data();
	FILE *fin = fopen(infile, "rb");
	if (fin == NULL)
	{
		printf("\nFile not found! %s\n", infile);
		return;
	}
	printf("Reading input file %s ......", infile); fflush(stdout);
    long long n_vertices_new, n_dim_new;
	fscanf(fin, "%lld%lld", &n_vertices_new, &n_dim_new);
	if (append)
	{
        if (n_dim != n_dim_new)
        {
            printf("\nERROR: dim of %s (%lld) does not match existing points (%lld).\n", infile, n_dim_new, n_dim);
            interrupt = true;return;
        }
	    n_old_vertices = n_vertices;
	    n_vertices += n_vertices_new;
    	real *vec_new = new real[n_vertices * n_dim];
    	memcpy(vec_new, vec, sizeof(real)*n_old_vertices*n_dim);
    	delete[] vec;
    	vec = vec_new;
	}
	else
	{
	    n_vertices = n_vertices_new;
	    n_dim = n_dim_new;
    	vec = new real[n_vertices * n_dim];
    }
	for (long long i = append ? n_old_vertices : 0; i < n_vertices; ++i)
	{
		for (long long j = 0; j < n_dim; ++j)
		{
			fscanf(fin, "%f", &vec[i * n_dim + j]);
		}
	}
	fclose(fin);
	//printf(" Done.\n");
	//printf("Total vertices : %lld\tDimension : %lld\n", n_vertices, n_dim);

	if (timefile != NULL)
	{
		fin = fopen(timefile, "rb");
		if (fin == NULL)
		{
			printf("\nFile not found! %s\n", timefile);
			return;
		}
		printf("Reading time file %s ......", timefile); fflush(stdout);
		if (append)
		{
			real *time_new = new real[n_vertices];
			memcpy(time_new, time, sizeof(real)*n_old_vertices);
			delete[] time;
			time = time_new;
		}
		else
		{
			time = new real[n_vertices];
		}
		for (long long i = append ? n_old_vertices : 0; i < n_vertices; ++i)
		{
			fscanf(fin, "%f", &time[i]);
		}
		fclose(fin);
		printf(" Done.\n");
	}
	if (normalize) this->normalize();
	if (append && resume) init_by_nn();
}

void LargeVis::init_by_nn()
{
    ensure_index();
	if (interrupt) return;
    run_ann();
	logout("Initializing %lld new points ......",n_vertices-n_old_vertices);
	real *weights = new real[n_neighbors];

	for (long long i = n_old_vertices; i < n_vertices; i++)
	{
		//printf("Point %d: %d neighbors\n", i, knn_vec[i].size());
		real sum = 0;
		for (int j = 0; j < knn_vec[i].size(); ++j)
		{
			weights[j] = 1./CalcDist(i, knn_vec[i][j]);
			sum += weights[j];
		}
		for (int j = 0; j < knn_vec[i].size(); ++j)
		{
			weights[j] /= sum;
		}
		for (int j = 0; j < out_dim; ++j)
		{
			vis[i*out_dim+j] = 0;
		}
		for (int j = 0; j < knn_vec[i].size(); ++j)
		{
			for (int k = 0; k < out_dim; ++k)
			{
				vis[i*out_dim+k] += vis[knn_vec[i][j]*out_dim+k]*weights[j];
			}
		}
	}
	printf("Done.\n");
	delete[] weights;
}

char peek(FILE *f) // TODO: remove
{
    char c = getc(f);
    fseek(f, -1, SEEK_CUR);
    return c;
}

void LargeVis::init_from_file(char *infile)
{
    resume = true;
	FILE *fin = fopen(infile, "rb");
	if (fin == NULL)
	{
		printf("\nFile not found!\n");
		return;
	}
	printf("Initializing from file %s ......", infile); fflush(stdout);
    long long n_vertices_new;
	fscanf(fin, "%lld%lld", &n_vertices_new, &out_dim);
	fseek(fin, 1, SEEK_CUR);
	if (n_vertices_new != n_vertices)
	{
	    printf("\nERROR: %s has different number of vertices (%lld) than feature file (%lld).\n", infile, n_vertices_new, n_vertices);
	    exit(1);
	}
	vis = new real[n_vertices * out_dim];
	update.resize(n_vertices, 0);
	for (long long i = 0; i < n_vertices; ++i)
	{
		while (peek(fin) == '\n')
		{
		    fseek(fin, 1, SEEK_CUR);
		    //printf("New vertex: %lld\n", i);
		    update[i] = true;
		    ++i;
		}
		for (long long j = 0; j < out_dim; ++j)
		{
			fscanf(fin, "%f", &vis[i * out_dim + j]);
		}
		fseek(fin, 1, SEEK_CUR);
	}
	fclose(fin);
	printf(" Done.\n");
/*	for (int i = 0; i < n_vertices; ++i)
	{
	    bool b = update[i];
	    printf("%d\n", b);
	}*/
	printf("Total vertices : %lld\tDimension : %lld\n", n_vertices, n_dim);
}

void LargeVis::load_from_data(real *data, long long n_vert, long long n_di, bool normalize=true, real *time_)
{
	clean_data();
	vec = data;
	time = time_;
	n_vertices = n_vert;
	n_dim = n_di;
	//logout("Total vertices : %lld\tDimension : %lld\n", n_vertices, n_dim);
	if (normalize) this->normalize();
}

void LargeVis::load_from_index(char *infile)
{
    printf("ERROR: load from index not implemented\n");
    exit(1);
}

void LargeVis::load_from_graph(char *infile)
{
	clean_data();
	char *w1 = new char[1000];
	char *w2 = new char[10000];
	long long x, y, i, p;
	real weight;
	std::map<std::string, long long> dict;
	n_vertices = 0;
	FILE *fin = fopen(infile, "rb");
	if (fin == NULL)
	{
		printf("\nFile not found!\n");
		return;
	}
	printf("Reading input file %s ......%c", infile, 13);
	while (fscanf(fin, "%s%s%f", w1, w2, &weight) == 3)
	{
		if (!dict.count(w1)) { dict[w1] = n_vertices++; names.push_back(w1); }
		if (!dict.count(w2)) { dict[w2] = n_vertices++; names.push_back(w2); }
		x = dict[w1];
		y = dict[w2];
		edge_from.push_back(x);
		edge_to.push_back(y);
		edge_weight.push_back(weight);
		next.push_back(-1);
		++n_edge;
		if (n_edge % 5000 == 0)
		{
			logout("Reading input file %s ...... %lldK edges%c", infile, n_edge / 1000, 13);
		}
	}
	fclose(fin);
	delete[] w1;
	delete[] w2;

	head = new long long[n_vertices];
	for (i = 0; i < n_vertices; ++i) head[i] = -1;
	for (p = 0; p < n_edge; ++p)
	{
		x = edge_from[p];
		next[p] = head[x];
		head[x] = p;
	}
	printf("\nTotal vertices : %lld\tTotal edges : %lld\n", n_vertices, n_edge);
}

void LargeVis::saveindex(char *outfile)
{
    //faiss::write_index(faiss_index, outfile);
}

void LargeVis::save(char *outfile)
{
	FILE *fout = fopen(outfile, "wb");
	fprintf(fout, "%lld %lld\n", n_vertices, out_dim);
	for (long long i = 0; i < n_vertices; ++i)
	{
		if (names.size()) fprintf(fout, "%s ", names[i].c_str());
		for (long long j = 0; j < out_dim; ++j)
		{
			if (j) fprintf(fout, " ");
			fprintf(fout, "%.6f", vis[i * out_dim + j]);
		}
		fprintf(fout, "\n");
	}
	fclose(fout);
}

real *LargeVis::get_ans()
{
	return vis;
}

long long LargeVis::get_n_vertices()
{
	return n_vertices;
}

long long LargeVis::get_out_dim()
{
	return out_dim;
}

void LargeVis::normalize()
{
	logout("Normalizing ......");
	real *mean = new real[n_dim];
	for (long long i = 0; i < n_dim; ++i) mean[i] = 0;
	for (long long ll = 0; ll < n_vertices * n_dim; ll += n_dim)
	{
		for (long long j = 0; j < n_dim; ++j)
			mean[j] += vec[ll + j];
	}
	for (long long j = 0; j < n_dim; ++j)
		mean[j] /= n_vertices;
	real *var = new real[n_dim];
	for (long long ll = 0; ll < n_vertices * n_dim; ll += n_dim)
	{
		for (long long j = 0; j < n_dim; ++j)
		{
			vec[ll + j] -= mean[j];
			var[j] += vec[ll + j] * vec[ll + j];
		}
	}
	
	for (long long i = 0; i < n_dim; ++i) var[i] = n_vertices / var[i]; // invert variance for mult later
	for (long long ll = 0; ll < n_vertices * n_dim; ll += n_dim)
	{
		for (long long j = 0; j < n_dim; ++j) vec[ll + j] *= var[j]; // variance inverse
	}
	delete[] mean;
	delete[] var;
	
	if (time)
	{
		real time_min=time[0], time_max=time[0];
		for (long long i = 0; i < n_vertices; ++i)
		{
			if (time[i] < time_min) time_min = time[i];
			if (time[i] > time_max) time_max = time[i];
		}
		real time_scale = 1./(time_max-time_min+1e-10);
		for (long long i = 0; i < n_vertices; ++i)
		{
			time[i] = pow((time[i] - time_min) * time_scale + 0.001, 2);
		}
	}
	logout(" Done.\n");
}

real LargeVis::CalcDist(long long x, long long y)
{
	real ret = 0;
	long long i, lx = x * n_dim, ly = y * n_dim;
	for (i = 0; i < n_dim; ++i)
		ret += (vec[lx + i] - vec[ly + i]) * (vec[lx + i] - vec[ly + i]);
	return ret;
}

void LargeVis::init_alias_table()
{
    // Init table for implementing the alias method of sampling a categorical
    // distribution (see https://en.wikipedia.org/wiki/Alias_method)
    
	alias = new long long[n_edge];
	prob = new real[n_edge];

	real *norm_prob = new real[n_edge];
	long long *large_block = new long long[n_edge];
	long long *small_block = new long long[n_edge];

	real sum = 0;
	long long cur_small_block, cur_large_block;
	long long num_small_block = 0, num_large_block = 0;

    // First, bin edges by whether they are below ("small") or above
    // ("large") the average edge size.
    //
	for (long long k = 0; k < n_edge; ++k) sum += edge_weight[k];
	for (long long k = 0; k < n_edge; ++k) norm_prob[k] = edge_weight[k] * n_edge / sum;
    //
	for (long long k = n_edge - 1; k >= 0; --k)
	{
		if (norm_prob[k] < 1)
			small_block[num_small_block++] = k;
		else
			large_block[num_large_block++] = k;
	}

    // divide edge weight mass into n_edge bins, each with mass from at most 2 edges
    //
	while (num_small_block && num_large_block)
	{
		cur_small_block = small_block[--num_small_block]; // pop a small edge
		cur_large_block = large_block[--num_large_block]; // pop a large edge
		
		// start the bin with a small edge
		prob[cur_small_block] = norm_prob[cur_small_block];
		
		// fill the rest of the bin with the large edge
		alias[cur_small_block] = cur_large_block;
		
		// compute leftover of large edge that didn't fit in bin
		norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1;
		
		// put the leftover of the large edge back in the appropriate bin
		// (might be small now)
		//
		if (norm_prob[cur_large_block] < 1)
			small_block[num_small_block++] = cur_large_block;
		else
			large_block[num_large_block++] = cur_large_block;
	}

    // give remaining edges their own bin
    //
	while (num_large_block) prob[large_block[--num_large_block]] = 1;
	while (num_small_block) prob[small_block[--num_small_block]] = 1;

	delete[] norm_prob;
	delete[] small_block;
	delete[] large_block;
}

long long LargeVis::sample_an_edge(real rand_value1, real rand_value2)
{
	long long k = (long long)((n_edge - 0.1) * rand_value1);
	return rand_value2 <= prob[k] ? k : alias[k];
}

void LargeVis::ann_thread(int id)
{
    bool new_only = n_old_vertices > -1 && annoy_index->get_n_items() == n_old_vertices;
    long long start = new_only ? n_old_vertices : 0;
    long long count = new_only ? n_vertices - n_old_vertices : n_vertices;
	long long lo = id * count / n_threads + start;
	long long hi = (id + 1) * count / n_threads + start;
	AnnoyIndex<int, real, Euclidean, Kiss64Random> *cur_annoy_index = NULL;
	//printf("lo: %d, hi: %d\n", lo, hi);fflush(stdout);

	if (id > 0)
	{
		cur_annoy_index = new AnnoyIndex<int, real, Euclidean, Kiss64Random>(n_dim);
		cur_annoy_index->load("annoy_index_file");
	}
	else
		cur_annoy_index = annoy_index;
	for (long long i = lo; i < hi; ++i)
	{
		if (interrupt) break;
		if ( id == 0 && i > lo && (i-lo)%((hi-lo)/100) == 0 ) logoutp("%cQuerying ANN index with %d neighbors ......%lld%%", 13, n_neighbors, 100*(i-lo)/((hi-lo))+1);
		cur_annoy_index->get_nns_by_item(i, n_neighbors + 1, (n_neighbors + 1) * n_trees, &knn_vec[i], NULL);
		for (long long j = 0; j < knn_vec[i].size(); ++j)
			if (knn_vec[i][j] == i)
			{
				knn_vec[i].erase(knn_vec[i].begin() + j);
				break;
			}
	}
	if (id > 0) delete cur_annoy_index;
}

void LargeVis::add_to_index_thread(long long start)
{
	for (long long i = start; i < n_vertices; ++i) {
		if (interrupt) break;
		annoy_index->add_item(i, vec + i * n_dim);
	}
}

void *LargeVis::add_to_index_thread_caller(void *arg)
{
	LargeVis *ptr = (LargeVis*)(((arg_struct_index*)arg)->ptr);
	ptr->add_to_index_thread(((arg_struct_index*)arg)->start);
	pthread_exit(NULL);
}

void LargeVis::add_to_index(long long start = 0)
{
	pthread_t *pt = new pthread_t[1];
	pthread_create(&pt[0], NULL, LargeVis::add_to_index_thread_caller, new arg_struct_index(this, start));
	wait_for_threads(pt, 1);
}

void LargeVis::ensure_index()
{
    if (!annoy_index)
    {
    	logout("Building ANNOY index with %lld trees ......\n", n_trees);
		annoy_index = new AnnoyIndex<int, real, Euclidean, Kiss64Random>(n_dim);
		//add_to_index(0);
		add_to_index(0);
		annoy_index->build(n_trees);
		if (n_threads > 1) annoy_index->save("annoy_index_file");
    	logout(" Done with %d points in index.\n", annoy_index->get_n_items());
    }
}

void LargeVis::index_new_points()
{
    logout("Adding %lld new points to ANN index ......", n_vertices - n_old_vertices);
	add_to_index(n_old_vertices);
	annoy_index->build(n_trees);
    logout(" Done.\n");
}

void *LargeVis::ann_thread_caller(void *arg)
{
	LargeVis *ptr = (LargeVis*)(((arg_struct*)arg)->ptr);
	ptr->ann_thread(((arg_struct*)arg)->id);
	pthread_exit(NULL);
}

void LargeVis::run_ann()
{
    ensure_index();
	if (interrupt) return;
	knn_vec = new std::vector<int>[n_vertices];
    //printf("Querying %d points in ANN index ......", faiss_Index_ntotal(faiss_index)); fflush(stdout);
	pthread_t *pt = new pthread_t[n_threads];
	for (int j = 0; j < n_threads; ++j) pthread_create(&pt[j], NULL, LargeVis::ann_thread_caller, new arg_struct(this, j));
	wait_for_threads(pt, n_threads);
	logout(" Done.\n");
}

void LargeVis::propagation_thread(int id)
{
	long long lo = id * n_vertices / n_threads;
	long long hi = (id + 1) * n_vertices / n_threads;
	int *check = new int[n_vertices];
	std::priority_queue< pair<real, int> > heap;
	long long x, y, i, j, l1, l2;
	for (x = 0; x < n_vertices; ++x) check[x] = -1;
	for (x = lo; x < hi; ++x)
	{
		if (interrupt) break;
		check[x] = x;
		std::vector<int> &v1 = old_knn_vec[x];
		l1 = v1.size();
		for (i = 0; i < l1; ++i)
		{
			y = v1[i];
			if (y == x) continue;
			check[y] = x;
			heap.push(std::make_pair(CalcDist(x, y), y));
			if (heap.size() == n_neighbors + 1) heap.pop();
		}
		for (i = 0; i < l1; ++i)
		{
			std::vector<int> &v2 = old_knn_vec[v1[i]];
			l2 = v2.size();
			for (j = 0; j < l2; ++j) if (check[y = v2[j]] != x)
			{
				check[y] = x;
				heap.push(std::make_pair(CalcDist(x, y), (int)y));
				if (heap.size() == n_neighbors + 1) heap.pop();
			}
		}
		i = 0;
		while (!heap.empty())
		{
			knn_vec[x].push_back(heap.top().second);
			heap.pop();
		}
	}
	delete[] check;
}

void *LargeVis::propagation_thread_caller(void *arg)
{
	LargeVis *ptr = (LargeVis*)(((arg_struct*)arg)->ptr);
	ptr->propagation_thread(((arg_struct*)arg)->id);
	pthread_exit(NULL);
}

void LargeVis::run_propagation()
{
	for (int i = 0; i < n_propagations; ++i)
	{
		logout("%cRunning propagation %d/%lld", 13, i + 1, n_propagations);
		old_knn_vec = knn_vec;
		knn_vec = new std::vector<int>[n_vertices];
		pthread_t *pt = new pthread_t[n_threads];
		for (int j = 0; j < n_threads; ++j) pthread_create(&pt[j], NULL, LargeVis::propagation_thread_caller, new arg_struct(this, j));
		wait_for_threads(pt, n_threads);
		delete[] old_knn_vec;
		old_knn_vec = NULL;
	}
	logout("\n");
}

void LargeVis::compute_similarity_thread(int id)
{
	long long lo = id * n_vertices / n_threads;
	long long hi = (id + 1) * n_vertices / n_threads;
	long long x, iter, p;
	real beta, lo_beta, hi_beta, sum_weight, H, tmp;
	for (x = lo; x < hi; ++x)
	{
		if (interrupt) break;
		beta = 1;
		lo_beta = hi_beta = -1;
		for (iter = 0; iter < 200; ++iter)
		{
			if (interrupt) break;
			H = 0;
            		sum_weight = FLT_MIN;
			for (p = head[x]; p >= 0; p = next[p])
			{
				sum_weight += tmp = exp(-beta * edge_weight[p]);
				H += beta * (edge_weight[p] * tmp);
			}
			H = (H / sum_weight) + log(sum_weight);
			if (fabs(H - log(perplexity)) < 1e-5) break;
			if (H > log(perplexity))
			{
				lo_beta = beta;
				if (hi_beta < 0) beta *= 2; else beta = (beta + hi_beta) / 2;
			}
			else{
				hi_beta = beta;
				if (lo_beta < 0) beta /= 2; else beta = (lo_beta + beta) / 2;
			}
            		if(beta > FLT_MAX) beta = FLT_MAX;
        	}
		for (p = head[x], sum_weight = FLT_MIN; p >= 0; p = next[p])
		{
			sum_weight += edge_weight[p] = exp(-beta * edge_weight[p]);
		}
		for (p = head[x]; p >= 0; p = next[p])
		{
			edge_weight[p] /= sum_weight;
		}
	}
}

void *LargeVis::compute_similarity_thread_caller(void *arg)
{
	LargeVis *ptr = (LargeVis*)(((arg_struct*)arg)->ptr);
	ptr->compute_similarity_thread(((arg_struct*)arg)->id);
	pthread_exit(NULL);
}

real LargeVis::rng()
{
	return (*distr)(*gen);
}

void LargeVis::search_reverse_thread(int id)
{
	long long lo = id * n_vertices / n_threads;
	long long hi = (id + 1) * n_vertices / n_threads;
	long long x, y, p, q;
	for (x = lo; x < hi; ++x)
	{
		if (interrupt) break;
		for (p = head[x]; p >= 0; p = next[p])
		{
			y = edge_to[p];
			for (q = head[y]; q >= 0; q = next[q])
			{
				if (edge_to[q] == x) break;
			}
			reverse[p] = q;
		}
	}
}

void *LargeVis::search_reverse_thread_caller(void *arg)
{
	LargeVis *ptr = (LargeVis*)(((arg_struct*)arg)->ptr);
	ptr->search_reverse_thread(((arg_struct*)arg)->id);
	pthread_exit(NULL);
}

void LargeVis::compute_similarity()
{
    logout("Computing similarities ......");
	n_edge = 0;
	head = new long long[n_vertices];
	long long i, x, y, p, q;
	real sum_weight = 0;
	for (i = 0; i < n_vertices; ++i) head[i] = -1;
	for (x = 0; x < n_vertices; ++x)
	{
		for (i = 0; i < knn_vec[x].size(); ++i)
		{
			edge_from.push_back((int)x);
			edge_to.push_back((int)(y = knn_vec[x][i]));
			edge_weight.push_back(CalcDist(x, y));
			next.push_back(head[x]);
			reverse.push_back(-1);
			head[x] = n_edge++;
		}
	}
    	delete[] vec; vec = NULL;
    	delete[] knn_vec; knn_vec = NULL;
	pthread_t *pt = new pthread_t[n_threads];
	for (int j = 0; j < n_threads; ++j) pthread_create(&pt[j], NULL, LargeVis::compute_similarity_thread_caller, new arg_struct(this, j));
	wait_for_threads(pt, n_threads);
	if (interrupt) return;
	pt = new pthread_t[n_threads];
	for (int j = 0; j < n_threads; ++j) pthread_create(&pt[j], NULL, LargeVis::search_reverse_thread_caller, new arg_struct(this, j));
	wait_for_threads(pt, n_threads);

	for (x = 0; x < n_vertices; ++x)
	{
		for (p = head[x]; p >= 0; p = next[p])
		{
			y = edge_to[p];
			q = reverse[p];
			if (q == -1)
			{
				edge_from.push_back((int)y);
				edge_to.push_back((int)x);
				edge_weight.push_back(0);
				next.push_back(head[y]);
				reverse.push_back(p);
				q = reverse[p] = head[y] = n_edge++;
			}
			if (x > y){
				sum_weight += edge_weight[p] + edge_weight[q];
				edge_weight[p] = edge_weight[q] = (edge_weight[p] + edge_weight[q]) / 2;
			}
		}
	}
	logout(" Done.\n");
}

void LargeVis::test_accuracy()
{
	long long test_case = 100;
	std::priority_queue< pair<real, int> > *heap = new std::priority_queue< pair<real, int> >;
	long long hit_case = 0, i, j, x, y;
	for (i = 0; i < test_case; ++i)
	{
		x = floor(rng() * (n_vertices - 0.1));
		for (y = 0; y < n_vertices; ++y) if (x != y)
		{
			heap->push(std::make_pair(CalcDist(x, y), y));
			if (heap->size() == n_neighbors + 1) heap->pop();
		}
		while (!heap->empty())
		{
			y = heap->top().second;
			heap->pop();
			for (j = 0; j < knn_vec[x].size(); ++j) if (knn_vec[x][j] == y)
				++hit_case;
		}
	}
    	delete heap;
	logout("Test knn accuracy : %.2f%%\n", hit_case * 100.0 / (test_case * n_neighbors));
}

void LargeVis::construct_knn()
{
    if (n_old_vertices > -1) index_new_points();
	run_ann(); if (interrupt) return;
	test_accuracy(); if (interrupt) return;
	run_propagation(); if (interrupt) return;
	test_accuracy(); if (interrupt) return;
	compute_similarity(); if (interrupt) return;

	/*FILE *fout = fopen("knn_graph.txt", "wb");
	for (long long p = 0; p < n_edge; ++p)
	{
		fprintf(fout, "%lld %lld ", edge_from[p], edge_to[p]);
		double tmp = edge_weight[p];
		fwrite(&tmp, sizeof(double), 1, fout);
		fprintf(fout, "\n");
	}
	fclose(fout);*/
}

void LargeVis::init_neg_table()
{
	long long x, p, i;
	neg_size = 1e8;
    reverse.clear(); vector<long long> (reverse).swap(reverse);
	bool new_only = !update_old_points && n_old_vertices > -1;
	long long start = new_only ? n_old_vertices : 0;
	long long count = new_only ? n_vertices - n_old_vertices : n_vertices;
	real sum_weights = 0, dd, *weights = new real[count];
	for (i = 0; i < count; ++i) weights[i] = 0;
	for (x = start; x < n_vertices; ++x)
	{
		for (p = head[x]; p >= 0; p = next[p])
		{
			weights[x-start] += edge_weight[p];
		}
		sum_weights += weights[x-start] = pow(weights[x-start], 0.75);
	}
    	next.clear(); vector<long long> (next).swap(next);
    	delete[] head; head = NULL;
	neg_table = new int[neg_size];
	dd = weights[0];
	for (i = x = 0; i < neg_size; ++i)
	{
		neg_table[i] = x;
		if (i / (real)neg_size > dd / sum_weights && x < n_vertices - 1)
		{
			dd += weights[++x-start];
		}
	}
	delete[] weights;
}

void LargeVis::visualize_thread(int id)
{
	long long edge_count = 0, last_edge_count = 0;
	long long x, y, p, lx, ly, i, j;
	real f, g, gg, cur_alpha = initial_alpha;
	real *cur = new real[out_dim];
	real *tail = new real[out_dim];
	real *err = new real[out_dim]; // for theta only
	real grad_clip = 0.5;
	real grad_clip_rho = 0.001;
	real beta_cur = 1;
	long long updates = 0;
	char buffer[10000];
	while (1)
	{
		if (interrupt) break;
		if (edge_count > n_samples / n_threads + 2) break;
		if (edge_count - last_edge_count > 10000)
		{
			edge_count_actual += edge_count - last_edge_count;
			last_edge_count = edge_count;
			cur_alpha = initial_alpha * (1 - edge_count_actual / (n_samples + 1.0));
			//cur_alpha = cur_alpha * cur_alpha;
			if (cur_alpha < initial_alpha * 0.0001) cur_alpha = initial_alpha * 0.0001;
			beta_cur = 0;//(1 - edge_count_actual / (n_samples + 1.0))*(1.-beta)+beta;
			logoutp("%cFitting model\tAlpha: %f\tProgress: %.3lf%%", 13, cur_alpha, (real)edge_count_actual / (real)(n_samples + 1) * 100);
			const bool checkpoints = true;
			if (checkpoints && id == 0 && updates++ % (n_samples/10000000) == 0)
			{
				sprintf(buffer, "%lld.out", 10000000 * updates / n_samples);
				//save(buffer);
			}
		}
		p = sample_an_edge(rng(), rng()); // positive sample
		x = edge_from[p]; // head node
		y = edge_to[p]; // tail node
		lx = x * out_dim; // head index
		real bias=zeta;//exp(zeta)/(1+exp(zeta));
		real er = exp(rho);
		real rx = 100*(bias+pow(vis[lx+1], er)*(1-bias));
		real gr = 0;
		real gz = 0;
		
		if (!update_old_points && x < n_old_vertices && y < n_old_vertices) continue;

		// initialize cur as current embedding vector of x
		// initialize err to 0
		//
		if (time)
		{
			cur[0] = rx*cos(vis[lx]);
			cur[1] = rx*sin(vis[lx]);
			err[0] = 0;
		}
		else
			for (i = 0; i < out_dim; ++i) cur[i] = vis[lx + i], err[i] = 0;
		
		// first iteration is positive edge; the rest are negative
		//
		for (i = 0; i < n_negatives + 1; ++i)
		{
			if (i > 0)
			{
				y = neg_table[(unsigned long long)floor(rng() * (neg_size - 0.1))];
				if (y == edge_to[p]) continue;
			}
			ly = y * out_dim; // tail index for positive edge (i==0) or negative edge (i>0)
			
			real ry = 100*(bias+pow(vis[ly+1], er)*(1-bias));
			if (time)
			{
				tail[0] = ry*cos(vis[ly]);
				tail[1] = ry*sin(vis[ly]);
			}
			else
				for (j = 0; j < out_dim; ++j) tail[j] = vis[ly + j];

			// f = norm^2(x_e, y_e)
			//
			for (j = 0, f= 0; j < out_dim; ++j) f += (cur[j] - tail[j]) * (cur[j] - tail[j]);
			
			real d = vis[lx] - vis[ly];
			if (d > pi) d -= twopi;
			if (d < -pi) d += twopi;
			d *= 32;
			real ff = d*d;
			//const real beta = 0.5;
			real h;

			if (i == 0) {g = -2 / (1 + f); h = -2 / (1+ff);}
			else {g = 2 * gamma / (1 + f) / (0.1 + f); h = 2 * gamma / (1 + ff) / (0.1 + ff);}

			for (j = 0; j < out_dim; ++j)
			//for (j = 0; j < 1; ++j)
			{
				gg = (1-beta) * g * (cur[j] - tail[j]) * rx * (j == 0 ? -sin(vis[lx]) : cos(vis[lx]));
				if (j == 0) gg += beta * h * d;// * vis[lx+1] * (j == 0 ? -sin(vis[lx]) : cos(vis[lx]));
				err[time ? 0 : j] += gg;
			
				gr += g * (cur[j] - tail[j]) * rx * er * log(vis[lx+1]) * (j == 0 ? cos(vis[lx]) : sin(vis[lx]));
				gz += g * (cur[j] - tail[j]) * bias * (1-bias) * (j == 0 ? cos(vis[lx]) : sin(vis[lx]));

				if (update_old_points || y >= n_old_vertices)
				{
					gg = (1 - beta) * g * (tail[j] - cur[j]) * ry * (j == 0 ? -sin(vis[ly]) : cos(vis[ly]));
					if (j == 0) gg += beta * h * -d;// * vis[ly+1] * (j == 0 ? -sin(vis[ly]) : cos(vis[ly]));
					if (gg > grad_clip)	 gg = grad_clip;
					if (gg < -grad_clip) gg = -grad_clip;
					vis[ly] += gg * cur_alpha;
					gr += g * (tail[j] - cur[j]) * ry * log(vis[ly+1]) * er * (j == 0 ? cos(vis[ly]) : sin(vis[ly]));
					gz += g * (tail[j] - cur[j]) * bias * (1-bias) * (j == 0 ? cos(vis[ly]) : sin(vis[ly]));
				}
			}
			if (vis[ly] < 0) vis[ly] += twopi;
			if (vis[ly] > twopi) vis[ly] -= twopi;
		}
		if (update_old_points || x >= n_old_vertices)
		{
			for (j = 0; j < (time ? 1 : out_dim); ++j)
			{
				err[j] *= cur_alpha;
				if (err[j] > grad_clip) err[j] = grad_clip;
				if (err[j] < -grad_clip) err[j] = -grad_clip;
				vis[lx + j] += err[j] * cur_alpha;
				if (time)
				{
					if (vis[lx] < 0) vis[lx] += twopi;
					if (vis[lx] > twopi) vis[lx] -= twopi;
				}
			}

			gr /= 10000.;
			if (gr > grad_clip_rho) gr = grad_clip_rho;
			if (gr < -grad_clip_rho) gr = -grad_clip_rho;
			//rho += gr * cur_alpha;

			//zeta += gz * cur_alpha / 10000.;
		}
		++edge_count;
	}
	
	delete[] cur;
	delete[] tail;
}

void *LargeVis::visualize_thread_caller(void *arg)
{
	LargeVis *ptr = (LargeVis*)(((arg_struct*)arg)->ptr);
	ptr->visualize_thread(((arg_struct*)arg)->id);
	pthread_exit(NULL);
}

void LargeVis::visualize()
{
	long long i;
	if (!vis)
	{
	    vis = new real[n_vertices * out_dim];
    	for (i = 0; i < n_vertices; ++i)
		{
			vis[i<<1] = rng() * 2 * 3.1415926;
			vis[(i<<1)+1] = time[i];
		}
    }
	//zeta = -2;
	init_neg_table();
	init_alias_table();
	edge_count_actual = 0;
	pthread_t *pt = new pthread_t[n_threads];
	for (int j = 0; j < n_threads; ++j) pthread_create(&pt[j], NULL, LargeVis::visualize_thread_caller, new arg_struct(this, j));
	wait_for_threads(pt, n_threads);
	if (interrupt) return;
	if (time)
	{
		real r, er = exp(rho);
		for (long long i = 0; i < n_vertices; ++i)
		{
			// swap radius and angle for standard polar coordinates
			r = zeta + pow(vis[i * out_dim + 1], er) * (1 - zeta);

			vis[i * out_dim + 1] = vis[i * out_dim];
			vis[i * out_dim] = r;
		}
	}
	logout("\n");
}

bool LargeVis::run(long long out_d, long long n_thre, long long n_samp, long long n_tree, long long n_prop, real alph, real beta_, real rho_, real zeta_, long long n_nega, long long n_neig, real gamm, real perp, long long init_neighbors, bool update_old_points, char *outindex)
{
	interrupt = false;

	std::random_device rd;
	gen = new std::mt19937(rd()); // TODO: make seed an argument (need to change Kiss64Random in AnnoyLib to accept seed)
    distr = new std::uniform_real_distribution<real>(0.0f, 1.0f);

	clean_model();
	this->update_old_points = update_old_points;
	if (!vec && !head)
	{
		printf("Missing training data!\n");
		return false;
	}
	if (out_d > 0)
	{
	    if (out_dim > 0)
    	    printf("WARNING: Outdim set to %lld by -init; ignoring -outdim %lld)\n", out_dim, out_d);
    	else
    	    out_dim = out_d;
    }
	out_dim = out_d < 0 ? 2 : out_d;
	n_threads = n_thre < 0 ? 8 : n_thre;
	initial_alpha = alph < 0 ? 1.0 : alph;
	beta = beta_ < 0 ? 0.5 : beta_;
	rho = rho_;
	zeta = zeta_;
	n_samples = n_samp;
	n_trees = n_tree;
	n_negatives = n_nega < 0 ? 5 : n_nega;
	n_neighbors = n_neig < 0 ? 150 : n_neig;
	n_propagations = n_prop < 0 ? 3 : n_prop;
	gamma = gamm < 0 ? (time ? 128 : 7.0) : gamm;
	perplexity = perp < 0 ? 50.0 : perp;
	
	if (n_trees < 0)
	{
		if (n_vertices < 100000)
			n_trees = 10;
		else if (n_vertices < 1000000)
			n_trees = 20;
		else if (n_vertices < 5000000)
			n_trees = 50;
		else n_trees = 100;
	}
	if (n_samples < 0)
	{
		if (n_vertices < 10000)
			n_samples = 1000;
		else if (n_vertices < 1000000)
			n_samples = (n_vertices - 10000) * 9000 / (1000000 - 10000) + 1000;
		else n_samples = n_vertices / 100;
	}
	n_samples *= 1000000;
	if (vec)
	{
	    clean_graph();
	    construct_knn();
		if (interrupt)
		{
			clean_model();
			vec = time = NULL;
			delete annoy_index;
			annoy_index = NULL;
			return false;
		}
	}
	if (outindex && outindex[0]) saveindex(outindex);
	delete annoy_index;
	annoy_index = NULL;

	visualize();

	clean_model();
	delete distr;
	delete gen;

	if (interrupt)
	{
		return false;
	}
	return true;
}

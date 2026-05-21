// Based on https://raw.githubusercontent.com/lferry007/LargeVis/refs/heads/master/Linux/LargeVis.h
// Modified by Brian Ondov on 2026-05-21

#ifndef LARGEVIS_H
#define LARGEVIS_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>

#include "ANNOY/annoylib.h"
#include "ANNOY/kissrandom.h"

#include <pthread.h>
#include <random>

typedef float real;

struct arg_struct{
	void *ptr;
	int id;
	arg_struct(void *x, int y) :ptr(x), id(y){}
};

struct arg_struct_index{
	void *ptr;
	long long start;
	arg_struct_index(void *x, long long y) :ptr(x), start(y){}
};

class LargeVis{
private:
	long long n_vertices, n_dim, out_dim, n_samples, n_trees, n_threads, n_negatives, n_neighbors, n_propagations, edge_count_actual, n_old_vertices;
	real initial_alpha, gamma, perplexity;
	real *vec, *time;
	real *vis;
	real rho;
	real zeta;
	real beta;
	std::vector<std::string> names;
	std::vector<bool> update; // TODO: remove
	bool resume;
	std::vector<int> *knn_vec, *old_knn_vec;
	AnnoyIndex<int, real, Euclidean, Kiss64Random> *annoy_index;
	long long n_edge, *head;
	bool interrupt;
    std::vector<long long> next, reverse;
    std::vector<int> edge_from, edge_to;
	std::vector<real> edge_weight;
    int *neg_table;
    long long neg_size;
	long long *alias;
	real *prob;
	bool update_old_points;

	std::mt19937 * gen;
	std::uniform_real_distribution<real> * distr;

	void clean_model();
	void clean_data();
	void clean_graph();
	void normalize();
	real CalcDist(long long x, long long y);
	void init_alias_table();
	long long sample_an_edge(real rand_value1, real rand_value2);
	void ensure_index();
	void add_to_index(long long start);
	void add_to_index_thread(long long start);
	static void *add_to_index_thread_caller(void *arg);
	void run_ann();
	void init_by_nn();
	void ann_thread(int id);
	static void *ann_thread_caller(void *arg);
	void run_propagation();
	void propagation_thread(int id);
	static void *propagation_thread_caller(void *arg);
	void test_accuracy();
	void compute_similarity();
	void compute_similarity_thread(int id);
	static void *compute_similarity_thread_caller(void *arg);
	real rng();
	void search_reverse_thread(int id);
	static void *search_reverse_thread_caller(void *arg);
	void construct_knn();
    void index_new_points();
	void init_neg_table();
	void visualize_thread(int id);
	static void *visualize_thread_caller(void *arg);
	void visualize();
	void wait_for_threads(pthread_t *pt, int n_threads);
public:
	LargeVis(long long n_thre = -1);
	void load_from_file(char *infile, char *timefile = NULL, bool normalize = true, bool append = false, long long n_init_neighbors = -1);
	void load_from_graph(char *infile);
	void load_from_data(real *data, long long n_vert, long long n_di, bool normalize, real *time_ = NULL);
    void load_from_index(char *infile);
	void init_from_file(char *infile);
	void save(char *outfile);
	void saveindex(char *outfile);
	bool run(long long out_d = -1, long long n_thre = -1, long long n_samp = -1, long long n_tree = -1, long long n_prop = -1, real alph = -1, real beta = -1, real rho = 0, real zeta = 0, long long n_nega = -1, long long n_neig = -1, real gamm = -1, real perp = -1, long long init_neighbors = -1, bool update_old_points = false, char *outindex = 0);
	real *get_ans();
	long long get_n_vertices();
	long long get_out_dim();
};

#endif
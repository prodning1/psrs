#define TRUE 1
#define FALSE 0

#define DATA_SIZE (2 << 24)
#define P_DEFAULT 1536
#define MAX_RAND 100000000
#define P_RANDOMIZING 16

#define SOFT_MAX TRUE
#define DISABLE_PRINT_ARRAY TRUE

#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "malloc.h"
#include "pthread.h"

struct array_st {
	int *arr;
	int elements;
	int th_id;
};

void psrs_sort(int *input, int size);
void rand_array(int *result, int size);
void print_array(int *input, int size);
void print_array_ptr(int **input, int size);
void split_array(int *input, int size, int k,  int** segment_indices, int* segment_sizes);
void find_pivots(int *data, int **indices, int *sizes, int n, int p, int* output);
void quicksort(int *arr, int elements);
void dumpMallinfo(void);
void rand_array_threaded(int *result, int size, int p);
void *quicksort_th(void* arg);
void *rand_array_thread(void* arg);

int main(void)
{
	printf("Allocating data...\n");
	fflush(stdout);
	int* data = malloc(DATA_SIZE * sizeof(int));

	if (data) {

		printf("Randomizing data...");
		#ifdef PTHREADS_RND
		printf("pthreads enabled...");
		fflush(stdout);
		rand_array_threaded(data, DATA_SIZE, P_RANDOMIZING);
		#else
		printf("no threading...");
		rand_array(data, DATA_SIZE);
		#endif
		printf("done.\n");


		print_array(data, DATA_SIZE);

		psrs_sort(data, DATA_SIZE);


		free(data);

	} else {
		fprintf(stderr, "ERROR: unable to allocate memory for data.\n");
		exit(EXIT_FAILURE);
	}

	return EXIT_SUCCESS;
}

//CUDA host
void psrs_sort(int *input, int size)
{
	#if SOFT_MAX == TRUE
	int ds_max = (int) pow(size, (1.0/3.0));
	#else
	int ds_max = (int) pow(size, (1.0/2.0));
	#endif

	int p;

	if(P_DEFAULT > ds_max)
		p = ds_max;
	else
		p = P_DEFAULT;

	int** segment_indices = malloc(p * sizeof(int*));
	int* segment_sizes = malloc(p * sizeof(int));
	
	if (segment_indices && segment_sizes) {
		printf("Splitting...");
		split_array(input, size, p, segment_indices, segment_sizes);
		printf("done.\n");

		int i;
		for(i = 0; i < p; i++)
			print_array(segment_indices[i], segment_sizes[i]);

		printf("quicksort segments...");
		fflush(stdout);

		#ifdef PTHREADS
		printf("pthreads enabled...");
	
		struct array_st *qs_arg = malloc(p * sizeof(struct array_st));
		pthread_t *th = malloc(p * sizeof(pthread_t));

		for(i = 0; i < p; i++) {
			qs_arg[i].arr = segment_indices[i];
			qs_arg[i].elements = segment_sizes[i];
			
			pthread_create(&th[i], NULL, quicksort_th, (void*) &qs_arg[i]);
		}

		printf("waiting...");
		fflush(stdout);

		for(i = 0; i < p; i++)
			pthread_join(th[i], NULL);

		free(th);
		free(qs_arg);
		#else

		printf("no threading...");
		//CUDA loop
		for(i = 0; i < p; i++)
			quicksort(segment_indices[i], segment_sizes[i]);
		//end CUDA loop

		#endif

		printf("done.\n");

		printf("finding pivots...\n");
		int* pivots = malloc((p - 1) * sizeof(int));
		if(pivots) {

			find_pivots(input, segment_indices, segment_sizes, size, p, pivots);

			printf("pivots: ");
			print_array(pivots, p - 1);

			free(pivots);
		} else {
			fprintf(stderr, "ERROR: Unable to allocate memory for pivots\n");
			exit(EXIT_FAILURE);
		}
		
		printf("done\n");

		for(i = 0; i < p; i++)
			print_array(segment_indices[i], segment_sizes[i]);

		free(segment_indices);
		free(segment_sizes);
	} else {
		fprintf(stderr, "ERROR: Unable to allocate memory for segment_indices and segment_sizes\n");
		exit(EXIT_FAILURE);
	}
}

//CUDA host
void find_pivots(int *data, int **segment_indices, int *segment_sizes, int n, int p, int* output)
{
	int i;
	int *samples = malloc(p * p * sizeof(int));
	if(samples) {
		//CUDA loop (will become global function)
		for(i = 0; i < p; i++) {
			long long j;
			for(j = 0; j < p; j++) {
				long long local_index = (long long) (j * (long long) n / (p * p));
				samples[i * p + j] = segment_indices[i][local_index];
			}
		}

		quicksort(samples, p * p);

		//int index_offset = (int) (((p * p) - (n / (p * p)) * (p - 2)) / 2);
		int index_offset = (int) p / 2;

		printf("offset:%d\tsmp:%d\tn:%d\tp:%d\n", index_offset, p * p, n, p);

		for(i = 0; i < p - 1; i++) {
			int pivot_index = i * p + index_offset;
			output[i] = samples[pivot_index];
		}

		print_array(samples, p * p);
		print_array(output, p - 1);

		free(samples);
	} else {
		fprintf(stderr, "ERROR: Unable to allocate memory for samples\n");
		exit(EXIT_FAILURE);
	}
}

//  quicksort
//
//  This public-domain C implementation by Darel Rex Finley
//  who by the way writes the UGLIEST CODE OF ALL TIME.
//  seriously, damn that code was ugly before I reformatted it
//
//  * Returns TRUE if sort was successful, or FALSE if the nested
//    pivots went too deep, in which case your array will have
//    been re-ordered, but probably not sorted correctly.
//
//  * This function assumes it is called with valid parameters.
//
//  * Example calls:
//    quicksort(&myArray[0],5); // sorts elements 0, 1, 2, 3, and 4
//    quicksort(&myArray[3],5); // sorts elements 3, 4, 5, 6, and 7


//CUDA global
void quicksort(int *arr, int elements)
{

	#define MAX_LEVELS 1000

	int piv, beg[MAX_LEVELS], end[MAX_LEVELS], i=0, L, R;

	beg[0] = 0;
	end[0] = elements;
	while (i >= 0) {
		L = beg[i];
		R = end[i] - 1;
		if (L < R) {
			piv = arr[L];

			if (i == MAX_LEVELS - 1)
				return;
			while (L < R) {
				while (arr[R] >= piv && L < R)
					R--;

				if (L < R)
					arr[L++] = arr[R];

				while (arr[L] <= piv && L < R)
					L++;

				if (L<R)
					arr[R--]=arr[L];
			}
			arr[L] = piv;
			beg[i+1] = L+1;
			end[i+1] = end[i];
			end[i++] = L;
		} else {
			i--;
		}
	}
}

void *quicksort_th(void* arg)
{
	struct array_st *args = (struct array_st *) arg;
	int* arr = args->arr;
	int elements = args->elements;
	#define MAX_LEVELS 1000

	int piv, beg[MAX_LEVELS], end[MAX_LEVELS], i=0, L, R;

	beg[0] = 0;
	end[0] = elements;
	while (i >= 0) {
		L = beg[i];
		R = end[i] - 1;
		if (L < R) {
			piv = arr[L];

			if (i == MAX_LEVELS - 1)
				return;
			while (L < R) {
				while (arr[R] >= piv && L < R)
					R--;

				if (L < R)
					arr[L++] = arr[R];

				while (arr[L] <= piv && L < R)
					L++;

				if (L<R)
					arr[R--]=arr[L];
			}
			arr[L] = piv;
			beg[i+1] = L+1;
			end[i+1] = end[i];
			end[i++] = L;
		} else {
			i--;
		}
	}
}

/* CUDA host
 * input should be of size size
 * k is the number of divisions, and should be less than or equal to size
 * segment_indices is a list of pointers to the split arrays, and should be of size = k * sizeof(int*)
 * segment_sizes is a list of sizes of the split arrays, and should be of size = k * sizeof(int)
 */
void split_array(int *input, int size, int k,  int** segment_indices, int* segment_sizes)
{
	int div_size = size / k;
	int rem = size % k;

	int i;
	for(i = 0; i < k; i++) {
		if(i < rem)
			segment_sizes[i] = div_size + 1;
		else
			segment_sizes[i] = div_size;

		if(i == 0)
			segment_indices[i] = input;
		else
			segment_indices[i] = segment_indices[i - 1] + segment_sizes[i - 1];
	}
}

void rand_array_threaded(int *result, int size, int p) {
	struct array_st *rand_arr_st = malloc(p * sizeof(struct array_st));
	pthread_t *th = malloc(p * sizeof(pthread_t));

	int** segment_indices = malloc(p * sizeof(int*));
	int* segment_sizes = malloc(p * sizeof(int));

	if(!(segment_indices && segment_sizes && rand_arr_st && th)) {
		fprintf(stderr, "ERROR: unable to allocate memory for randomizing.\n");
		exit(EXIT_FAILURE);
	}

	split_array(result, size, p, segment_indices, segment_sizes);

	int i;
	for(i = 0; i < p; i++) {
		rand_arr_st[i].arr = segment_indices[i];
		rand_arr_st[i].elements = segment_sizes[i];
		rand_arr_st[i].th_id = i;

		pthread_create(&th[i], NULL, rand_array_thread, (void*) &rand_arr_st[i]);
	}

	printf("waiting...");
	fflush(stdout);

	for(i = 0; i < p; i++)
		pthread_join(th[i], NULL);

	printf("done.\n");


	printf("freeing memory...\n");

	free(segment_sizes);
	free(segment_indices);

	free(th);
	free(rand_arr_st);
}

//CUDA host
void *rand_array_thread(void* ptr)
{
	struct array_st *arr_st = (struct array_st *) ptr;

	int *result = arr_st->arr;
	int size = arr_st->elements;

	int i;
	for (i = 0; i < size; i++) {
		int seed = (int) time(NULL) + arr_st->th_id;
		result[i] = rand_r(&seed) % MAX_RAND;
	}
}

//CUDA host
void rand_array(int *result, int size)
{
	srand(time(NULL));

	int i;
	for (i = 0; i < size; i++) {
		result[i] = rand() % MAX_RAND;
	}
}

//CUDA host
void print_array(int *input, int size)
{
	#if DISABLE_PRINT_ARRAY == FALSE
	int i;
	printf("[");
	for(i = 0; i < size; i++)
		printf("%d%s", input[i], i == size - 1 ? "]\n" : " ");
	if(size == 0)
		printf(" ]\n");
	#endif
}

//CUDA host
void print_array_ptr(int **input, int size)
{
	int i;
	printf("[");
	for(i = 0; i < size; i++)
		printf("%d%s", *(input[i]), i == size - 1 ? "]\n" : " ");
	if(size == 0)
		printf(" ]\n");
}

void dumpMallinfo(void)
{
	struct mallinfo m = mallinfo();
	printf("uordblks = %d\tfordblks = %d\n", m.uordblks, m.fordblks);
}

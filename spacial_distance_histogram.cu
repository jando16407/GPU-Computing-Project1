/* ==================================================================
	The basic SDH algorithm implementation for 3D data
	To compile: nvcc spacial_distance_histogram.c -o SDH in the c4cuda machines
   ==================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>


#define BOX_SIZE	23000 /* size of the data box on one dimension            */

/* descriptors for single atom in the tree */
typedef struct atomdesc {
	double x_pos;
	double y_pos;
	double z_pos;
} atom;

typedef struct hist_entry{
	//float min;
	//float max;
	long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;


bucket * histogram;		/* list of all buckets in the histogram   */
bucket * histogram_GPU;	/* list of all buckets in the histogram used for GPU computing */
long long total_num_data_points;   /* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
double bucket_width;	/* value of w                             */
atom * atom_list;		/* list of all data points                */

/* These are for an old way of tracking time */
struct timezone Idunno;	
struct timeval startTime, endTime;


/* 
	distance of two points in the atom_list 
*/
double p2p_distance(int ind1, int ind2) {
	
	double x1 = atom_list[ind1].x_pos;
	double x2 = atom_list[ind2].x_pos;
	double y1 = atom_list[ind1].y_pos;
	double y2 = atom_list[ind2].y_pos;
	double z1 = atom_list[ind1].z_pos;
	double z2 = atom_list[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}


/* 
	brute-force SDH solution in a single CPU thread 
*/
int PDH_baseline() {
	int i, j, h_pos;
	double dist;
	
	for(i = 0; i < total_num_data_points; i++) {
		for(j = i+1; j < total_num_data_points; j++) {
			dist = p2p_distance(i,j);
			h_pos = (int) (dist / bucket_width);
			histogram[h_pos].d_cnt++;
		} 
	}
	return 0;
}

/* 
	Kernel version of SDH solution
*/
__global__
void PDH_Kernel(bucket * hist, atom * at_list, long long total_points, double b_width){
	/* Declare variables */
	//Get index of histogram
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j, h_pos;
	double dist;

	/* Compute distance and update histogram */
	for( j=i+1; j<total_points; ++j ){
		double x1 = at_list[i].x_pos;
        double x2 = at_list[j].x_pos;
        double y1 = at_list[i].y_pos;
        double y2 = at_list[j].y_pos;
        double z1 = at_list[i].z_pos;
		double z2 = at_list[j].z_pos;
		
		dist = sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
		h_pos = (int) (dist / b_width);
		hist[h_pos].d_cnt++;	
	}
	//Synchronize all threads
	__syncthreads();
}

/* 
	set a checkpoint and show the (natural) running time in seconds 
*/
double report_running_time_CPU() {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("\nRunning time for CPU version: %ld.%06ld\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}

/*
	Measure time for GPU comptation
*/
double report_running_time_GPU() {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("\nRunning time for GPU version: %ld.%06ld\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}


/* 
	print the counts in all buckets of the histogram 
*/
void output_histogram(){
	int i; 
	long long total_cnt = 0;
	printf("\nHistogram of CPU computing");
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histogram[i].d_cnt);
		total_cnt += histogram[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}

/* 
	Prints out the counts in histogram for GPU
*/
void output_histogram_GPU(){
	int i; 
	long long total_cnt = 0;
	printf("\nHistogram of GPU computing");
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histogram_GPU[i].d_cnt);
		total_cnt += histogram_GPU[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}

/*
	print the difference in each histogram
*/
void output_histogram_difference(){
	int i; 
	long long total_cnt = 0;
	printf("\nHistogram dirreference of CPU and GPU.");
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histogram_GPU[i].d_cnt - histogram[i].d_cnt);
		total_cnt += histogram_GPU[i].d_cnt - histogram[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}


int main(int argc, char **argv)
{
	/* Read command line value */
	total_num_data_points = atoi(argv[1]);
	bucket_width	 = atof(argv[2]);

	/* Initialize variables */
	num_buckets = (int)(BOX_SIZE * 1.732 / bucket_width) + 1;
	histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);
	histogram_GPU = (bucket *)malloc(sizeof(bucket)*num_buckets);
	atom_list = (atom *)malloc(sizeof(atom)*total_num_data_points);

	/* Declare variables */
	int i;
	bucket * d_histogram_GPU;
	atom * d_atom_list;

	
	/* generate data following a uniform distribution */
	srand(1);
	for(i = 0;  i < total_num_data_points; i++) {
		atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}


	/* GPU memory allocation */
	cudaMalloc((void**)&d_histogram_GPU, sizeof(bucket)*num_buckets);
	cudaMalloc((void**)&d_atom_list, sizeof(atom)*total_num_data_points);
	
	/* GPU memory initialization */
	cudaMemset(d_histogram_GPU, 0, sizeof(bucket)*num_buckets);
	cudaMemcpy(d_atom_list, atom_list, sizeof(atom)*total_num_data_points, cudaMemcpyHostToDevice);

	/* GPU block setting */
	int num_block;
	if( (total_num_data_points%32) != 0 ){
		num_block = (int)(total_num_data_points/32) + 1;
	} else {
		num_block = (int)total_num_data_points / BOX_SIZE;
	}
	dim3 grid(num_block, 1, 1);
	dim3 block(32, 1, 1);


	/******* time the GPU computing ***********************************/
	gettimeofday(&startTime, &Idunno);
	PDH_Kernel<<<grid, block>>>(d_histogram_GPU, d_atom_list, total_num_data_points, bucket_width);
	report_running_time_GPU();

	//Deal with memories
	cudaMemcpy(histogram_GPU, d_histogram_GPU, sizeof(bucket)*num_buckets, cudaMemcpyDeviceToHost);
	cudaFree(d_histogram_GPU);
	cudaFree(d_atom_list);

	//Output the result for GPU
	output_histogram_GPU();

	

	/******* time the CPU computing ***********************************/
	/* start counting time */
	gettimeofday(&startTime, &Idunno);
	
	/* call CPU single thread version to compute the histogram */
	PDH_baseline();
	
	/* check the total running time */ 
	report_running_time_CPU();
	
	/* print out the histogram */
	output_histogram();

	//Display the difference
	output_histogram_difference();
	
	return 0;
}



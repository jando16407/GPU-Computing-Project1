# Summary

This project is a school project for Computing on Massively Parallel Systems. 

It is about computing spatial distance histogram of a collection of 3D points.

The code will:

* Create given number of data points with random x, y, z coordinations.
* Compute the number of particle-to-particle distances.
* Sort distances falling into a series of ranges of given width.
* Display run time of GPU vs CPU.
* Display histogram of how many particles are falling in each bucket for CPU and GPU.
* Display histogram of difference between CPU and GPU bucket.




# Before Running the code

This code will run on CUDA enabled machines.
CUDA version 7.5 is recommended.

# To compile the code

Simply use the makefile to compile the code.

	make

Or 

You can also compile by typing

	nvcc spacial_distance_histogram.cu -o SDH

# To run the code

You need to specify the number of data points (1st arg) and width of each bucket (2nd arg).

1st arg type is (long long) and 2nd arg type is (double).

For example, in order to create 10000 data points and width of 500, run by typing

	./SDH 10000 500.0
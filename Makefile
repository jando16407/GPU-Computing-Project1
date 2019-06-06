
main: spacial_distance_histogram.cu
	nvcc spacial_distance_histogram.cu -o SDH --compiler-options -Wall

.PHONY: clean

clean:
	rm SDH
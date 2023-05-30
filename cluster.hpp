#pragma once

#include <stdint.h>

extern "C"
{
    void *cupcl_init_extract_cluster(void *stream, unsigned int minClusterSize, unsigned int maxClusterSize, float voxelX, float voxelY, float voxelZ, int countThreshold);
    int32_t cupcl_extract_cluster(void *cluster_instance, void *stream, float *cloud_in, int32_t n_count, float *output, uint32_t *index);
    void cupcl_free_extract_cluster(void *cluster);
}
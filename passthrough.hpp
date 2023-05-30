#pragma once

#include <stdint.h>

extern "C"
{
    void cupcl_passthrough_filter(
        void* stream,
        void* cloud,
        uint32_t num_points, 
        uint32_t point_step, 
        float min_dist,
        float max_dist,
        float min_x,
        float min_y,
        float min_z,
        float max_x,
        float max_y,
        float max_z,
        bool invert_bounding_box,
        bool invert_distance,
        float* cloud_filtered,
        uint32_t* num_points_filtered);


    void* cupcl_init_voxel_filter(void* stream, float voxel_size_x, float voxel_size_y, float voxel_size_z);
    uint32_t cupcl_voxel_filter(void* filter_instance, void* stream, float* input, uint32_t input_n, float* filtered);
    void cupcl_free_voxel_filter(void* filter_instance);
}
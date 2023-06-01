#pragma once

extern "C"
{
    void *cupcl_create_stream();
    float *cupcl_create_pointcloud(void *stream, float *point_buffer_cpu, unsigned int n_points);
    void *cupcl_create_buffer(void *stream, void *buffer_cpu, unsigned int element_size, unsigned int n_points);
    void *cupcl_create_empty_buffer(void *stream, unsigned int element_size, unsigned int n);
    void cupcl_upload_buffer(void* stream, void* gpu_buffer, void* cpu_buffer, unsigned int element_size, unsigned int n);
    void cupcl_free_buffer(void *buffer);
    float *cupcl_create_empty_pointcloud(void *stream, unsigned int n_points);
    void cupcl_free_pointcloud(float *cloud);
    void cupcl_destroy_stream(void *stream);
}
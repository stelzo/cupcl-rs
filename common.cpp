#include "common.hpp"
#include <cuda_runtime.h>

extern "C"
{
    void *cupcl_create_empty_buffer(void *stream, unsigned int element_size, unsigned int n)
    {
        cudaStream_t stream_ = (cudaStream_t)stream;
        void *buff = NULL;
        cudaMallocManaged(&buff, element_size * n, cudaMemAttachHost);
        cudaStreamAttachMemAsync(stream_, buff);
        cudaMemsetAsync(buff, 0, element_size * n, stream_);
        cudaStreamSynchronize(stream_);
        return buff;
    }

    void cupcl_upload_buffer(void* stream, void* gpu_buffer, void* cpu_buffer, unsigned int element_size, unsigned int n)
    {
        cudaStream_t stream_ = (cudaStream_t)stream;
        cudaMemcpyAsync(gpu_buffer, cpu_buffer, element_size * n, cudaMemcpyHostToDevice, stream_);
        cudaStreamSynchronize(stream_);
    }

    void *cupcl_create_buffer(void *stream, void *buffer_cpu, unsigned int element_size, unsigned int n)
    {
        cudaStream_t stream_ = (cudaStream_t)stream;
        void *buff = NULL;
        cudaMallocManaged(&buff, element_size * n, cudaMemAttachHost);
        cudaStreamAttachMemAsync(stream_, buff);
        cudaMemcpyAsync(buff, buffer_cpu, element_size * n, cudaMemcpyHostToDevice, stream_);
        cudaStreamSynchronize(stream_);
        return buff;
    }

    void cupcl_free_buffer(void *buffer)
    {
        if (buffer != NULL)
        {
            cudaFree(buffer);
            buffer = NULL;
        }
    }

    void *cupcl_create_stream()
    {
        cudaStream_t stream = NULL;
        cudaStreamCreate(&stream);
        return (void *)stream;
    }

    float *cupcl_create_pointcloud(void *stream, float *point_buffer_cpu, unsigned int n_points)
    {
        return (float *)cupcl_create_buffer(stream, point_buffer_cpu, sizeof(float) * 4, n_points);
    }

    float *cupcl_create_empty_pointcloud(void *stream, unsigned int n_points)
    {
        return (float *)cupcl_create_empty_buffer(stream, sizeof(float) * 4, n_points);
    }

    void cupcl_free_pointcloud(float *cloud)
    {
        cupcl_free_buffer(cloud);
    }

    void cupcl_destroy_stream(void *stream)
    {
        if (stream != NULL)
        {
            cudaStreamDestroy((cudaStream_t)stream);
            stream = NULL;
        }
    }
}

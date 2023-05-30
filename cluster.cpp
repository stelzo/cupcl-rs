#include "cluster.hpp"
#include "common.hpp"

#include <cuda_runtime.h>

typedef struct
{
  unsigned int minClusterSize;
  unsigned int maxClusterSize;
  float voxelX;
  float voxelY;
  float voxelZ;
  int countThreshold;
} extractClusterParam_t;

struct cudaExtractCluster
{
  cudaExtractCluster(cudaStream_t stream = 0);
  ~cudaExtractCluster(void);
  int set(extractClusterParam_t param);
  int extract(float *cloud_in, int nCount, float *output, unsigned int *index);

private:
  void *m_handle = NULL;
};

extern "C"
{
  void *cupcl_init_extract_cluster(void *stream, unsigned int minClusterSize, unsigned int maxClusterSize, float voxelX, float voxelY, float voxelZ, int countThreshold)
  {
    cudaStream_t stream_ = (cudaStream_t)stream;
    cudaExtractCluster *cluster = new cudaExtractCluster(stream_);
    extractClusterParam_t param;
    param.minClusterSize = minClusterSize;
    param.maxClusterSize = maxClusterSize;
    param.voxelX = voxelX;
    param.voxelY = voxelY;
    param.voxelZ = voxelZ;
    param.countThreshold = countThreshold;
    cluster->set(param);
    return cluster;
  }

  /**
   * output is same size as input. index is also the same size in float but saves uint. index[0] is the number of clusters.
   * then each following index[i] is the size of a new cluster. Then from index[1] to index[i-1], we sum up to get "outoff".
   * then we need a k for each point in the cluster. outoff is the offset where the cluster starts, so we iterate up to index[i]
   * and access the point with output[(outoff + k) * 4 + 0-1-2];
   */
  int32_t cupcl_extract_cluster(void *cluster_instance, void *stream, float *cloud_in, int32_t n_count, float *output, uint32_t *index)
  {
    cudaStream_t stream_ = (cudaStream_t)stream;
    cudaExtractCluster *cluster = (cudaExtractCluster *)cluster_instance;
    int32_t ret = cluster->extract(cloud_in, n_count, output, index);
    cudaStreamSynchronize(stream_);
    return ret;
  }

  void cupcl_free_extract_cluster(void *cluster)
  {
    if (cluster == NULL)
      return;
    delete (cudaExtractCluster *)cluster;
    cluster = NULL;
  }
}
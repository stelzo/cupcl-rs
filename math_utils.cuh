#include <cuda_runtime.h>
#include <cmath>

__forceinline__ __device__
float4 add_float4(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__forceinline__ __device__ __host__
float safe_angle(float angle)
{
    const int factor = floor(angle / (2.0f * M_PI));
    return angle - factor * 2.0f * M_PI;
}

__forceinline__ __device__
float angle2d(float2 v1, float2 v2)
{
    return atan2(v1.y, v1.x) - atan2(v2.y, v2.x);
}

__forceinline__ __device__
float4 mul_float4(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__forceinline__ __device__
float dot_float4(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__forceinline__ __device__
float4 conjugate(float4 q)
{
    return make_float4(q.x, -q.z, -q.z, q.x);
}

__forceinline__ __device__
float dot_float3(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__forceinline__ __device__
float3 mul_float3(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__forceinline__ __device__
float3 mul_float3_scalar(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__forceinline__ __device__
float3 cross_float3(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}

__forceinline__ __device__
float3 add_float3(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__forceinline__ __device__
float3 rotate_by_quaternion(float3 p, float4 q)
{
    float3 u = make_float3(q.x, q.y, q.z);
    float s = q.w;

    float3 x = mul_float3_scalar(u, 2.0f * dot_float3(u, p));
    float3 y = mul_float3_scalar(p, s*s - dot_float3(u, u));
    float3 z = mul_float3_scalar(cross_float3(u, p), 2.0f * s);

    return add_float3(add_float3(x, y), z);
}

__forceinline__ __device__
float3 transform_point(float3 p, float4 rotation, float3 translation)
{
    float3 rotated = rotate_by_quaternion(p, rotation);
    return add_float3(rotated, translation);
}

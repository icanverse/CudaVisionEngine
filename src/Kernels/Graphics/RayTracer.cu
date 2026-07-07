#include "../Kernels/Graphics/RayTracer.cuh"
#include "../../include/Graphics/Types3D.cuh"
#include "../Graphics/Shaders.cuh"
#include "Graphics/Shaders.cuh"
#include "Graphics/ElementaryNodes.h"

__device__ inline float3 normalizeVec(float3 v) {
    float length = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (length > 0.00001f) return {v.x / length, v.y / length, v.z / length};
    return {0.0f, 0.0f, 0.0f};
}

__device__ inline float3 rotateX(float3 v, float angle) {
    float cosA = cosf(angle), sinA = sinf(angle);
    return {v.x, v.y * cosA - v.z * sinA, v.y * sinA + v.z * cosA};
}

__device__ inline float3 rotateY(float3 v, float angle) {
    float cosA = cosf(angle), sinA = sinf(angle);
    return {v.x * cosA - v.z * sinA, v.y, v.x * sinA + v.z * cosA};
}

__device__ inline float3 rotateZ(float3 v, float angle) {
    float cosA = cosf(angle), sinA = sinf(angle);
    return {v.x * cosA - v.y * sinA, v.x * sinA + v.y * cosA, v.z};
}

__device__ inline float3 applyTransform(float3 v, float3 pos, float3 rot) {
    v = rotateX(v, rot.x);
    v = rotateY(v, rot.y);
    v = rotateZ(v, rot.z);
    return {v.x + pos.x, v.y + pos.y, v.z + pos.z};
}

__global__ void renderMultiObjectMesh(float* d_data, int width, int height, int channels,
                                      Object3D* d_objects, int numObjects,
                                      PointLight* d_lights, int numLights,
                                      Camera cam, float time) {
    int dx = threadIdx.x + blockIdx.x * blockDim.x;
    int dy = threadIdx.y + blockIdx.y * blockDim.y;

    if (dx >= width || dy >= height) return;

    int index1D = dy * width + dx;
    int index3D = index1D * channels;

    float aspect = (float)width / (float)height;
    float u = ((float)dx / (float)width) * 2.0f - 1.0f;
    float v = -(((float)dy / (float)height) * 2.0f - 1.0f);
    u = u * aspect;

    float3 rayO;
    float3 rayD;

    if (cam.isOrthographic) {
        rayD = {0.0f, 0.0f, 1.0f};
        float scale = cam.orthoSize;
        rayO = {cam.position.x + (u * scale), cam.position.y + (v * scale / aspect), cam.position.z};

        rayD = rotateX(rayD, cam.rotation.x);
        rayD = rotateY(rayD, cam.rotation.y);
        rayD = rotateZ(rayD, cam.rotation.z);
    } else {
        rayO = cam.position;
        rayD = normalizeVec({u, v, 1.0f});

        rayD = rotateX(rayD, cam.rotation.x);
        rayD = rotateY(rayD, cam.rotation.y);
        rayD = rotateZ(rayD, cam.rotation.z);
    }

    float closest_t = 999999.0f;
    bool hit = false;
    int hit_obj_idx = -1;
    int hit_tri_idx = -1;

    for (int objIdx = 0; objIdx < numObjects; objIdx++) {
        Object3D obj = d_objects[objIdx];

        float3 localRayO = sub(rayO, obj.position);
        localRayO = rotateZ(localRayO, -obj.rotation.z);
        localRayO = rotateY(localRayO, -obj.rotation.y);
        localRayO = rotateX(localRayO, -obj.rotation.x);

        float3 localRayD = rotateZ(rayD, -obj.rotation.z);
        localRayD = rotateY(localRayD, -obj.rotation.y);
        localRayD = rotateX(localRayD, -obj.rotation.x);

        float3 invD = {1.0f / localRayD.x, 1.0f / localRayD.y, 1.0f / localRayD.z};

        float tx1 = (obj.aabbMin.x - localRayO.x) * invD.x;
        float tx2 = (obj.aabbMax.x - localRayO.x) * invD.x;
        float tmin = fminf(tx1, tx2);
        float tmax = fmaxf(tx1, tx2);

        float ty1 = (obj.aabbMin.y - localRayO.y) * invD.y;
        float ty2 = (obj.aabbMax.y - localRayO.y) * invD.y;
        tmin = fmaxf(tmin, fminf(ty1, ty2));
        tmax = fminf(tmax, fmaxf(ty1, ty2));

        float tz1 = (obj.aabbMin.z - localRayO.z) * invD.z;
        float tz2 = (obj.aabbMax.z - localRayO.z) * invD.z;
        tmin = fmaxf(tmin, fminf(tz1, tz2));
        tmax = fminf(tmax, fmaxf(tz1, tz2));

        if (tmax < tmin || tmax < 0.0f) continue;

        for (int triIdx = 0; triIdx < obj.numTriangles; triIdx++) {
            int3 triIndices = obj.d_indices[triIdx];

            float3 v0 = applyTransform(obj.d_vertices[triIndices.x], obj.position, obj.rotation);
            float3 v1 = applyTransform(obj.d_vertices[triIndices.y], obj.position, obj.rotation);
            float3 v2 = applyTransform(obj.d_vertices[triIndices.z], obj.position, obj.rotation);

            float current_t;
            if (intersectTriangle(rayO, rayD, v0, v1, v2, current_t)) {
                if (current_t < closest_t) {
                    closest_t = current_t;
                    hit = true;
                    hit_obj_idx = objIdx;
                    hit_tri_idx = triIdx;
                }
            }
        }
    }

    if (hit) {
        Object3D hitObj = d_objects[hit_obj_idx];
        int3 triIndices = hitObj.d_indices[hit_tri_idx];

        float3 v0 = applyTransform(hitObj.d_vertices[triIndices.x], hitObj.position, hitObj.rotation);
        float3 v1 = applyTransform(hitObj.d_vertices[triIndices.y], hitObj.position, hitObj.rotation);
        float3 v2 = applyTransform(hitObj.d_vertices[triIndices.z], hitObj.position, hitObj.rotation);

        float3 edge1 = sub(v1, v0);
        float3 edge2 = sub(v2, v0);
        float3 normal = normalizeVec(crossProduct(edge2, edge1));

        float3 hitPoint = {rayO.x + rayD.x * closest_t, rayO.y + rayD.y * closest_t, rayO.z + rayD.z * closest_t};
        float3 viewDir = normalizeVec(sub(cam.position, hitPoint));

        float total_diffuse = 0.0f;
        float total_specular = 0.0f;

        for (int l = 0; l < numLights; l++) {
            PointLight light = d_lights[l];
            float3 lightDir = normalizeVec(sub(light.position, hitPoint));

            float diff = dotProduct(normal, lightDir);
            if (diff > 0.0f) {
                total_diffuse += diff * light.intensity;

                float3 halfDir = normalizeVec({lightDir.x + viewDir.x, lightDir.y + viewDir.y, lightDir.z + viewDir.z});
                float specAngle = dotProduct(normal, halfDir);

                if (specAngle > 0.0f) {
                    float spec = powf(specAngle, hitObj.material.shininess);
                    total_specular += spec * light.intensity;
                }
            }
        }

        float final_ambient  = hitObj.material.ambient;
        float final_diffuse  = total_diffuse * hitObj.material.diffuse;
        float final_specular = total_specular * hitObj.material.specular;

        float final_light = final_ambient + final_diffuse + final_specular;
        if (final_light > 1.0f) final_light = 1.0f;

        float final_r = hitObj.material.color.x * final_light;
        float final_g = hitObj.material.color.y * final_light;
        float final_b = hitObj.material.color.z * final_light;

        if (hitObj.material.effectFlags & 1) {
            sGlow(final_r, final_g, final_b, time, hitObj.material.glowSpeed);
        }
        if (hitObj.material.effectFlags & 2) {
            sScanlines(final_r, final_g, final_b, hitPoint.y, time, hitObj.material.scanFreq, hitObj.material.scanSpeed);
        }
        if (hitObj.material.effectFlags & 4) {
            sTronGrid(final_r, final_g, final_b, hitPoint, hitObj.material.tronGridSize, hitObj.material.tronThickness);
        }
        if (hitObj.material.effectFlags & 8) {
            sRadarPing(final_r, final_g, final_b, hitPoint, hitObj.position, time, hitObj.material.radarFreq, hitObj.material.radarSpeed);
        }
        if (hitObj.material.effectFlags & 16) {
            sMatrixJitter(final_r, final_g, final_b, hitPoint, time, hitObj.material.jitterIntensity);
        }
        if (hitObj.material.effectFlags & 32) {
            sThanosSnapDissolve(final_r, final_g, final_b, hitPoint, time, hitObj.material.dissolveSpeed);
        }
        if (hitObj.material.effectFlags & 64) {
            sNegativeZone(final_r, final_g, final_b);
        }
        if (hitObj.material.effectFlags & 128) {
            sRGBDisco(final_r, final_g, final_b, time);
        }
        if (hitObj.material.effectFlags & 256) {
            sNormalDebugger(final_r, final_g, final_b, normal.x, normal.y, normal.z);
        }
        if (hitObj.material.effectFlags & 512) {
            sCelShading(final_r, final_g, final_b, final_light, hitObj.material.celBands);
        }
        if (hitObj.material.effectFlags & 1024) {
            sLinearDepthFog(final_r, final_g, final_b, closest_t, hitObj.material.fogStart, hitObj.material.fogEnd, hitObj.material.fogColor);
        }
        if (hitObj.material.effectFlags & 2048) {
            sExponentialDepthFog(final_r, final_g, final_b, closest_t, hitObj.material.fogDensity, hitObj.material.fogColor, 2);
        }
        if (hitObj.material.effectFlags & 4096) {
            sFresnelShield(final_r, final_g, final_b, normal.x, normal.y, normal.z, viewDir.x, viewDir.y, viewDir.z, hitObj.material.shieldColor, hitObj.material.rimPower, hitObj.material.rimIntensity);
        }
        if (hitObj.material.effectFlags & 16384) {
            sLidarScanner(final_r, final_g, final_b, hitPoint, rayO, time);
        }
        if (hitObj.material.effectFlags & 32768) {
            nStaticTV(final_r, final_g, final_b, hitPoint, time, hitObj.material.noiseScale);
        }
        if (hitObj.material.effectFlags & 65536) {
            sQuantumGlitch(final_r, final_g, final_b, hitPoint, time, hitObj.material.noiseScale);
        }
        if (hitObj.material.effectFlags & 131072) {
            sPureWhiteNoise(final_r, final_g, final_b, hitPoint, time, hitObj.material.noiseScale);
        }

        if (hitObj.material.effectFlags & 262144) {
            sLiquidFlow(final_r, final_g, final_b, hitPoint, time,
                        hitObj.material.liquidSpeed,
                        hitObj.material.liquidFreq);
        }
        d_data[index3D]     = fminf(final_r, 1.0f);
        d_data[index3D + 1] = fminf(final_g, 1.0f);
        d_data[index3D + 2] = fminf(final_b, 1.0f);
    }
}
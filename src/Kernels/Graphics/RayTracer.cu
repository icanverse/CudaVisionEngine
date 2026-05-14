#include "../Kernels/Graphics/RayTracer.cuh"
#include "../../include/Graphics/Types3D.cuh" // Objelerin Struct yapılarını tanıyabilmesi için

/// Rotasyon Yardımcıları
__device__ inline float3 normalizeVec(float3 v) {
    float length = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (length > 0.00001f) return {v.x / length, v.y / length, v.z / length};
    return {0.0f, 0.0f, 0.0f};
}

/// Rotasyon Matrisleri -- euler açıları

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
/// /// ///


// Bir köşeyi önce objenin açısına göre döndürür, sonra objenin uzaydaki konumuna öteler
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

    // --- GÖREV 1: Kamera ve Işın Kurulumu ---
    float aspect = (float)width / (float)height;                // Yatay Dikey Oranı
    float u = ((float)dx / (float)width) * 2.0f - 1.0f;         // u ve v için [-1,1] sıkıştırma
    float v = -(((float)dy / (float)height) * 2.0f - 1.0f);     // Kartezyen ile ekranda y artış yönü farklı olduğundan
    u = u * aspect;

    // Işının başlangıcı kamera koordinatıdır.
    float3 rayO = cam.position;

    // Işının yönü
    // :: Kameradan 1 birim uzağa izdüşüm yaratır.
    float3 rayD = normalizeVec({u, v, 1.0f});

    // Kameranın açısına göre ışınların yönünü büküyoruz
    rayD = rotateX(rayD, cam.rotation.x);   // pitch - aşağı/yukarı
    rayD = rotateY(rayD, cam.rotation.y);   // yaw - sağa/sola
    rayD = rotateZ(rayD, cam.rotation.z);   // roll - eğilme

    float closest_t = 999999.0f;    // En yakın mesafe bu olsun varsayımı
    bool hit = false;

    int hit_obj_idx = -1; // Çarpılan obje indeksi
    int hit_tri_idx = -1; // Çarpılan objenin üçgeni indeksi

    // Tüm objeler
    for (int objIdx = 0; objIdx < numObjects; objIdx++) {
        Object3D obj = d_objects[objIdx];

        // Tüm üçgenler
        for (int triIdx = 0; triIdx < obj.numTriangles; triIdx++) {
            int3 triIndices = obj.d_indices[triIdx];

            // Lokal uzaydaki modelin tüm köşelerini globale (sahneye) göre büker
            float3 v0 = applyTransform(obj.d_vertices[triIndices.x], obj.position, obj.rotation);
            float3 v1 = applyTransform(obj.d_vertices[triIndices.y], obj.position, obj.rotation);
            float3 v2 = applyTransform(obj.d_vertices[triIndices.z], obj.position, obj.rotation);

            // Işın ile üçgen kesişiyor mu :: Möller-Trumbore algoritması
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

    // Çarpışmalar için ışıkları topla ve çiz
    // --- GÖREV 3: Çarpışma Varsa Işıkları Topla ve Çiz ---
    // --- GÖREV 3: Çarpışma Varsa Işıkları Topla ve Çiz ---
    if (hit) {
        Object3D hitObj = d_objects[hit_obj_idx];
        int3 triIndices = hitObj.d_indices[hit_tri_idx];

        // Yüzey Normalini hesapla
        float3 v0 = applyTransform(hitObj.d_vertices[triIndices.x], hitObj.position, hitObj.rotation);
        float3 v1 = applyTransform(hitObj.d_vertices[triIndices.y], hitObj.position, hitObj.rotation);
        float3 v2 = applyTransform(hitObj.d_vertices[triIndices.z], hitObj.position, hitObj.rotation);

        float3 edge1 = sub(v1, v0);
        float3 edge2 = sub(v2, v0);
        float3 normal = normalizeVec(crossProduct(edge2, edge1));

        float3 hitPoint = {rayO.x + rayD.x * closest_t, rayO.y + rayD.y * closest_t, rayO.z + rayD.z * closest_t};

        // --- YENİ MATERYAL MATEMATİĞİ (BLINN-PHONG) ---
        // Kameraya giden yön vektörü (Specular hesaplaması için şarttır)
        float3 viewDir = normalizeVec(sub(cam.position, hitPoint));

        float total_diffuse = 0.0f;
        float total_specular = 0.0f; // YENİ: Parlama havuzu

        for (int l = 0; l < numLights; l++) {
            PointLight light = d_lights[l];
            float3 lightDir = normalizeVec(sub(light.position, hitPoint));

            // 1. Diffuse (Dağılan Işık - Mat Yüzeyler)
            float diff = dotProduct(normal, lightDir);
            if (diff > 0.0f) {
                total_diffuse += diff * light.intensity;

                // 2. Specular (Parlama - Metalik/Camsı Yüzeyler)
                // Halfway Vektörü: Işık yönü ile Bakış yönünün tam ortası
                float3 halfDir = normalizeVec({lightDir.x + viewDir.x, lightDir.y + viewDir.y, lightDir.z + viewDir.z});
                float specAngle = dotProduct(normal, halfDir);

                if (specAngle > 0.0f) {
                    // Parlaklık odağı (Shininess) ile üssünü alıyoruz.
                    // Üs büyüdükçe parlama noktası küçülür ve keskinleşir (Cam/Metal gibi).
                    float spec = powf(specAngle, hitObj.material.shininess);
                    total_specular += spec * light.intensity;
                }
            }
        }

        // Materyal çarpanlarını uygula
        float final_ambient  = hitObj.material.ambient;
        float final_diffuse  = total_diffuse * hitObj.material.diffuse;
        float final_specular = total_specular * hitObj.material.specular;

        // Toplam ışık şiddeti
        float final_light = final_ambient + final_diffuse + final_specular;

        // Patlamaları engelle (Maksimum 1.0f olabilir)
        if (final_light > 1.0f) final_light = 1.0f;

        // Objenin temel rengi (Albedo) ile nihai ışığı çarp ve VRAM'e yaz
        // Parlama (Specular) genellikle ışığın kendi rengindedir (beyaz), ama şimdilik objenin rengiyle harmanlıyoruz.
        d_data[index3D]     = hitObj.material.color.x * final_light; // R
        d_data[index3D + 1] = hitObj.material.color.y * final_light; // G
        d_data[index3D + 2] = hitObj.material.color.z * final_light; // B
    }
}
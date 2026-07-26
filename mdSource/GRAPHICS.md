# Graphics Modülü

Graphics sahne üretimi, GPU texture'ları ve ekrana sunumdan sorumludur.

## Sorumluluklar

- Scene ve `SceneBuilder`
- 3D renderer ve ray tracing
- Shader'lar
- Particle system
- OpenGL texture oluşturma
- CUDA–OpenGL köprüsü
- Workspace ve thumbnail için gösterilebilir texture üretme

## Texture yaklaşımı

`TextureUtility` dosyadan texture ve thumbnail üretir:

```cpp
auto texture = TextureUtility::LoadThumbnailFromFile(
    path,
    256,
    256,
    originalWidth,
    originalHeight
);
```

Thread ayrımı önemlidir:

- `LoadResizedPixels()` CPU/STB tarafında çalışabilir.
- `CreateTextureFromPixels()` aktif OpenGL context bulunan render thread'inde
  çağrılmalıdır.

UI texture belleğinin sahibi olmamalı; Graphics tarafından verilen texture
kimliğini göstermelidir.

## Görüntü kopyaları

Bir görsel için önerilen üç temsil:

1. Tam çözünürlüklü kaynak
2. Workspace/filtre önizleme proxy'si
3. Thumbnail

Her filtre paneli için yeni bir tam çözünürlüklü kopya tutulmaz. Filtre
ayarları sürüklenirken ortak proxy işlenir; nihai sonuç tam çözünürlükte
hesaplanır.

## Graphics'e konulmaması gerekenler

- Görüntü dosyasının format decode ayrıntıları
- Vision filtre kernel'ları
- ImGui panel yerleşimleri
- Core bellek sınıflarının implementasyonu


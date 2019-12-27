#include <math.h>
#include <pthread.h>
#include <time.h>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include "sys/sysctl.h"

typedef char s8;
typedef short s16;
typedef int s32;
typedef long long int s64;
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long int u64;
typedef float f32;
typedef double f64;

union vec3 {
  vec3() : x(0.f), y(0.f), z(0.f) {}
  vec3(f32 x, f32 y, f32 z) : x(x), y(y), z(z) {}
  struct {
    f32 x, y, z;
  };
  struct {
    f32 r, g, b;
  };
};

vec3 operator+(const vec3 &a, const vec3 &b) {
  return vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}
vec3 operator-(const vec3 &a) { return vec3(-a.x, -a.y, -a.z); }

vec3 operator-(const vec3 &a, const vec3 &b) {
  return vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

vec3 operator*(const vec3 &a, s32 c) { return vec3(c * a.x, c * a.y, c * a.z); }
vec3 operator*(const vec3 &a, f32 c) { return vec3(c * a.x, c * a.y, c * a.z); }
vec3 operator*(s32 c, const vec3 &a) { return vec3(c * a.x, c * a.y, c * a.z); }
vec3 operator*(f32 c, const vec3 &a) { return vec3(c * a.x, c * a.y, c * a.z); }

f32 clamp(f32 v, f32 min, f32 max) {
  if(v < min)
    return min;
  if(v > max)
    return max;
  return v;
}

// dot(A, B) == 0 if A, B are perpendicular
// geometrical interpretation: length of projection of A onto normalized(B).
f32 dot(const vec3 &a, const vec3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

// result is vec3 C penperdicular to A and B.
// right hand rule: thumb points in C
vec3 cross(const vec3 &a, const vec3 &b) {
  return vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
              a.x * b.y - a.y * b.x);
}

f32 length(const vec3 &a) { return sqrt(a.x * a.x + a.y * a.y + a.z * a.z); }

vec3 normalize(const vec3 &a) {
  f32 l = length(a);
  return vec3(a.x / l, a.y / l, a.z / l);
}

struct Preset {
  const char *name;
  u32 image_width;
  u32 image_height;
  u32 rays_per_pixel;
  u32 max_bounce_count;
};

Preset presets[] = {{"low", 854, 480, 8, 4},
                    {"medium", 1280, 720, 128, 16},
                    {"high", 1280, 720, 256, 16},
                    {"ultra", 1280, 720, 512, 32}};
Preset default_preset = presets[0];

s32 ColorToRgba(vec3 c, f32 alpha) {
  return u8(255.f * c.b) | (u8(255.f * c.g) << 8) | (u8(255.f * c.r) << 16) |
         (u8(255.f * alpha) << 24);
}

// maps linear color from <0, 1> into sRGB <0, 1>.
f32 LinearTosRGB(f32 l) {
  l = clamp(l, 0.f, 1.f);
  f32 s;
  if(l <= 0.0031308) {
    s = l * 12.92;
  } else {
    s = 1.055f * pow(l, 1.f / 2.4f) - 0.055f;
  }
  return s;
}

// todo(kstasik): change Color to vec3, do not have alpha here? what about di
s32 LinearColorTosRGBA(vec3 c, f32 alpha) {
  return u8(255.f * LinearTosRGB(c.b)) | (u8(255.f * LinearTosRGB(c.g)) << 8) |
         (u8(255.f * LinearTosRGB(c.r)) << 16) | (u8(255.f * alpha) << 24);
}

template <size_t SIZE, class T> inline size_t sizeofcarray(T (&arr)[SIZE]) {
  return SIZE;
}

// pixel format: 0xAARRGGBB
bool write_bmp(const char *filename, const s32 width, const s32 height,
               s32 *pixeldata) {
#pragma pack(push, 1)
  struct bmp_header {
    u16 signature;
    u32 file_size;
    u16 reserved0, reserved1;
    u32 imagedata_offset;
  };

  struct bmpinfo_header {
    u32 header_size;
    s32 width;
    s32 height;
    u16 num_planes;
    u16 bits_per_pixel;
    u32 compression;
    u32 image_size;
    s32 xpixels_per_meter;
    s32 ypixels_per_meter;
    u32 num_palette_colors;
    u32 important_colors;
  };

  struct header {
    header(s32 w, s32 h) {
      head.signature = 0x4D42;
      head.file_size = sizeof(header);
      head.imagedata_offset = sizeof(header);
      info.header_size = sizeof(bmpinfo_header);
      info.width = w;
      info.height = h;
      info.num_planes = 1;
      info.bits_per_pixel = 32;
      info.compression = 0; // no compression
      info.image_size = 4 * w * h;
      info.xpixels_per_meter = 4096;
      info.ypixels_per_meter = 4096;
      info.num_palette_colors = 0;
      info.important_colors = 0;
    }

    bmp_header head;
    bmpinfo_header info;
    u32 red_bitmask;
    u32 green_bitmask;
    u32 blue_bitmask;
    u32 alpha_bitmask;
  };
#pragma pack(pop)

  static_assert(sizeof(bmp_header) == 14, "");
  static_assert(sizeof(bmpinfo_header) == 40, "");

  FILE *f = fopen(filename, "wb+");
  if(!f) {
    return -1;
  }

  header h(width, height);
  size_t written =
      fwrite(&h, sizeof(header), 1, f); // todo(kstasik): written unused
  const s32 pixeldata_size = width * height * sizeof(s32);
  written = fwrite(pixeldata, pixeldata_size, 1, f);
  if(written != pixeldata_size) {
    fclose(f);
    return false;
  }

  fclose(f);
  return true;
}

struct Material {
  f32 roughness;    // 1 is rough, 0 is pure specular
  f32 opacity;      // 0 opaque, 1 transparent
  f32 reflectivity; // 0 only albedo, 1 only reflection
  vec3 albedo;
  vec3 fresnel0;
};

struct Sphere {
  vec3 position;
  f32 radius;
  u32 material;
};

struct Plane {
  vec3 normal;
  f32 d;
  u32 material;
};

// fresnel f0 values.
namespace f0 {
const vec3 gold(1.f, 0.782f, 0.344f);
const vec3 zink(0.664f, 0.824f, 0.850f);
const vec3 chromium(0.549f, 0.556f, 0.554f);
const vec3 copper(0.955f, 0.638f, 0.538f);
const vec3 water(0.02f, 0.02f, 0.02f);
const vec3 plastic(0.04f, 0.04f, 0.04f);
const vec3 diamond(0.15f, 0.15f, 0.15f);
} // namespace f0

f32 schlick_approximation(f32 f0, vec3 l, vec3 n) {
  return f0 + (1.f - f0) * (1.f - pow(dot(l, n), 5));
}

// returns percentage of light that get reflected at a given light angle
// todo(kstask): change n to h? surface normal
vec3 fresnel(vec3 ior, vec3 l, vec3 n) {
  f32 d = dot(l, n);
  f32 p = pow(1.f - d, 5);
  vec3 var(ior.x + (1.f - ior.x) * p, ior.y + (1.f - ior.y) * p,
           ior.z + (1.f - ior.z) * p);
  var.x = clamp(var.x, 0.f, 1.f);
  var.y = clamp(var.y, 0.f, 1.f);
  var.z = clamp(var.z, 0.f, 1.f);
  return var;
}

struct World {
  World(int count) {
    sphere_count = count * count;
    sphere = new Sphere[sphere_count];

    plane_count = 1;
    plane = new Plane[plane_count];
    plane[0] = {vec3(0.f, 0.f, 1.f), 0.f, 1};

    material_count = (count * count) + 2;
    material = new Material[material_count];
    material[0].roughness = 0.f;
    material[0].opacity = 0.f;
    material[0].reflectivity = 1.f;
    material[0].albedo = vec3(1.f, 1.f, 1.f);
    material[0].fresnel0 = f0::gold; // light

    // plane
    material[1].roughness = 0.01f;
    material[1].opacity = 0.f;
    material[1].reflectivity = 0.95f;
    // material[1].albedo = vec3(0.7f, 0.1f, 0.2f);
    material[1].albedo = vec3(0.25f, 0.52f, 0.95f);
    material[1].fresnel0 = f0::water;

    f32 spacing = 2.f;

    for(u32 x = 0; x < count; ++x) {
      f32 factor_x = (f32)x / (f32)(count - 1);
      f32 bifactor_x = -1.f + 2.f * factor_x;
      f32 pos_x = bifactor_x * (count * spacing);
      for(u32 y = 0; y < count; ++y) {
        f32 factor_y = (f32)y / (f32)(count - 1);
        f32 bifactor_y = -1.f + 2.f * factor_y;
        f32 pos_y = bifactor_y * (count * spacing);
        u32 i = count * x + y;
        sphere[i] = {vec3(pos_x, pos_y, 2.f), 2.f, 2 + i};
        material[2 + i].roughness = 1.f * factor_x;
        material[2 + i].opacity = 0.0f;
        material[2 + i].reflectivity = 0.95f;
        material[2 + i].albedo = vec3(1.f, 1.f, 1.f);
        material[2 + i].fresnel0 =
            vec3(1.f - factor_y, 1.f - factor_y, 1.f - factor_y);
      }
    }
  }

  World() {
    sphere_count = 5;
    sphere = new Sphere[sphere_count];
    sphere[0] = {vec3(0.f, 0.f, 2.f), 2.f, 2};
    sphere[1] = {vec3(1.7f, 5.f, 2.f), 2.f, 3};
    sphere[2] = {vec3(-15.f, 29.f, 2.f), 3.f, 4};
    sphere[3] = {vec3(5.f, -3.f, 1.5f), 1.f, 5};
    sphere[4] = {vec3(-1.f, -10.f, 0.5f), 0.25f, 6};

    plane_count = 1;
    plane = new Plane[plane_count];
    plane[0] = {vec3(0.f, 0.f, 1.f), 0.f, 1};

    material_count = 7;
    material = new Material[material_count];
    material[0].roughness = 0.f;
    material[0].opacity = 0.f;
    material[0].reflectivity = 1.f;
    material[0].albedo = vec3(1.f, 1.f, 1.f);
    material[0].fresnel0 = f0::gold; // light

    // plane
    material[1].roughness = 0.01f;
    material[1].opacity = 0.f;
    material[1].reflectivity = 0.95f;
    // material[1].albedo = vec3(0.7f, 0.1f, 0.2f);
    material[1].albedo = vec3(0.25f, 0.52f, 0.95f);
    material[1].fresnel0 = f0::water;

    material[2].roughness = 0.01f;
    material[2].opacity = 0.87f;
    material[2].reflectivity = 0.1f;
    material[2].albedo = vec3(0.6f, 0.4f, 0.7f);
    material[2].fresnel0 = f0::chromium;

    material[3].roughness = 0.2f;
    material[3].opacity = 0.f;
    material[3].reflectivity = 0.1f;
    material[3].albedo = vec3(0.1f, 0.8f, 1.f);
    material[3].fresnel0 = f0::plastic;

    material[4].roughness = 0.01f;
    material[4].opacity = 0.f;
    material[4].reflectivity = 0.1f;
    material[4].albedo = vec3(0.5f, 0.9f, 0.5f);
    material[4].fresnel0 = f0::plastic;

    material[5].roughness = 0.5f;
    material[5].opacity = 0.f;
    material[5].reflectivity = 0.2f;
    material[5].albedo = vec3(1.f, 0.2f, 0.1f);
    material[5].fresnel0 = f0::plastic;

    material[6].roughness = 0.5f;
    material[6].opacity = 0.f;
    material[6].reflectivity = 0.1f;
    material[6].albedo = vec3(0.1f, 0.8f, 0.95f);
    material[6].fresnel0 = f0::plastic;
  }
  ~World() {
    delete[] sphere;
    delete[] plane;
    delete[] material;
  }
  Sphere *sphere;
  u32 sphere_count;
  Plane *plane;
  u32 plane_count;
  Material *material;
  u32 material_count;
};

World world;

bool raycast(World &world, vec3 ray_origin, vec3 ray_dir, vec3 *hit_point,
             vec3 *hit_normal, u32 *hit_material) {
  bool hit = false;
  f32 distance = std::numeric_limits<f32>::max();

  for(u32 i = 0; i < world.plane_count; ++i) {
    auto &plane = world.plane[i];
    f32 tolerance = 0.0001f;
    f32 denom = dot(plane.normal, ray_dir);
    if(abs(denom) < tolerance) {
      continue; // ray parallel to plane
    }

    f32 t =
        ((plane.normal.z * plane.d) - dot(plane.normal, ray_origin)) / denom;
    if(t < tolerance) {
      continue; // hit behind camera
    }
    if(t < distance) {
      distance = t;
      *hit_material = plane.material;
      *hit_point = ray_origin + t * ray_dir;
      *hit_normal = plane.normal;
      hit = true;
    }
  }

  for(u32 i = 0; i < world.sphere_count; ++i) {
    auto &sphere = world.sphere[i];
    f32 tolerance = 0.0001f;

    vec3 sphere_relative_ray_origin = ray_origin - sphere.position;
    f32 a = dot(ray_dir, ray_dir);
    f32 b = 2.f * dot(ray_dir, sphere_relative_ray_origin);
    f32 c = dot(sphere_relative_ray_origin, sphere_relative_ray_origin) -
            (sphere.radius * sphere.radius);

    f32 denom = 2.f * a;
    f32 root_term = sqrt((b * b) - (4.f * a * c));
    f32 t0 = (-b + root_term) / denom;
    f32 t1 = (-b - root_term) / denom;

    if(abs(root_term) < tolerance) {
      continue; // no hit
    }

    f32 t = t0;
    if(t1 > 0 && t1 < t0) {
      t = t1;
    }
    if(t < tolerance) {
      continue; // hit behind origin
    }
    if(t < distance) {
      distance = t;
      *hit_point = ray_origin + t * ray_dir;
      *hit_normal = normalize(*hit_point - sphere.position);
      *hit_material = sphere.material;
      hit = true;
    }
  }
  return hit;
}

// todo(kstasik): do we need those?
vec3 cmul(vec3 a, vec3 b) { return vec3(a.x * b.x, a.y * b.y, a.z * b.z); }
vec3 &operator+=(vec3 &a, const vec3 &b) {
  a = a + b;
  return a;
}

f32 Random() { return (f32)std::rand() / (f32)RAND_MAX; }

bool generate_filename(char *buffer, u32 length, const char *postfix) {
  time_t t = time(nullptr);
  if(t == -1) {
    return false;
  }
  struct tm *date = gmtime(&t);
  int res = snprintf(buffer, length, "../images/image_%d-%d-%d_%d-%s.bmp",
                     1900 + date->tm_year, 1 + date->tm_mon, date->tm_mday,
                     date->tm_hour * 60 * 60 + date->tm_min * 60 + date->tm_sec,
                     postfix);
  return res > 0 && res < length;
}

vec3 refract(vec3 l, vec3 n, f32 ior) {
  f32 dp = dot(l, n);
  f32 etai = 1.f;
  f32 etat = ior;
  if(dp < 0.f) {
    // outside
    dp = -dp;
  } else {
    n = -n;
    std::swap(etai, etat);
  }
  f32 eta = etai / etat;
  f32 k = 1.f - (eta * eta) * (1.f - (dp * dp));
  if(k < 0.f) {
    return vec3(0.f, 0.f, 0.f);
  }
  vec3 dir = eta * l + (eta * dp - sqrtf(k)) * n;
  normalize(dir);
  return dir;
}

vec3 trace(vec3 ray_origin, vec3 ray_dir, vec3 attenuation, u32 depth,
           bool first = false) {
  vec3 hit_point, hit_normal;
  u32 hit_material;
  vec3 result(0.f, 0.f, 0.f);

  bool ray_hit = raycast(world, ray_origin, ray_dir, &hit_point, &hit_normal,
                         &hit_material);
  if(ray_hit && depth > 0) {
    Material mat = world.material[hit_material];

    f32 cosine_term = dot(-ray_dir, hit_normal);
    if(cosine_term < 0.f)
      cosine_term = 0.f;

    // split attenuation into reflection and refraction
    // vec3 reflection_attenuation = (1.f - mat.opacity) * attenuation;
    // vec3 refraction_attenuation = mat.opacity * attenuation;

    /*vec3 refracted_color(0.f, 0.f, 0.f);
    if(mat.opacity > 0.f) {
      vec3 refract_dir = refract(normalize(ray_dir), hit_normal, 1.02);
      // refraction_attenuation = cmul(refraction_attenuation, cosine_term *
      // transmitance);
      f32 bias = 0.0001f;
      vec3 color = trace(hit_point + refract_dir * bias, refract_dir,
                         refraction_attenuation, depth - 1);

      // refracted_color = color;
      refracted_color = mat.opacity * color;
      // refracted_color = cmul(transmitance, color);
    }*/

    vec3 bounce = ray_dir - 2.f * dot(ray_dir, hit_normal) * hit_normal;
    vec3 bounce_rand(-1.f + 2.f * Random(), -1.f + 2.f * Random(),
                     -1.f + 2.f * Random());
    vec3 specular_ray_dir = bounce + mat.roughness * bounce_rand;

    vec3 h = normalize(-ray_dir + specular_ray_dir);

    // vec3 albedo_att = reflection_attenuation * (1.f - mat.reflectivity);
    // vec3 reflect_att = reflection_attenuation * mat.reflectivity;

    vec3 reflect_att =
        fresnel(mat.fresnel0, normalize(-ray_dir), /*hit_normal*/ h);
    vec3 albedo_att = vec3(1.f, 1.f, 1.f) - reflect_att;

    vec3 specular = trace(hit_point, specular_ray_dir, reflect_att, depth - 1);

    vec3 albedo = cmul(albedo_att, mat.albedo);
    vec3 speccc = cmul(reflect_att, specular);
    result = cosine_term * (albedo + speccc);
  } else {
    Material mat = world.material[0]; // sky material
    result += cmul(attenuation, mat.albedo);
  }
  return result;
}

vec3 compute_color(vec3 ray_origin, vec3 ray_dir, u32 max_bounce_count) {
  vec3 attenuation(1.f, 1.f, 1.f);
  return trace(ray_origin, ray_dir, attenuation, max_bounce_count, true);
}

/** coordinates
    z
    |
    |
    * - - y
   /
  /
 x
 */

Preset parse_args(int argc, char **argv) {
  Preset preset = default_preset;
  for(u32 i = 0; i < argc; ++i) {
    if(strcmp(argv[i], "-p") == 0) {
      if(++i < argc) {
        for(auto &p : presets) {
          if(strcmp(argv[i], p.name) == 0) {
            preset = p;
          }
        }
      }
    }
  }
  return preset;
}

struct Work {
  u32 y_from;
  u32 y_to;
  u32 x_from;
  u32 x_to;
  u32 rays_per_pixel;
  s32 *image;
  Preset preset;
  vec3 camera_position;
  vec3 look_at;
};

std::atomic<int> g_pixelcounter(0);

void *do_work(void *arg) {
  Work *work = static_cast<Work *>(arg);
  const u32 image_height = work->preset.image_height;
  const u32 image_width = work->preset.image_width;
  const u32 rays_per_pixel = work->preset.rays_per_pixel;
  const f32 half_pixel_width = 0.5f / image_width;
  const f32 half_pixel_height = 0.5f / image_height;
  const f32 frame_dist = 1.f;
  f32 image_aspect_ratio = (f32)image_width / (f32)image_height;
  const f32 frame_width = 1.f * image_aspect_ratio;
  const f32 frame_height = 1.f;

  const f32 frame_half_w = frame_width * 0.5f;
  const f32 frame_half_h = frame_height * 0.5f;

  const s32 pixelcount = image_width * image_height;

  const vec3 camera_position = work->camera_position;
  const vec3 look_at = work->look_at;

  // camera points at -z axis
  const vec3 camera_z = normalize(camera_position - look_at);
  const vec3 camera_x = normalize(cross(vec3(0.f, 0.f, 1.f), camera_z));
  const vec3 camera_y = normalize(cross(camera_z, camera_x));

  vec3 ray_origin = camera_position;
  vec3 film_center = ray_origin - frame_dist * camera_z;

  for(u32 y = work->y_from; y < work->y_to; ++y) {
    f32 film_y = -1.f + 2.f * ((f32)y / (f32)image_height);
    for(u32 x = work->x_from; x < work->x_to; ++x) {
      f32 film_x = -1.f + 2.f * ((f32)x / (f32)image_width);
      u32 pixel = y * image_width + x;

      g_pixelcounter++;

      vec3 color(0.f, 0.f, 0.f);
      f32 contribution = 1.f / (f32)rays_per_pixel;

      for(u32 ray_index = 0; ray_index < rays_per_pixel; ++ray_index) {
        f32 pix_offset_x = half_pixel_width * (-1.f + 2.f * Random());
        f32 pix_offset_y = half_pixel_height * (-1.f + 2.f * Random());
        vec3 film_xoffset = (film_x + pix_offset_x) * frame_half_w * camera_x;
        vec3 film_yoffset = (film_y + pix_offset_y) * frame_half_h * camera_y;
        vec3 ray_destination = film_center + film_xoffset + film_yoffset;
        vec3 ray_dir = /*normalize*/ (ray_destination - ray_origin);
        color += contribution * compute_color(ray_origin, ray_dir,
                                              work->preset.max_bounce_count);
      }

      work->image[pixel] = LinearColorTosRGBA(color, 1.f);
    }
    printf("\rray tracing... %d%%",
           (u32)(((f32)g_pixelcounter / (f32)pixelcount) * 100.f));
    fflush(stdout);
  }
  printf(" thread ended\n");
  return nullptr;
}

int main(int argc, char **argv) {
  Preset preset = parse_args(argc, argv);

  printf("starting with preset: '%s': %dx%d image, rpp: %d\n", preset.name,
         preset.image_width, preset.image_height, preset.rays_per_pixel);

  const s32 image_width = preset.image_width;
  const s32 image_height = preset.image_height;

  const vec3 cameras[] = {vec3(0.f, -25.f, 15.f), vec3(10.f, -15.f, 7.f),
                          vec3(6.f, -10.f, 8.f), vec3(-1.f, -14.f, 20.f)};
  // const vec3 cameras[] = {vec3(0.f, -25.f, 25.f), vec3(0.f, -25.f, 20.f),
  //                       vec3(0.f, -25.f, 15.f), vec3(0.f, -25.f, 10.f),
  //                      vec3(0.f, -25.f, 5.f),  vec3(0.f, -25.f, 1.f)};
  u32 num_cameras = (u32)sizeofcarray(cameras);

  s32 thread_count = 1;
  size_t thread_count_len = sizeof(thread_count);
  sysctlbyname("hw.logicalcpu", &thread_count, &thread_count_len, NULL, 0);
  printf("thread count: %d\n", thread_count);

  pthread_t *thread_ids = new pthread_t[thread_count];
  Work *work = new Work[thread_count];

  // process each camera
  for(u32 cam_index = 0; cam_index < num_cameras; ++cam_index) {
    printf("processing camera %u/%u..\n", cam_index + 1, num_cameras);

    const vec3 camera_position = cameras[cam_index];
    const vec3 look_at(0.f, 0.f, 0.f);

    const s32 pixelcount = image_width * image_height;
    s32 *image = new s32[pixelcount];

    clock_t cpu_start = clock();
    struct timespec wall_start;
    clock_gettime(CLOCK_MONOTONIC, &wall_start);

    g_pixelcounter = 0;

    // start and setup threads
    for(u32 thread_index = 0; thread_index < thread_count; ++thread_index) {
      Work &w = work[thread_index];
      w.y_from = thread_index * (image_height / thread_count);
      w.y_to = w.y_from + (image_height / thread_count);
      w.x_from = 0;
      w.x_to = image_width;
      w.image = image;
      w.preset = preset;
      w.camera_position = camera_position;
      w.look_at = look_at;

      s32 ret =
          pthread_create(&thread_ids[thread_index], NULL, do_work, (void *)&w);
      if(ret != 0) {
        printf("failed to create thread %d \n", ret);
        exit(-1);
      }
    }

    // wait for threads to finish
    for(u32 thread_index = 0; thread_index < thread_count; ++thread_index) {
      pthread_join(thread_ids[thread_index], NULL);
    }

    clock_t cpu_diff = clock() - cpu_start;
    struct timespec wall_end;
    clock_gettime(CLOCK_MONOTONIC, &wall_end);

    f64 wall_diff_sec = (wall_end.tv_sec + 1.0e-9 * wall_end.tv_nsec) -
                        (wall_start.tv_sec + 1.0e-9 * wall_start.tv_nsec);

    f32 cpu_msec = 1000.f * cpu_diff / CLOCKS_PER_SEC;
    printf("finished processing image. cputime %.2fms, walltime: %.2fms\n",
           cpu_msec, wall_diff_sec * 1000.f);

    write_bmp("test.bmp", image_width, image_height, image);
    char namebuffer[256];
    if(generate_filename(namebuffer, (u32)sizeofcarray(namebuffer),
                         preset.name)) {
      write_bmp(namebuffer, image_width, image_height, image);
    }
    delete[] image;
  }

  delete[] thread_ids;
  delete[] work;

  return 0;
}

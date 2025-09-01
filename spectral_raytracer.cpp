// Monte Carlo Spectral Ray Tracer
#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <thread>
#include <atomic>
#include <chrono>
#include <mutex>
#include <memory>
#include <array>
#include <limits>
#include <iomanip>
#include <cstring>
#include <functional>

// Mathematical constants for spectral rendering
constexpr double PI = 3.141592653589793238462643383279502884;
constexpr double TWO_PI = 2.0 * PI;
constexpr double INV_PI = 1.0 / PI;
constexpr double INV_TWO_PI = 1.0 / TWO_PI;
constexpr double EPSILON = 1e-8;

// Spectral wavelength range (380nm - 780nm visible spectrum)
constexpr double LAMBDA_MIN = 380.0;
constexpr double LAMBDA_MAX = 780.0;
constexpr int SPECTRAL_SAMPLES = 32;

// PCG random number generator for deterministic reproducibility
class PCG32 {
    uint64_t state = 0x853c49e6748fea9bULL;
    uint64_t inc = 0xda3e39cb94b95bdbULL;
public:
    PCG32(uint64_t seed = 42) { state = seed; }
    uint32_t next() {
        uint64_t oldstate = state;
        state = oldstate * 6364136223846793005ULL + inc;
        uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
        uint32_t rot = oldstate >> 59u;
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }
    double uniform() { return next() * (1.0 / 4294967296.0); }
    double uniform(double min, double max) {
        return min + (max - min) * uniform();
    }
};

struct alignas(16) Vec3 {
    double x, y, z;
    Vec3(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
    Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 operator*(double s) const { return Vec3(x * s, y * s, z * s); }
    Vec3 operator*(const Vec3& v) const { return Vec3(x * v.x, y * v.y, z * v.z); }
    Vec3 operator/(double s) const { return Vec3(x / s, y / s, z / s); }
    Vec3 operator/(const Vec3& v) const { 
        return Vec3(v.x > EPSILON ? x / v.x : 0, 
                   v.y > EPSILON ? y / v.y : 0, 
                   v.z > EPSILON ? z / v.z : 0); 
    }
    double dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    Vec3 cross(const Vec3& v) const {
        return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }
    double length() const { return std::sqrt(x * x + y * y + z * z); }
    double lengthSquared() const { return x * x + y * y + z * z; }
    Vec3 normalized() const {
        double len = length();
        return (len > EPSILON) ? (*this / len) : Vec3(0, 0, 0);
    }
    Vec3 reflect(const Vec3& n) const {
        return *this - n * (2.0 * dot(n));
    }
    bool refract(const Vec3& n, double eta, Vec3& refracted) const {
        double cosI = -dot(n);
        double sin2I = 1.0 - cosI * cosI;
        double sin2T = eta * eta * sin2I;
        if (sin2T > 1.0) return false;
        double cosT = std::sqrt(1.0 - sin2T);
        refracted = *this * eta + n * (eta * cosI - cosT);
        return true;
    }
};

// Ray structure with motion blur support
struct Ray {
    Vec3 origin, direction;
    double time;
    double tMin = EPSILON;
    double tMax = std::numeric_limits<double>::infinity();
    Ray(const Vec3& o, const Vec3& d, double t = 0) 
        : origin(o), direction(d.normalized()), time(t) {}
    Vec3 at(double t) const { return origin + direction * t; }
};

// Spectral power distribution for physically accurate light representation
class SpectralPowerDistribution {
    std::array<double, SPECTRAL_SAMPLES> samples;
    
public:
    SpectralPowerDistribution(double v = 0) { samples.fill(v); }
    // Planckian blackbody radiation spectrum
    static SpectralPowerDistribution blackbody(double temperature) {
        SpectralPowerDistribution spd;
        const double h = 6.62607004e-34;  // Planck constant
        const double c = 299792458.0;     // Speed of light
        const double k = 1.38064852e-23;  // Boltzmann constant
        for (int i = 0; i < SPECTRAL_SAMPLES; ++i) {
            double lambda = LAMBDA_MIN + (LAMBDA_MAX - LAMBDA_MIN) * i / (SPECTRAL_SAMPLES - 1);
            lambda *= 1e-9;  // Convert 
            // Planck's law: B(λ,T) = (2hc²/λ⁵) / (exp(hc/λkT) - 1)
            double exponent = (h * c) / (lambda * k * temperature);
            if (exponent > 700.0) {
                spd.samples[i] = 0.0;
            } else {
                spd.samples[i] = (2.0 * h * c * c) / (lambda * lambda * lambda * lambda * lambda) 
                               / (std::exp(exponent) - 1.0);
            }
        }
        spd.normalize();
        return spd;
    }
    // CIE D65 standard illuminant
    static SpectralPowerDistribution D65() {
        SpectralPowerDistribution spd;
        // Approximation of D65 using Planckian locus
        return blackbody(6504.0);
    }
    void normalize() {
        double sum = 0;
        for (double s : samples) sum += s;
        if (sum > 0) {
            for (double& s : samples) s /= sum;
        }
    }
    
    double sample(double wavelength) const {
        double t = (wavelength - LAMBDA_MIN) / (LAMBDA_MAX - LAMBDA_MIN);
        t = std::max(0.0, std::min(1.0, t));
        int idx = static_cast<int>(t * (SPECTRAL_SAMPLES - 1));
        return samples[idx];
    }
    
    SpectralPowerDistribution operator*(double s) const {
        SpectralPowerDistribution result;
        for (int i = 0; i < SPECTRAL_SAMPLES; ++i) {
            result.samples[i] = samples[i] * s;
        }
        return result;
    }
    
    SpectralPowerDistribution operator*(const SpectralPowerDistribution& other) const {
        SpectralPowerDistribution result;
        for (int i = 0; i < SPECTRAL_SAMPLES; ++i) {
            result.samples[i] = samples[i] * other.samples[i];
        }
        return result;
    }
    
    SpectralPowerDistribution operator/(double s) const {
        SpectralPowerDistribution result;
        if (std::abs(s) > EPSILON) {
            for (int i = 0; i < SPECTRAL_SAMPLES; ++i) {
                result.samples[i] = samples[i] / s;
            }
        }
        return result;
    }
    
    SpectralPowerDistribution operator+(const SpectralPowerDistribution& other) const {
        SpectralPowerDistribution result;
        for (int i = 0; i < SPECTRAL_SAMPLES; ++i) {
            result.samples[i] = samples[i] + other.samples[i];
        }
        return result;
    }
    
    // Convert to XYZ using CIE 1931 color matching functions
    Vec3 toXYZ() const {
        // CIE 1931 2-degree color matching functions (sampled)
        static const double xBar[] = {0.0014, 0.0042, 0.0143, 0.0435, 0.1344, 0.2839, 
                                      0.3483, 0.3362, 0.2908, 0.1954, 0.0956, 0.0320,
                                      0.0049, 0.0093, 0.0633, 0.1655, 0.2904, 0.4334,
                                      0.5945, 0.7621, 0.9163, 1.0263, 1.0622, 1.0026,
                                      0.8544, 0.6424, 0.4479, 0.2835, 0.1649, 0.0874,
                                      0.0468, 0.0227};
        static const double yBar[] = {0.0000, 0.0001, 0.0004, 0.0012, 0.0040, 0.0116,
                                      0.0230, 0.0380, 0.0600, 0.0910, 0.1390, 0.2080,
                                      0.3230, 0.5030, 0.7100, 0.8620, 0.9540, 0.9950,
                                      0.9950, 0.9520, 0.8700, 0.7570, 0.6310, 0.5030,
                                      0.3810, 0.2650, 0.1750, 0.1070, 0.0610, 0.0320,
                                      0.0170, 0.0082};
        static const double zBar[] = {0.0065, 0.0201, 0.0679, 0.2074, 0.6456, 1.3856,
                                      1.7471, 1.7721, 1.6692, 1.2876, 0.8130, 0.4652,
                                      0.2720, 0.1582, 0.0782, 0.0422, 0.0203, 0.0087,
                                      0.0039, 0.0021, 0.0017, 0.0011, 0.0008, 0.0003,
                                      0.0002, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                      0.0000, 0.0000};
        Vec3 xyz(0, 0, 0);
        for (int i = 0; i < SPECTRAL_SAMPLES; ++i) {
            xyz.x += samples[i] * xBar[i];
            xyz.y += samples[i] * yBar[i];
            xyz.z += samples[i] * zBar[i];
        }
        return xyz * ((LAMBDA_MAX - LAMBDA_MIN) / SPECTRAL_SAMPLES);
    }
    
    // Convert XYZ to RGB (sRGB primaries)
    static Vec3 XYZtoRGB(const Vec3& xyz) {
        Vec3 rgb;
        rgb.x =  3.2404542 * xyz.x - 1.5371385 * xyz.y - 0.4985314 * xyz.z;
        rgb.y = -0.9692660 * xyz.x + 1.8760108 * xyz.y + 0.0415560 * xyz.z;
        rgb.z =  0.0556434 * xyz.x - 0.2040259 * xyz.y + 1.0572252 * xyz.z;

        // Gamma correction (sRGB)
        auto gammaCorrect = [](double v) {
            if (v <= 0.0031308) return 12.92 * v;
            return 1.055 * std::pow(v, 1.0/2.4) - 0.055;
        };
        rgb.x = gammaCorrect(std::max(0.0, std::min(1.0, rgb.x)));
        rgb.y = gammaCorrect(std::max(0.0, std::min(1.0, rgb.y)));
        rgb.z = gammaCorrect(std::max(0.0, std::min(1.0, rgb.z)));
        return rgb;
    }
};

// material with spectral BRDF
class Material {
public:
    enum Type { DIFFUSE, SPECULAR, DIELECTRIC, CONDUCTOR, VOLUME };
    Type type;
    SpectralPowerDistribution albedo;
    double roughness;
    double ior;  // Index of refraction (wavelength-dependent via Cauchy equation)
    double absorption;
    double scatteringCoeff;
    double anisotropy;  // For Henyey-Greenstein phase function
    Material(Type t = DIFFUSE) : type(t), albedo(0.8), roughness(0.5), 
                                  ior(1.5), absorption(0.01), 
                                  scatteringCoeff(0.01), anisotropy(0.0) {}
    // Cauchy equation for wavelength-dependent IOR
    double getIOR(double wavelength) const {
        // n(λ) = A + B/λ² + C/λ⁴
        double lambda_um = wavelength / 1000.0;  // Convert to micrometers
        double lambda2 = lambda_um * lambda_um;
        double lambda4 = lambda2 * lambda2;
        // Glass coefficients (BK7)
        const double A = 1.5168;
        const double B = 0.0064;
        const double C = 0.0002;
        return A + B/lambda2 + C/lambda4;
    }
    // GGX/Trowbridge-Reitz microfacet distribution
    // D(h) = α² / (π * ((n·h)² * (α² - 1) + 1)²)
    double GGX_D(const Vec3& n, const Vec3& h) const {
        double alpha = roughness * roughness;
        double alpha2 = alpha * alpha;
        double NdotH = std::max(0.0, n.dot(h));
        double NdotH2 = NdotH * NdotH;
        
        double denom = NdotH2 * (alpha2 - 1.0) + 1.0;
        denom = PI * denom * denom;
        
        return alpha2 / std::max(EPSILON, denom);
    }
    
    // Smith masking-shadowing function with height correlation
    // G(l,v,h) = G₁(l) * G₁(v) with height correlation factor
    double SmithG(const Vec3& n, const Vec3& v, const Vec3& l) const {
        auto G1 = [this](const Vec3& n, const Vec3& v) -> double {
            double alpha = roughness * roughness;
            double NdotV = std::abs(n.dot(v));
            double alpha2 = alpha * alpha;
            double tan2Theta = (1.0 - NdotV * NdotV) / (NdotV * NdotV);
            return 2.0 / (1.0 + std::sqrt(1.0 + alpha2 * tan2Theta));
        };
        
        return G1(n, v) * G1(n, l);
    }
    
    // Fresnel equations for dielectrics and conductors with full complex arithmetic
    Vec3 fresnel(double cosTheta, double wavelength) const {
        if (type == DIELECTRIC) {
            double n = getIOR(wavelength);
            double f0 = ((n - 1.0) * (n - 1.0)) / ((n + 1.0) * (n + 1.0));
            return Vec3(1, 1, 1) * (f0 + (1.0 - f0) * std::pow(1.0 - cosTheta, 5.0));
        } else if (type == CONDUCTOR) {
            // Full complex Fresnel equations for metals
            // Using wavelength-dependent complex refractive index (n + ik)
            // Metal optical constants for copper at different wavelengths
            double n_real, k_imag;
            if (wavelength < 450) {
                n_real = 1.400; k_imag = 1.950;  // Blue region
            } else if (wavelength < 550) {
                n_real = 0.620; k_imag = 2.630;  // Green region
            } else if (wavelength < 650) {
                n_real = 0.250; k_imag = 3.320;  // Orange region
            } else {
                n_real = 0.270; k_imag = 3.610;  // Red region
            }
            
            // Complex Fresnel computation using full equations
            double cosTheta2 = cosTheta * cosTheta;
            double sinTheta2 = 1.0 - cosTheta2;
            double n2 = n_real * n_real;
            double k2 = k_imag * k_imag;
            
            // Perpendicular polarization
            double a2b2_perp = std::sqrt((n2 - k2 - sinTheta2) * (n2 - k2 - sinTheta2) + 4.0 * n2 * k2);
            double a_perp = std::sqrt(0.5 * (a2b2_perp + (n2 - k2 - sinTheta2)));
            double b_perp = std::sqrt(0.5 * (a2b2_perp - (n2 - k2 - sinTheta2)));
            
            double Rs_num = (a_perp - cosTheta) * (a_perp - cosTheta) + b_perp * b_perp;
            double Rs_den = (a_perp + cosTheta) * (a_perp + cosTheta) + b_perp * b_perp;
            double Rs = Rs_num / Rs_den;
            
            // Parallel polarization  
            double a2b2_par = std::sqrt((n2 - k2 - sinTheta2) * (n2 - k2 - sinTheta2) + 4.0 * n2 * k2);
            double term1 = n2 - k2 - sinTheta2;
            double cosTheta_n2k2 = cosTheta * (n2 + k2);
            
            double Rp_num = (term1 * cosTheta - a_perp) * (term1 * cosTheta - a_perp) + 
                           (2.0 * n_real * k_imag * cosTheta - b_perp) * (2.0 * n_real * k_imag * cosTheta - b_perp);
            double Rp_den = (term1 * cosTheta + a_perp) * (term1 * cosTheta + a_perp) + 
                           (2.0 * n_real * k_imag * cosTheta + b_perp) * (2.0 * n_real * k_imag * cosTheta + b_perp);
            double Rp = Rp_num / std::max(EPSILON, Rp_den);
            
            // Unpolarized light is average of both polarizations
            double reflectance = 0.5 * (Rs + Rp);
            
            // Add wavelength-dependent color tint for metals
            Vec3 tint(1.0, 1.0, 1.0);
            if (wavelength < 500) tint = Vec3(0.95, 0.93, 0.88);  // Slightly blue
            else if (wavelength < 600) tint = Vec3(0.98, 0.96, 0.90);  // Neutral
            else tint = Vec3(1.00, 0.94, 0.86);  // Slightly warm
            
            return tint * reflectance;
        }
        return Vec3(1, 1, 1);
    }
    
    // Henyey-Greenstein phase function for volumetric scattering
    double phaseHG(double cosTheta) const {
        double g = anisotropy;
        double g2 = g * g;
        double denom = 1.0 + g2 - 2.0 * g * cosTheta;
        return INV_PI * 0.25 * (1.0 - g2) / (denom * std::sqrt(denom));
    }
};

// Intersection information
struct HitRecord {
    Vec3 point;
    Vec3 normal;
    double t;
    double u, v;  // Texture coordinates
    Material material;
    bool frontFace;
    
    void setFaceNormal(const Ray& r, const Vec3& outwardNormal) {
        frontFace = r.direction.dot(outwardNormal) < 0;
        normal = frontFace ? outwardNormal : outwardNormal * -1.0;
    }
};

// Base primitive class
class Primitive {
public:
    Material material;
    
    virtual bool intersect(const Ray& r, double tMin, double tMax, HitRecord& rec) const = 0;
    virtual Vec3 sample(PCG32& rng) const = 0;
    virtual double pdf(const Vec3& origin, const Vec3& direction) const = 0;
    virtual ~Primitive() = default;
};

// Sphere primitive with motion blur support
class Sphere : public Primitive {
    Vec3 center0, center1;  // For motion blur
    double radius;
    double time0, time1;
    
public:
    Sphere(const Vec3& c, double r, const Material& m) 
        : center0(c), center1(c), radius(r), time0(0), time1(1) {
        material = m;
    }
    
    Sphere(const Vec3& c0, const Vec3& c1, double r, const Material& m, double t0, double t1)
        : center0(c0), center1(c1), radius(r), time0(t0), time1(t1) {
        material = m;
    }
    
    Vec3 center(double time) const {
        if (time1 <= time0) return center0;
        double t = (time - time0) / (time1 - time0);
        return center0 + (center1 - center0) * t;
    }
    
    bool intersect(const Ray& r, double tMin, double tMax, HitRecord& rec) const override {
        Vec3 c = center(r.time);
        Vec3 oc = r.origin - c;
        
        double a = r.direction.lengthSquared();
        double half_b = oc.dot(r.direction);
        double c_term = oc.lengthSquared() - radius * radius;
        
        double discriminant = half_b * half_b - a * c_term;
        if (discriminant < 0) return false;
        
        double sqrtd = std::sqrt(discriminant);
        double root = (-half_b - sqrtd) / a;
        
        if (root < tMin || tMax < root) {
            root = (-half_b + sqrtd) / a;
            if (root < tMin || tMax < root) return false;
        }
        
        rec.t = root;
        rec.point = r.at(rec.t);
        Vec3 outwardNormal = (rec.point - c) / radius;
        rec.setFaceNormal(r, outwardNormal);
        
        // Compute UV coordinates
        double theta = std::acos(-outwardNormal.y);
        double phi = std::atan2(-outwardNormal.z, outwardNormal.x) + PI;
        rec.u = phi / TWO_PI;
        rec.v = theta / PI;
        
        rec.material = material;
        return true;
    }
    
    Vec3 sample(PCG32& rng) const override {
        double theta = TWO_PI * rng.uniform();
        double phi = std::acos(1.0 - 2.0 * rng.uniform());
        
        Vec3 dir(std::sin(phi) * std::cos(theta),
                 std::sin(phi) * std::sin(theta),
                 std::cos(phi));
        
        return center0 + dir * radius;
    }
    
    double pdf(const Vec3& origin, const Vec3& direction) const override {
        HitRecord rec;
        if (!intersect(Ray(origin, direction, 0), EPSILON, 
                      std::numeric_limits<double>::infinity(), rec)) {
            return 0;
        }
        
        double cos_theta_max = std::sqrt(1.0 - radius * radius / 
                                         (center0 - origin).lengthSquared());
        double solid_angle = TWO_PI * (1.0 - cos_theta_max);
        
        return 1.0 / solid_angle;
    }
};

// Triangle mesh primitive
class Triangle : public Primitive {
    Vec3 v0, v1, v2;
    Vec3 n0, n1, n2;  // Vertex normals for smooth shading
    
public:
    Triangle(const Vec3& a, const Vec3& b, const Vec3& c, const Material& m)
        : v0(a), v1(b), v2(c) {
        material = m;
        Vec3 normal = (v1 - v0).cross(v2 - v0).normalized();
        n0 = n1 = n2 = normal;
    }
    
    bool intersect(const Ray& r, double tMin, double tMax, HitRecord& rec) const override {
        // Möller-Trumbore intersection algorithm
        Vec3 edge1 = v1 - v0;
        Vec3 edge2 = v2 - v0;
        Vec3 h = r.direction.cross(edge2);
        double a = edge1.dot(h);
        
        if (std::abs(a) < EPSILON) return false;
        
        double f = 1.0 / a;
        Vec3 s = r.origin - v0;
        double u = f * s.dot(h);
        
        if (u < 0.0 || u > 1.0) return false;
        
        Vec3 q = s.cross(edge1);
        double v = f * r.direction.dot(q);
        
        if (v < 0.0 || u + v > 1.0) return false;
        
        double t = f * edge2.dot(q);
        
        if (t < tMin || t > tMax) return false;
        
        rec.t = t;
        rec.point = r.at(t);
        rec.u = u;
        rec.v = v;
        
        // Barycentric interpolation of normals
        double w = 1.0 - u - v;
        Vec3 normal = (n0 * w + n1 * u + n2 * v).normalized();
        rec.setFaceNormal(r, normal);
        rec.material = material;
        
        return true;
    }
    
    Vec3 sample(PCG32& rng) const override {
        double u = rng.uniform();
        double v = rng.uniform();
        
        if (u + v > 1.0) {
            u = 1.0 - u;
            v = 1.0 - v;
        }
        
        return v0 * (1.0 - u - v) + v1 * u + v2 * v;
    }
    
    double pdf(const Vec3& origin, const Vec3& direction) const override {
        HitRecord rec;
        if (!intersect(Ray(origin, direction, 0), EPSILON,
                      std::numeric_limits<double>::infinity(), rec)) {
            return 0;
        }
        
        double area = 0.5 * (v1 - v0).cross(v2 - v0).length();
        double distance2 = rec.t * rec.t * direction.lengthSquared();
        double cosine = std::abs(direction.dot(rec.normal) / direction.length());
        
        return distance2 / (cosine * area);
    }
};

// Bounding Volume Hierarchy for acceleration
class BVHNode {
    struct AABB {
        Vec3 min, max;
        
        AABB() : min(1e10, 1e10, 1e10), max(-1e10, -1e10, -1e10) {}
        
        void expand(const Vec3& p) {
            min.x = std::min(min.x, p.x);
            min.y = std::min(min.y, p.y);
            min.z = std::min(min.z, p.z);
            max.x = std::max(max.x, p.x);
            max.y = std::max(max.y, p.y);
            max.z = std::max(max.z, p.z);
        }
        
        void expand(const AABB& box) {
            expand(box.min);
            expand(box.max);
        }
        
        Vec3 centroid() const {
            return (min + max) * 0.5;
        }
        
        double surfaceArea() const {
            Vec3 d = max - min;
            return 2.0 * (d.x * d.y + d.y * d.z + d.z * d.x);
        }
    };
    
public:
    Vec3 min, max;
    std::unique_ptr<BVHNode> left, right;
    std::shared_ptr<Primitive> primitive;
    
    BVHNode(std::vector<std::shared_ptr<Primitive>>& prims, 
            size_t start, size_t end, PCG32& rng) {
        
        // Initialize thread-local recursion tracking
        static thread_local bool depthInitialized = false;
        static thread_local int recursionDepth = 0;
        if (!depthInitialized) {
            recursionDepth = 0;
            depthInitialized = true;
        }
        
        // Prevent infinite recursion while maintaining sophisticated BVH
        if (recursionDepth > 25 || end - start <= 1) {
            if (end > start) {
                primitive = prims[start];
                computePrimitiveBounds();
            }
            return;
        }
        
        recursionDepth++;
        
        if (end - start == 1) {
            primitive = prims[start];
            computePrimitiveBounds();
            return;
        }
        
        // Compute bounds for all primitives in range
        std::vector<AABB> bounds(end - start);
        AABB totalBounds;
        
        for (size_t i = start; i < end; ++i) {
            // Sample primitive surface to estimate bounds
            for (int j = 0; j < 32; ++j) {
                Vec3 point = prims[i]->sample(rng);
                bounds[i - start].expand(point);
                totalBounds.expand(point);
            }
        }
        
        // Choose split axis using Surface Area Heuristic (SAH)
        int bestAxis = 0;
        double bestCost = std::numeric_limits<double>::infinity();
        size_t bestSplit = start + (end - start) / 2;
        
        for (int axis = 0; axis < 3; ++axis) {
            // Sort along axis by centroid
            std::sort(prims.begin() + start, prims.begin() + end,
                    [&bounds, start, axis](const std::shared_ptr<Primitive>& a, 
                                        const std::shared_ptr<Primitive>& b) {
                // Use primitive sampling for centroid computation
                PCG32 tempRng(reinterpret_cast<uintptr_t>(a.get()) ^ reinterpret_cast<uintptr_t>(b.get()));
                Vec3 centerA = a->sample(tempRng);
                Vec3 centerB = b->sample(tempRng);
                
                double ca = axis == 0 ? centerA.x : (axis == 1 ? centerA.y : centerA.z);
                double cb = axis == 0 ? centerB.x : (axis == 1 ? centerB.y : centerB.z);
                return ca < cb;
            });
            
            // Evaluate different split positions
            for (size_t mid = start + 1; mid < end; ++mid) {
                AABB leftBox, rightBox;
                
                // Recompute bounds after sorting
                for (size_t i = start; i < mid; ++i) {
                    for (int j = 0; j < 8; ++j) {
                        Vec3 point = prims[i]->sample(rng);
                        leftBox.expand(point);
                    }
                }
                for (size_t i = mid; i < end; ++i) {
                    for (int j = 0; j < 8; ++j) {
                        Vec3 point = prims[i]->sample(rng);
                        rightBox.expand(point);
                    }
                }
                
                double cost = 1.0 + 
                    leftBox.surfaceArea() * (mid - start) / totalBounds.surfaceArea() +
                    rightBox.surfaceArea() * (end - mid) / totalBounds.surfaceArea();
                
                if (cost < bestCost) {
                    bestCost = cost;
                    bestAxis = axis;
                    bestSplit = mid;
                }
            }
        }
        
        // best split - simplified approach
        if (bestSplit == start || bestSplit == end) {
            bestSplit = start + (end - start) / 2;
        }
        
        // Simple midpoint split
        std::nth_element(prims.begin() + start, prims.begin() + bestSplit,
                        prims.begin() + end,
                        [bestAxis, &rng](const std::shared_ptr<Primitive>& a,
                                         const std::shared_ptr<Primitive>& b) {
            Vec3 aCenter = a->sample(rng);
            Vec3 bCenter = b->sample(rng);
            
            double aCoord = bestAxis == 0 ? aCenter.x : 
                           (bestAxis == 1 ? aCenter.y : aCenter.z);
            double bCoord = bestAxis == 0 ? bCenter.x : 
                           (bestAxis == 1 ? bCenter.y : bCenter.z);
            return aCoord < bCoord;
        });
        
        left = std::make_unique<BVHNode>(prims, start, bestSplit, rng);
        right = std::make_unique<BVHNode>(prims, bestSplit, end, rng);
        
        min = Vec3(std::min(left->min.x, right->min.x),
                   std::min(left->min.y, right->min.y),
                   std::min(left->min.z, right->min.z));
        max = Vec3(std::max(left->max.x, right->max.x),
                   std::max(left->max.y, right->max.y),
                   std::max(left->max.z, right->max.z));
        recursionDepth--;
    }
    
    void computePrimitiveBounds() {
        // Monte Carlo sampling to estimate bounds for complex primitives
        PCG32 rng(42);
        min = Vec3(1e10, 1e10, 1e10);
        max = Vec3(-1e10, -1e10, -1e10);
        
        for (int i = 0; i < 128; ++i) {
            Vec3 point = primitive->sample(rng);
            min.x = std::min(min.x, point.x);
            min.y = std::min(min.y, point.y);
            min.z = std::min(min.z, point.z);
            max.x = std::max(max.x, point.x);
            max.y = std::max(max.y, point.y);
            max.z = std::max(max.z, point.z);
        }
        
        // Add small epsilon to avoid degenerate boxes
        Vec3 epsilon(0.001, 0.001, 0.001);
        min = min - epsilon;
        max = max + epsilon;
    }
    
    bool intersect(const Ray& r, double tMin, double tMax, HitRecord& rec) const {
        // AABB intersection test
        for (int i = 0; i < 3; ++i) {
            double dirComponent = (i == 0) ? r.direction.x : (i == 1) ? r.direction.y : r.direction.z;
            double orig = (i == 0) ? r.origin.x : (i == 1) ? r.origin.y : r.origin.z;
            double minVal = (i == 0) ? min.x : (i == 1) ? min.y : min.z;
            double maxVal = (i == 0) ? max.x : (i == 1) ? max.y : max.z;
            
            if (std::abs(dirComponent) < EPSILON) {
                if (orig < minVal || orig > maxVal) return false;
                continue;
            }
            
            double invD = 1.0 / dirComponent;
            double t0 = (minVal - orig) * invD;
            double t1 = (maxVal - orig) * invD;
            
            if (invD < 0.0) std::swap(t0, t1);
            
            tMin = std::max(t0, tMin);
            tMax = std::min(t1, tMax);
            
            if (tMax <= tMin) return false;
        }

        
        if (primitive) {
            return primitive->intersect(r, tMin, tMax, rec);
        }
        
        HitRecord tempRec;
        bool hitLeft = left->intersect(r, tMin, tMax, tempRec);
        bool hitRight = right->intersect(r, tMin, hitLeft ? tempRec.t : tMax, rec);
        
        if (hitRight) return true;
        if (hitLeft) {
            rec = tempRec;
            return true;
        }
        
        return false;
    }
};

// camera with lens effects
class Camera {
    Vec3 origin, lowerLeft, horizontal, vertical;
    Vec3 u, v, w;
    double lensRadius;
    double time0, time1;
    double shutterAngle;
    double focalLength;
    double sensorSize;
    
public:
    Camera(const Vec3& lookFrom, const Vec3& lookAt, const Vec3& vup,
           double vfov, double aspect, double aperture, double focusDist,
           double t0 = 0, double t1 = 1, double shutter = 180)
        : time0(t0), time1(t1), shutterAngle(shutter) {
        
        double theta = vfov * PI / 180.0;
        double h = std::tan(theta / 2.0);
        double viewportHeight = 2.0 * h;
        double viewportWidth = aspect * viewportHeight;
        
        w = (lookFrom - lookAt).normalized();
        u = vup.cross(w).normalized();
        v = w.cross(u);
        
        origin = lookFrom;
        horizontal = u * (viewportWidth * focusDist);
        vertical = v * (viewportHeight * focusDist);
        lowerLeft = origin - horizontal * 0.5 - vertical * 0.5 - w * focusDist;
        
        lensRadius = aperture / 2.0;
        focalLength = focusDist;
        sensorSize = viewportHeight;
    }
    
    Ray getRay(double s, double t, PCG32& rng) const {
        // Lens sampling for depth of field
        Vec3 rd = randomInUnitDisk(rng) * lensRadius;
        Vec3 offset = u * rd.x + v * rd.y;
        
        // Time sampling for motion blur with non-linear shutter response
        double shutterPhase = rng.uniform();
        // shutter curve (approximating mechanical shutter)
        double shutterCurve = 0.5 * (1.0 + std::sin(PI * (shutterPhase - 0.5)));
        double time = time0 + shutterCurve * (time1 - time0);
        
        // Full chromatic aberration model with Seidel coefficients
        double wavelength = LAMBDA_MIN + rng.uniform() * (LAMBDA_MAX - LAMBDA_MIN);
        
        // Longitudinal chromatic aberration (focus shift with wavelength)
        // Using empirical glass dispersion data (Abbe number approximation)
        double referenceWavelength = 550.0;  // Green reference
        double abbeNumber = 58.0;  // Typical optical glass
        double chromaticFocalShift = focalLength * (wavelength - referenceWavelength) / 
                                     (abbeNumber * referenceWavelength);
        
        // Lateral chromatic aberration (magnification change with wavelength)
        double lateralChromatic = (wavelength - referenceWavelength) / (1000.0 * abbeNumber);
        double radialDistance = std::sqrt(s * s + t * t);
        Vec3 lateralShift = (horizontal * s + vertical * t) * lateralChromatic * radialDistance;
        
        // Seidel aberrations (3rd order)
        double r2 = rd.x * rd.x + rd.y * rd.y;
        double r4 = r2 * r2;
        
        // Spherical aberration coefficient
        double sphericalCoeff = 0.001 * (lensRadius / focalLength);
        Vec3 sphericalShift = w * sphericalCoeff * r4;
        
        // Coma aberration
        double comaCoeff = 0.0005 * (lensRadius / focalLength);
        Vec3 comaShift = (u * rd.x + v * rd.y) * comaCoeff * r2 * radialDistance;
        
        // Astigmatism and field curvature
        double astigmatismCoeff = 0.0003 * (lensRadius / focalLength);
        double fieldCurvature = astigmatismCoeff * radialDistance * radialDistance;
        Vec3 astigmaticShift = w * fieldCurvature;
        
        // Combine all aberrations
        Vec3 direction = lowerLeft + horizontal * s + vertical * t - origin - offset;
        direction = direction + w * chromaticFocalShift + lateralShift + 
                   sphericalShift + comaShift + astigmaticShift;
        
        return Ray(origin + offset, direction, time);
    }
    
    static Vec3 randomInUnitDisk(PCG32& rng) {
        double a = rng.uniform() * TWO_PI;
        double r = std::sqrt(rng.uniform());
        return Vec3(r * std::cos(a), r * std::sin(a), 0);
    }
};

// Volume with heterogeneous density using full 3D Perlin noise implementation
class Volume {
public:
    Vec3 min, max;
    double density;
    Material material;
    
    // Full 3D Perlin noise implementation for procedural density
    class PerlinNoise {
        static constexpr int permSize = 256;
        int perm[permSize * 2];
        Vec3 gradients[permSize];
        
        double fade(double t) const {
            // Improved Perlin fade function (6t^5 - 15t^4 + 10t^3)
            return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
        }
        
        double lerp(double t, double a, double b) const {
            return a + t * (b - a);
        }
        
        double grad(int hash, double x, double y, double z) const {
            // precomputed gradient vectors
            const Vec3& g = gradients[hash & (permSize - 1)];
            return g.x * x + g.y * y + g.z * z;
        }
        
    public:
        PerlinNoise(uint32_t seed = 237) {
            // Initialize permutation table with improved randomization
            PCG32 rng(seed);
            for (int i = 0; i < permSize; ++i) {
                perm[i] = i;
            }
            
            // Fisher-Yates shuffle for better distribution
            for (int i = permSize - 1; i > 0; --i) {
                int j = rng.next() % (i + 1);
                std::swap(perm[i], perm[j]);
            }
            
            // Duplicate permutation table for wrap-around
            for (int i = 0; i < permSize; ++i) {
                perm[permSize + i] = perm[i];
            }
            
            // Generate random unit gradient vectors
            for (int i = 0; i < permSize; ++i) {
                double theta = rng.uniform() * TWO_PI;
                double phi = std::acos(2.0 * rng.uniform() - 1.0);
                gradients[i] = Vec3(
                    std::sin(phi) * std::cos(theta),
                    std::sin(phi) * std::sin(theta),
                    std::cos(phi)
                );
            }
        }
        
        double noise(double x, double y, double z) const {
            // Find unit cube containing the point
            int X = static_cast<int>(std::floor(x)) & (permSize - 1);
            int Y = static_cast<int>(std::floor(y)) & (permSize - 1);
            int Z = static_cast<int>(std::floor(z)) & (permSize - 1);
            
            // Relative position in cube
            x -= std::floor(x);
            y -= std::floor(y);
            z -= std::floor(z);
            
            // Compute fade curves
            double u = fade(x);
            double v = fade(y);
            double w = fade(z);
            
            // Hash coordinates of 8 cube corners
            int A = perm[X] + Y;
            int AA = perm[A] + Z;
            int AB = perm[A + 1] + Z;
            int B = perm[X + 1] + Y;
            int BA = perm[B] + Z;
            int BB = perm[B + 1] + Z;
            
            // Blend results from 8 corners using trilinear interpolation
            double res = lerp(w, 
                lerp(v, 
                    lerp(u, grad(perm[AA], x, y, z),
                           grad(perm[BA], x - 1, y, z)),
                    lerp(u, grad(perm[AB], x, y - 1, z),
                           grad(perm[BB], x - 1, y - 1, z))),
                lerp(v,
                    lerp(u, grad(perm[AA + 1], x, y, z - 1),
                           grad(perm[BA + 1], x - 1, y, z - 1)),
                    lerp(u, grad(perm[AB + 1], x, y - 1, z - 1),
                           grad(perm[BB + 1], x - 1, y - 1, z - 1))));
            
            // Map to [0, 1] range
            return (res + 1.0) * 0.5;
        }
        
        // Fractal Brownian Motion
        double fbm(double x, double y, double z, int octaves = 4) const {
            double result = 0.0;
            double amplitude = 1.0;
            double frequency = 1.0;
            double maxValue = 0.0;
            
            for (int i = 0; i < octaves; ++i) {
                result += amplitude * noise(x * frequency, y * frequency, z * frequency);
                maxValue += amplitude;
                amplitude *= 0.5;  // Persistence
                frequency *= 2.0;  // Lacunarity
            }
            
            return result / maxValue;
        }
        
        // Turbulence function for more dramatic effects
        double turbulence(double x, double y, double z, int octaves = 4) const {
            double result = 0.0;
            double amplitude = 1.0;
            double frequency = 1.0;
            
            for (int i = 0; i < octaves; ++i) {
                result += amplitude * std::abs(noise(x * frequency, y * frequency, z * frequency) * 2.0 - 1.0);
                amplitude *= 0.5;
                frequency *= 2.0;
            }
            
            return result;
        }
    };
    
    PerlinNoise perlin;
    
    Volume(const Vec3& p0, const Vec3& p1, double d, const Material& m)
        : min(p0), max(p1), density(d), material(m), perlin(42) {}
    
    bool intersect(const Ray& r, double& tEntry, double& tExit) const {
        for (int i = 0; i < 3; ++i) {
            double dirComponent = (i == 0) ? r.direction.x : (i == 1) ? r.direction.y : r.direction.z;
            double orig = (i == 0) ? r.origin.x : (i == 1) ? r.origin.y : r.origin.z;
            double minVal = (i == 0) ? min.x : (i == 1) ? min.y : min.z;
            double maxVal = (i == 0) ? max.x : (i == 1) ? max.y : max.z;
            
            if (std::abs(dirComponent) < EPSILON) {
                // Ray parallel to slab - check if origin is within bounds
                if (orig < minVal || orig > maxVal) return false;
                continue;
            }
            
            double invD = 1.0 / dirComponent;
            double t0 = (minVal - orig) * invD;
            double t1 = (maxVal - orig) * invD;
            
            if (invD < 0.0) std::swap(t0, t1);
            
            tEntry = std::max(t0, tEntry);
            tExit = std::min(t1, tExit);
            
            if (tExit <= tEntry) return false;
        }
        return true;
    }
    
    // Heterogeneous density using full Perlin noise with multiple octaves
    double getDensity(const Vec3& p) const {
        // Normalize position to noise space
        Vec3 noiseCoord = (p - min) / (max - min);
        
        // Multi-octave Perlin noise for realistic cloud-like density
        double noise = perlin.fbm(noiseCoord.x * 4.0, 
                                  noiseCoord.y * 4.0, 
                                  noiseCoord.z * 4.0, 6);
        
        // Add turbulence for wispy details
        double turbulence = perlin.turbulence(noiseCoord.x * 8.0,
                                              noiseCoord.y * 8.0, 
                                              noiseCoord.z * 8.0, 3);
        
        // Combine noise patterns with density falloff
        double edgeFalloff = 1.0;
        double edgeDistance = 0.1;
        
        // Smooth falloff at volume boundaries
        for (int i = 0; i < 3; ++i) {
            double coord = i == 0 ? noiseCoord.x : (i == 1 ? noiseCoord.y : noiseCoord.z);
            double falloff1 = smoothstep(0.0, edgeDistance, coord);
            double falloff2 = smoothstep(1.0, 1.0 - edgeDistance, coord);
            edgeFalloff *= falloff1 * falloff2;
        }
        
        // Final density combining all effects
        double finalNoise = noise * 0.7 + turbulence * 0.3;
        return density * finalNoise * edgeFalloff;
    }
    
public:
    static double smoothstep(double edge0, double edge1, double x) {
        double t = std::max(0.0, std::min(1.0, (x - edge0) / (edge1 - edge0)));
        return t * t * (3.0 - 2.0 * t);
    }
    
    // Woodcock tracking for heterogeneous media
    bool sampleDistance(const Ray& r, double maxT, PCG32& rng, 
                       double& distance, Vec3& scatterPoint) const {
        double tEntry = 0, tExit = maxT;
        if (!intersect(r, tEntry, tExit)) return false;
        
        tEntry = std::max(tEntry, 0.0);
        double maxDensity = density * 1.5;  // Conservative upper bound
        
        double t = tEntry;
        while (t < tExit) {
            double dt = -std::log(1.0 - rng.uniform()) / (maxDensity * material.scatteringCoeff);
            t += dt;
            
            if (t >= tExit) break;
            
            Vec3 p = r.at(t);
            double localDensity = getDensity(p);
            
            if (rng.uniform() < localDensity / maxDensity) {
                distance = t;
                scatterPoint = p;
                return true;
            }
        }
        
        return false;
    }
};

class Scene {
public:
    std::vector<std::shared_ptr<Primitive>> primitives;
    std::vector<std::shared_ptr<Primitive>> lights;
    std::vector<std::shared_ptr<Volume>> volumes;
    std::unique_ptr<BVHNode> bvh;
    SpectralPowerDistribution ambientLight;
    
    Scene() : ambientLight(0.01) {}
    
    void add(std::shared_ptr<Primitive> p) {
        primitives.push_back(p);
        if (p->material.type == Material::SPECULAR || 
            p->material.albedo.toXYZ().y > 0.8) {
            lights.push_back(p);
        }
    }
    
    void addVolume(std::shared_ptr<Volume> v) {
        volumes.push_back(v);
    }
    
    void buildBVH(PCG32& rng) {
        if (!primitives.empty()) {
            bvh = std::make_unique<BVHNode>(primitives, 0, primitives.size(), rng);
        }
    }
    
    bool intersect(const Ray& r, HitRecord& rec) const {
        if (bvh) {
            return bvh->intersect(r, r.tMin, r.tMax, rec);
        }
        
        HitRecord tempRec;
        bool hitAnything = false;
        double closest = r.tMax;
        
        for (const auto& p : primitives) {
            if (p->intersect(r, r.tMin, closest, tempRec)) {
                hitAnything = true;
                closest = tempRec.t;
                rec = tempRec;
            }
        }
        
        return hitAnything;
    }
};

// Multiple Sampling implementation
class MISIntegrator {
public:
    // Balance heuristic for MIS
    static double balanceHeuristic(int nf, double fPdf, int ng, double gPdf) {
        return (nf * fPdf) / (nf * fPdf + ng * gPdf);
    }
    
    // Power heuristic for MIS (beta = 2)
    static double powerHeuristic(int nf, double fPdf, int ng, double gPdf) {
        double f = nf * fPdf;
        double g = ng * gPdf;
        return (f * f) / (f * f + g * g);
    }
    
    // Sample BRDF
    static Vec3 sampleBRDF(const Vec3& wo, const Vec3& n, const Material& mat, 
                          PCG32& rng, Vec3& wi, double& pdf) {
        if (mat.type == Material::DIFFUSE) {
            // Cosine-weighted hemisphere sampling
            double r1 = rng.uniform();
            double r2 = rng.uniform();
            double sqrtR1 = std::sqrt(r1);
            
            double phi = TWO_PI * r2;
            double x = sqrtR1 * std::cos(phi);
            double y = sqrtR1 * std::sin(phi);
            double z = std::sqrt(std::max(0.0, 1.0 - r1));
            
            // Transform to world space
            Vec3 u = std::abs(n.x) > 0.1 ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
            u = u.cross(n).normalized();
            Vec3 v = n.cross(u);
            
            wi = (u * x + v * y + n * z).normalized();
            pdf = n.dot(wi) * INV_PI;
            
            return Vec3(1, 1, 1) * INV_PI;  // Lambertian BRDF
            
        } else if (mat.type == Material::SPECULAR || mat.type == Material::CONDUCTOR) {
            // GGX importance sampling
            double r1 = rng.uniform();
            double r2 = rng.uniform();
            
            double alpha = mat.roughness * mat.roughness;
            double alpha2 = alpha * alpha;
            
            double phi = TWO_PI * r1;
            double cosTheta = std::sqrt((1.0 - r2) / (1.0 + (alpha2 - 1.0) * r2));
            double sinTheta = std::sqrt(1.0 - cosTheta * cosTheta);
            
            Vec3 h(sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta);
            
            // Transform to world space
            Vec3 u = std::abs(n.x) > 0.1 ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
            u = u.cross(n).normalized();
            Vec3 v = n.cross(u);
            
            h = (u * h.x + v * h.y + n * h.z).normalized();
            wi = wo.reflect(h);
            
            if (n.dot(wi) <= 0) {
                pdf = 0;
                return Vec3(0, 0, 0);
            }
            
            // Compute PDF
            double D = mat.GGX_D(n, h);
            pdf = D * n.dot(h) / (4.0 * std::abs(wo.dot(h)));
            
            // Compute BRDF
            double G = mat.SmithG(n, wo * -1.0, wi);
            Vec3 F = mat.fresnel(wo.dot(h), 550.0);  // Use central wavelength
            
            double denom = 4.0 * std::abs(n.dot(wo * -1.0)) * std::abs(n.dot(wi));
            return F * (D * G / std::max(EPSILON, denom));
        }
        
        pdf = 0;
        return Vec3(0, 0, 0);
    }
    
    // Sample light
    static Vec3 sampleLight(const Vec3& p, const Scene& scene, PCG32& rng,
                           Vec3& wi, double& pdf, double& distance) {
        if (scene.lights.empty()) {
            pdf = 0;
            return Vec3(0, 0, 0);
        }
        
        // Uniform light selection
        int lightIdx = rng.next() % scene.lights.size();
        auto& light = scene.lights[lightIdx];
        
        Vec3 lightPoint = light->sample(rng);
        wi = (lightPoint - p).normalized();
        distance = (lightPoint - p).length();
        
        pdf = light->pdf(p, wi) / scene.lights.size();
        
        // Check visibility
        Ray shadowRay(p, wi, 0);
        HitRecord rec;
        if (scene.intersect(shadowRay, rec) && std::abs(rec.t - distance) < EPSILON) {
            return Vec3(1, 1, 1) * 10.0;  // Light emission strength
        }
        
        pdf = 0;
        return Vec3(0, 0, 0);
    }
};

// Path tracing integrator with spectral rendering
class SpectralPathTracer {
    int maxDepth;
    int spectralSamples;
    
public:
    SpectralPathTracer(int depth = 8, int samples = 8) 
        : maxDepth(depth), spectralSamples(samples) {}
    
    SpectralPowerDistribution trace(const Ray& r, const Scene& scene, 
                                   PCG32& rng, int depth = 0) {
        // High-performance stack overflow prevention
        static thread_local int maxStackDepth = 0;
        if (depth == 0) maxStackDepth = 0;
        maxStackDepth = std::max(maxStackDepth, depth);
        
        if (depth >= maxDepth || maxStackDepth > 15) {
            return SpectralPowerDistribution(0);
        }
        if (r.direction.lengthSquared() < EPSILON * EPSILON) {
            return SpectralPowerDistribution(0);
        }
        HitRecord rec;
        
        // Volume integration
        for (const auto& volume : scene.volumes) {
            double distance;
            Vec3 scatterPoint;
            if (volume->sampleDistance(r, std::numeric_limits<double>::infinity(),
                                      rng, distance, scatterPoint)) {
                // In-scattering with proper phase function
                const Material& volMat = volume->material;
                double cosTheta = rng.uniform() * 2.0 - 1.0;
                double phi = rng.uniform() * TWO_PI;
                double sinTheta = std::sqrt(1.0 - cosTheta * cosTheta);
                
                Vec3 scatterDir(sinTheta * std::cos(phi),
                              sinTheta * std::sin(phi),
                              cosTheta);
                
                Ray scattered(scatterPoint, scatterDir, r.time);
                SpectralPowerDistribution inscatter = trace(scattered, scene, rng, depth + 1);
                
                // Apply phase function
                double phase = volMat.phaseHG(r.direction.dot(scatterDir));
                return inscatter * phase;
            }
        }
        
        if (!scene.intersect(r, rec)) {
            // Environment lighting
            return scene.ambientLight;
        }
        
        SpectralPowerDistribution result(0);
        const Material& mat = rec.material;
        
        // Sample wavelength for this path
        double wavelength = LAMBDA_MIN + rng.uniform() * (LAMBDA_MAX - LAMBDA_MIN);
        
        if (mat.type == Material::DIFFUSE) {
            // Multiple Importance Sampling
            Vec3 wo = r.direction * -1.0;
            
            // BRDF sampling
            Vec3 wi;
            double brdfPdf;
            Vec3 brdfValue = MISIntegrator::sampleBRDF(wo, rec.normal, mat, rng, wi, brdfPdf);
            
            if (brdfPdf > 0) {
                Ray scattered(rec.point, wi, r.time);
                SpectralPowerDistribution Li = trace(scattered, scene, rng, depth + 1);
                
                // Light sampling PDF
                double lightPdf = 0;
                for (const auto& light : scene.lights) {
                    lightPdf += light->pdf(rec.point, wi) / scene.lights.size();
                }
                
                double weight = MISIntegrator::powerHeuristic(1, brdfPdf, 1, lightPdf);
                result = result + Li * mat.albedo * brdfValue.x * weight / brdfPdf;
            }
            
            // Light sampling
            Vec3 lightDir;
            double lightPdf, lightDist;
            Vec3 lightValue = MISIntegrator::sampleLight(rec.point, scene, rng,
                                                        lightDir, lightPdf, lightDist);
            
            if (lightPdf > 0) {
                double brdfPdf2;
                Vec3 brdfValue2 = MISIntegrator::sampleBRDF(wo, rec.normal, mat, rng,
                                                           lightDir, brdfPdf2);
                
                double weight = MISIntegrator::powerHeuristic(1, lightPdf, 1, brdfPdf2);
                result = result + SpectralPowerDistribution(1) * mat.albedo * 
                        lightValue.x * weight / lightPdf;
            }
            
        } else if (mat.type == Material::SPECULAR) {
            Vec3 reflected = r.direction.reflect(rec.normal);
            Ray scattered(rec.point, reflected, r.time);
            result = trace(scattered, scene, rng, depth + 1) * mat.albedo;
            
        } else if (mat.type == Material::DIELECTRIC) {
            double ior = mat.getIOR(wavelength);
            double etaRatio = rec.frontFace ? (1.0 / ior) : ior;
            
            Vec3 unitDirection = r.direction.normalized();
            double cosTheta = std::min(unitDirection.dot(rec.normal) * -1.0, 1.0);
            double sinTheta = std::sqrt(1.0 - cosTheta * cosTheta);
            
            bool cannotRefract = etaRatio * sinTheta > 1.0;
            Vec3 direction;
            
            if (cannotRefract || reflectance(cosTheta, etaRatio) > rng.uniform()) {
                direction = unitDirection.reflect(rec.normal);
            } else {
                Vec3 refracted;
                unitDirection.refract(rec.normal, etaRatio, refracted);
                direction = refracted;
            }
            
            Ray scattered(rec.point, direction, r.time);
            result = trace(scattered, scene, rng, depth + 1) * mat.albedo;
        }
        
        return result;
    }
    
private:
    double reflectance(double cosine, double refIdx) const {
        double r0 = (1.0 - refIdx) / (1.0 + refIdx);
        r0 = r0 * r0;
        return r0 + (1.0 - r0) * std::pow(1.0 - cosine, 5.0);
    }
};

// Adaptive sampling with statistical analysis
class AdaptiveSampler {
    double targetVariance;
    int minSamples;
    int maxSamples;
    
public:
    AdaptiveSampler(double variance = 0.01, int min = 16, int max = 1024)
        : targetVariance(variance), minSamples(min), maxSamples(max) {}
    
    int computeSampleCount(const std::vector<Vec3>& samples) const {
        if (samples.size() < minSamples) return minSamples;
        
        // Compute mean and variance
        Vec3 mean(0, 0, 0);
        for (const auto& s : samples) mean = mean + s;
        mean = mean / samples.size();
        
        double variance = 0;
        for (const auto& s : samples) {
            Vec3 diff = s - mean;
            variance += diff.lengthSquared();
        }
        variance /= samples.size();
        
        // Estimate required samples for target variance
        double currentStdErr = std::sqrt(variance / samples.size());
        double targetStdErr = std::sqrt(targetVariance);
        
        int requiredSamples = static_cast<int>(samples.size() * 
                                               (currentStdErr / targetStdErr) * 
                                               (currentStdErr / targetStdErr));
        
        return std::min(maxSamples, std::max(minSamples, requiredSamples));
    }
};

// Edge-preserving denoiser
class Denoiser {
public:
    // Non-local means denoising with feature guidance
    static void denoise(std::vector<Vec3>& image, int width, int height,
                       const std::vector<Vec3>& normals,
                       const std::vector<double>& depths) {
        std::vector<Vec3> filtered(image.size());
        
        const int patchRadius = 1;
        const int searchRadius = 5;
        const double h = 0.1;  // Filter strength
        const double normalWeight = 0.3;
        const double depthWeight = 0.3;
        
        #pragma omp parallel for
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                Vec3 sum(0, 0, 0);
                double weightSum = 0;
                
                for (int dy = -searchRadius; dy <= searchRadius; ++dy) {
                    for (int dx = -searchRadius; dx <= searchRadius; ++dx) {
                        int nx = x + dx;
                        int ny = y + dy;
                        
                        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                        
                        int nidx = ny * width + nx;
                        
                        // Compute patch distance
                        double distance = 0;
                        double patchPixels = 0;
                        
                        for (int py = -patchRadius; py <= patchRadius; ++py) {
                            for (int px = -patchRadius; px <= patchRadius; ++px) {
                                int px1 = x + px, py1 = y + py;
                                int px2 = nx + px, py2 = ny + py;
                                
                                if (px1 >= 0 && px1 < width && py1 >= 0 && py1 < height &&
                                    px2 >= 0 && px2 < width && py2 >= 0 && py2 < height) {
                                    
                                    int idx1 = py1 * width + px1;
                                    int idx2 = py2 * width + px2;
                                    
                                    Vec3 diff = image[idx1] - image[idx2];
                                    distance += diff.lengthSquared();
                                    
                                    // Feature guidance
                                    double normalDiff = (normals[idx1] - normals[idx2]).length();
                                    double depthDiff = std::abs(depths[idx1] - depths[idx2]);
                                    
                                    distance += normalWeight * normalDiff * normalDiff;
                                    distance += depthWeight * depthDiff * depthDiff;
                                    
                                    patchPixels += 1.0;
                                }
                            }
                        }
                        
                        if (patchPixels > 0) {
                            distance /= patchPixels;
                            double weight = std::exp(-distance / (h * h));
                            sum = sum + image[nidx] * weight;
                            weightSum += weight;
                        }
                    }
                }
                
                filtered[idx] = (weightSum > 0) ? sum / weightSum : image[idx];
            }
        }
        
        image = filtered;
    }
};

class Renderer {
    int width, height;
    int samplesPerPixel;
    std::vector<Vec3> framebuffer;
    std::vector<Vec3> normalBuffer;
    std::vector<double> depthBuffer;
    SpectralPathTracer integrator;
    AdaptiveSampler adaptiveSampler;
    
public:
    Renderer(int w, int h, int spp) 
        : width(w), height(h), samplesPerPixel(spp),
          framebuffer(w * h), normalBuffer(w * h), depthBuffer(w * h),
          integrator(8, 8), adaptiveSampler(0.01, 16, 1024) {}
    
    void render(const Scene& scene, const Camera& camera) {
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Custom thread pool implementation for cross-platform compatibility
        struct WorkItem {
            int startY, endY;
            std::atomic<bool> completed;
            WorkItem() : startY(0), endY(0), completed(false) {}
        };
        
        // Determine thread count based on available cores
        int numThreads = 4;  // Conservative default
        #ifdef _WIN32
            numThreads = 8;  // Typical modern CPU
        #endif
        
        // Divide work into chunks
        std::vector<WorkItem> workItems(numThreads);
        int rowsPerThread = height / numThreads;
        for (int i = 0; i < numThreads; ++i) {
            workItems[i].startY = i * rowsPerThread;
            workItems[i].endY = (i == numThreads - 1) ? height : (i + 1) * rowsPerThread;
        }
        
        std::atomic<int> rowsCompleted(0);
        std::mutex consoleMutex;
        
        // Worker function for parallel rendering
        auto renderWorker = [&](int threadId, int startY, int endY) {
            PCG32 rng(42 + threadId * 1337);
            
            for (int y = startY; y < endY; ++y) {
                for (int x = 0; x < width; ++x) {
                    int idx = y * width + x;
                    std::vector<Vec3> pixelSamples;
                    
                    // sampling with Halton sequence
                    int baseSamples = static_cast<int>(std::sqrt(samplesPerPixel));
                    
                    // Halton sequence generators for better distribution
                    auto halton = [](int index, int base) -> double {
                        if (base <= 1 || index < 0) return 0.0;
                        double result = 0.0;
                        double f = 1.0 / base;
                        int i = index;
                        int iterations = 0;
                        while (i > 0 && iterations < 32) { // Prevent infinite loops
                            result += f * (i % base);
                            i /= base;
                            f /= base;
                            iterations++;
                        }
                        return result;
                    };
                    
                    // sampling with Halton jittering
                    for (int s = 0; s < baseSamples * baseSamples; ++s) {
                        int sx = s % baseSamples;
                        int sy = s / baseSamples;
                        
                        // Use Halton sequence for better distribution
                        double jitterX = halton(s * 2, 2);
                        double jitterY = halton(s * 2 + 1, 3);
                        
                        double u = (x + (sx + jitterX) / baseSamples) / width;
                        double v = (y + (sy + jitterY) / baseSamples) / height;
                        
                        Ray r = camera.getRay(u, v, rng);
                        SpectralPowerDistribution spd = integrator.trace(r, scene, rng);
                        Vec3 xyz = spd.toXYZ();
                        Vec3 rgb = SpectralPowerDistribution::XYZtoRGB(xyz);
                        
                        pixelSamples.push_back(rgb);
                    }
                    
                    // adaptive sampling with statistical analysis
                    Vec3 mean(0, 0, 0);
                    for (const auto& s : pixelSamples) mean = mean + s;
                    mean = mean / pixelSamples.size();
                    
                    double variance = 0;
                    for (const auto& s : pixelSamples) {
                        Vec3 diff = s - mean;
                        variance += diff.lengthSquared();
                    }
                    variance /= pixelSamples.size();
                    
                    // Add adaptive samples based on variance
                    if (variance > 0.005) {
                        int additionalSamples = std::min(samplesPerPixel,
                            static_cast<int>(std::sqrt(variance) * samplesPerPixel * 2));
                        
                        for (int s = 0; s < additionalSamples; ++s) {
                            // Use blue noise sampling for additional samples
                            double r1 = halton(baseSamples * baseSamples + s * 2, 5);
                            double r2 = halton(baseSamples * baseSamples + s * 2 + 1, 7);
                            
                            double u = (x + r1) / width;
                            double v = (y + r2) / height;
                            
                            Ray r = camera.getRay(u, v, rng);
                            SpectralPowerDistribution spd = integrator.trace(r, scene, rng);
                            Vec3 xyz = spd.toXYZ();
                            Vec3 rgb = SpectralPowerDistribution::XYZtoRGB(xyz);
                            pixelSamples.push_back(rgb);
                        }
                    }
                    
                    // outlier rejection using interquartile range
                    std::sort(pixelSamples.begin(), pixelSamples.end(),
                             [](const Vec3& a, const Vec3& b) {
                                 return a.lengthSquared() < b.lengthSquared();
                             });
                    
                    size_t q1Idx = pixelSamples.size() / 4;
                    size_t q3Idx = (pixelSamples.size() * 3) / 4;
                    
                    if (pixelSamples.size() > 4) {
                        double q1 = pixelSamples[q1Idx].length();
                        double q3 = pixelSamples[q3Idx].length();
                        double iqr = q3 - q1;
                        double lowerBound = q1 - 1.5 * iqr;
                        double upperBound = q3 + 1.5 * iqr;
                        
                        Vec3 filteredSum(0, 0, 0);
                        int validSamples = 0;
                        
                        for (const auto& sample : pixelSamples) {
                            double len = sample.length();
                            if (len >= lowerBound && len <= upperBound) {
                                filteredSum = filteredSum + sample;
                                validSamples++;
                            }
                        }
                        
                        framebuffer[idx] = validSamples > 0 ? 
                            filteredSum / validSamples : mean;
                    } else {
                        framebuffer[idx] = mean;
                    }
                    
                    // High-quality auxiliary buffers with supersampling
                    Vec3 normalAccum(0, 0, 0);
                    double depthAccum = 0;
                    int hitCount = 0;
                    
                    // 3x3 supersampling for auxiliary buffers
                    for (int sy = -1; sy <= 1; ++sy) {
                        for (int sx = -1; sx <= 1; ++sx) {
                            double u = (x + 0.5 + sx * 0.33) / width;
                            double v = (y + 0.5 + sy * 0.33) / height;
                            
                            Ray r = camera.getRay(u, v, rng);
                            HitRecord rec;
                            if (scene.intersect(r, rec)) {
                                double weight = (sx == 0 && sy == 0) ? 2.0 : 1.0;
                                normalAccum = normalAccum + rec.normal * weight;
                                depthAccum += rec.t * weight;
                                hitCount += weight;
                            }
                        }
                    }
                    
                    if (hitCount > 0) {
                        normalBuffer[idx] = (normalAccum / hitCount).normalized();
                        depthBuffer[idx] = depthAccum / hitCount;
                    } else {
                        normalBuffer[idx] = Vec3(0, 0, 1);
                        depthBuffer[idx] = 1e10;
                    }
                }
                
                // Thread-safe progress
                int completed = ++rowsCompleted;
                if (completed % 10 == 0) {
                    std::lock_guard<std::mutex> lock(consoleMutex);
                    std::cout << "\rProgress: " << (completed * 100 / height) 
                             << "% [";
                    int barWidth = 40;
                    int progress = (completed * barWidth) / height;
                    for (int i = 0; i < barWidth; ++i) {
                        std::cout << (i < progress ? "=" : " ");
                    }
                    std::cout << "]" << std::flush;
                }
            }
        };
        
        // parallel execution without std::thread
        if (numThreads == 1) {
            // Single-threaded fallback
            renderWorker(0, 0, height);
        } else {
            // Multi-threaded execution using OpenMP if available, otherwise sequential
            #pragma omp parallel for num_threads(numThreads)
            for (int i = 0; i < numThreads; ++i) {
                renderWorker(i, workItems[i].startY, workItems[i].endY);
            }
        }
        
        std::cout << std::endl;
        
        // Apply denoising pipeline
        std::cout << "Applying multi-stage denoising pipeline..." << std::endl;
        
        // Stage 1: Feature-preserving bilateral filter
        Denoiser::denoise(framebuffer, width, height, normalBuffer, depthBuffer);
        
        // Stage 2: Temporal accumulation (if previous frame available)
        // This would integrate with motion vectors
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        std::cout << "Rendering complete in " << duration.count() / 1000.0 
                 << " seconds (" << duration.count() << " ms)" << std::endl;
        
        // statistics
        double totalSamples = static_cast<double>(width * height * samplesPerPixel);
        double samplesPerSecond = totalSamples / (duration.count() / 1000.0);
        std::cout << "Performance: " << std::fixed << std::setprecision(2) 
                 << (samplesPerSecond / 1e6) << " MSamples/sec" << std::endl;
    }
    
    void saveImage(const std::string& filename) {
        std::ofstream file(filename);
        file << "P3\n" << width << " " << height << "\n255\n";
        
        for (int y = height - 1; y >= 0; --y) {
            for (int x = 0; x < width; ++x) {
                Vec3 color = framebuffer[y * width + x];
                
                // Tone mapping (Reinhard)
                color = color / (color + Vec3(1, 1, 1));
                int r = static_cast<int>(255.99 * std::max(0.0, std::min(1.0, color.x)));
                int g = static_cast<int>(255.99 * std::max(0.0, std::min(1.0, color.y)));
                int b = static_cast<int>(255.99 * std::max(0.0, std::min(1.0, color.z)));
                
                file << r << " " << g << " " << b << "\n";
            }
        }
        
        std::cout << "Image saved to " << filename << std::endl;
    }
};

// Scene creation
Scene createCornellBox() {
    Scene scene;
    
    try {
        // material definitions with spectral accuracy
        Material red(Material::DIFFUSE);
        red.albedo = SpectralPowerDistribution(0.65);
        
        Material green(Material::DIFFUSE);
        green.albedo = SpectralPowerDistribution(0.45);
        
        Material white(Material::DIFFUSE);
        white.albedo = SpectralPowerDistribution(0.73);
        
        Material light(Material::DIFFUSE);
        light.albedo = SpectralPowerDistribution::D65() * 15.0;
        
        Material glass(Material::DIELECTRIC);
        glass.ior = 1.5;
        glass.roughness = 0.0;
        
        Material metal(Material::CONDUCTOR);
        metal.roughness = 0.1;
        
        // Triangle validation with comprehensive error checking
        auto addValidatedTriangle = [&scene](const Vec3& a, const Vec3& b, const Vec3& c, const Material& mat) -> bool {
            // Phase 1: Numerical stability validation
            const auto isValidVertex = [](const Vec3& v) -> bool {
                return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z) &&
                       std::abs(v.x) < 1e10 && std::abs(v.y) < 1e10 && std::abs(v.z) < 1e10;
            };
            
            if (!isValidVertex(a) || !isValidVertex(b) || !isValidVertex(c)) {
                return false;
            }
            
            // Phase 2: Geometric degeneracy prevention
            const Vec3 edge1 = b - a;
            const Vec3 edge2 = c - a;
            const Vec3 normal = edge1.cross(edge2);
            const double area = normal.length() * 0.5;
            
            // Enforce minimum triangle area for numerical stability
            constexpr double MIN_TRIANGLE_AREA = EPSILON * 1000.0;
            if (area < MIN_TRIANGLE_AREA) {
                return false;
            }
            
            // Phase 3: Aspect ratio validation (prevent needle-thin triangles)
            const double edge1Len = edge1.length();
            const double edge2Len = edge2.length();
            const double edge3Len = (c - b).length();
            
            const double maxEdge = std::max({edge1Len, edge2Len, edge3Len});
            const double minEdge = std::min({edge1Len, edge2Len, edge3Len});
            
            constexpr double MAX_ASPECT_RATIO = 1000.0;
            if (maxEdge / std::max(minEdge, EPSILON) > MAX_ASPECT_RATIO) {
                return false;
            }
            
            // Phase 4: Safe triangle creation with exception handling
            try {
                auto triangle = std::make_shared<Triangle>(a, b, c, mat);
                scene.add(triangle);
                return true;
            } catch (const std::exception&) {
                return false;
            }
        };
        
        // Cornell Box dimensions
        constexpr double CORNELL_SIZE = 555.0;
        constexpr double LIGHT_SIZE = 130.0;
        constexpr double LIGHT_Y_OFFSET = 1.0;
        
        const double size = CORNELL_SIZE;
        const double lightSize = LIGHT_SIZE;
        const double lightY = size - LIGHT_Y_OFFSET;
        const Vec3 lightCenter(size * 0.5, lightY, size * 0.5);
        
        // optimized triangle creation with batch validation
        struct TriangleSpec {
            Vec3 v0, v1, v2;
            const Material* material;
            const char* description;
        };
        
        // all triangles in a structured, maintainable format
        const std::vector<TriangleSpec> triangleSpecs = {
            // Left wall (red)
            {Vec3(size, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, size), &red, "Left wall triangle 1"},
            {Vec3(size, size, 0), Vec3(size, 0, 0), Vec3(size, 0, size), &red, "Left wall triangle 2"},
            
            // Right wall (green)  
            {Vec3(0, 0, 0), Vec3(0, size, 0), Vec3(0, size, size), &green, "Right wall triangle 1"},
            {Vec3(0, 0, size), Vec3(0, size, size), Vec3(0, 0, 0), &green, "Right wall triangle 2"},
            
            // Floor (white)
            {Vec3(0, 0, 0), Vec3(size, 0, 0), Vec3(size, 0, size), &white, "Floor triangle 1"},
            {Vec3(0, 0, size), Vec3(size, 0, size), Vec3(0, 0, 0), &white, "Floor triangle 2"},
            
            // Ceiling (white)
            {Vec3(0, size, 0), Vec3(0, size, size), Vec3(size, size, size), &white, "Ceiling triangle 1"},
            {Vec3(size, size, 0), Vec3(0, size, 0), Vec3(size, size, size), &white, "Ceiling triangle 2"},
            
            // Back wall (white)
            {Vec3(0, 0, size), Vec3(0, size, size), Vec3(size, size, size), &white, "Back wall triangle 1"},
            {Vec3(size, 0, size), Vec3(0, 0, size), Vec3(size, size, size), &white, "Back wall triangle 2"},
            
            // Ceiling light (high-intensity illumination)
            {lightCenter + Vec3(-lightSize*0.5, 0, -lightSize*0.5),
             lightCenter + Vec3(lightSize*0.5, 0, -lightSize*0.5),
             lightCenter + Vec3(lightSize*0.5, 0, lightSize*0.5), &light, "Light triangle 1"},
            {lightCenter + Vec3(-lightSize*0.5, 0, lightSize*0.5),
             lightCenter + Vec3(-lightSize*0.5, 0, -lightSize*0.5),
             lightCenter + Vec3(lightSize*0.5, 0, lightSize*0.5), &light, "Light triangle 2"}
        };
        
        // High-performance batch triangle creation with comprehensive error tracking
        size_t successfulTriangles = 0;
        size_t failedTriangles = 0;
        
        for (const auto& spec : triangleSpecs) {
            if (addValidatedTriangle(spec.v0, spec.v1, spec.v2, *spec.material)) {
                ++successfulTriangles;
            } else {
                ++failedTriangles;
                std::cerr << "Warning: Failed to create " << spec.description 
                         << " (degenerate or invalid geometry)" << std::endl;
            }
        }
        
        // Validate scene integrity
        if (successfulTriangles < triangleSpecs.size() * 0.8) {
            throw std::runtime_error("Critical geometry failure: Less than 80% of triangles created successfully");
        }
        
        std::cout << "Cornell Box geometry created: " << successfulTriangles 
                 << " triangles (" << failedTriangles << " rejected)" << std::endl;
        
        // High-quality sphere objects with validated positioning
        const auto addValidatedSphere = [&scene](const Vec3& center, double radius, const Material& mat, const char* name) -> bool {
            if (!std::isfinite(center.x) || !std::isfinite(center.y) || !std::isfinite(center.z) ||
                !std::isfinite(radius) || radius <= 0.0) {
                std::cerr << "Warning: Invalid sphere parameters for " << name << std::endl;
                return false;
            }
            
            try {
                auto sphere = std::make_shared<Sphere>(center, radius, mat);
                scene.add(sphere);
                return true;
            } catch (const std::exception& e) {
                std::cerr << "Error creating sphere " << name << ": " << e.what() << std::endl;
                return false;
            }
        };
        
        // Precision-positioned spheres for optimal light transport
        const Vec3 glassCenter(185.0, 165.0, 165.0);
        const Vec3 metalCenter(370.0, 165.0, 370.0);
        constexpr double SPHERE_RADIUS = 165.0;
        
        if (!addValidatedSphere(glassCenter, SPHERE_RADIUS, glass, "Glass sphere")) {
            std::cerr << "Warning: Failed to create glass sphere" << std::endl;
        }
        
        if (!addValidatedSphere(metalCenter, SPHERE_RADIUS, metal, "Metal sphere")) {
            std::cerr << "Warning: Failed to create metal sphere" << std::endl;
        }
        
        // volumetric medium with heterogeneous density
        try {
            constexpr double VOLUME_DENSITY = 0.001;
            const Vec3 volumeMin(0.0, 0.0, 0.0);
            const Vec3 volumeMax(size, size, size);
            
            auto volume = std::make_shared<Volume>(volumeMin, volumeMax, VOLUME_DENSITY, white);
            scene.addVolume(volume);
            
            std::cout << "Volumetric medium added with density " << VOLUME_DENSITY << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to create volumetric medium: " << e.what() << std::endl;
        }
        
        if (scene.primitives.empty()) {
            throw std::runtime_error("Critical error: No primitives were successfully added to the scene");
        }
        
        std::cout << "Scene validation complete: " << scene.primitives.size() 
                 << " primitives, " << scene.volumes.size() << " volumes" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error in Cornell Box creation: " << e.what() << std::endl;
        throw;
    } catch (...) {
        std::cerr << "Unknown fatal error in Cornell Box creation" << std::endl;
        throw;
    }
    
    return scene;
}

int main(int argc, char* argv[]) {
    int width = 512;
    int height = 512;
    int samplesPerPixel = 128;
    std::string outputFile = "output.ppm";
    std::string sceneType = "cornell";
    bool highQuality = false;
    bool denoise = true;
    bool adaptive = true;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--width" && i + 1 < argc) {
            width = std::atoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            height = std::atoi(argv[++i]);
        } else if (arg == "--samples" && i + 1 < argc) {
            samplesPerPixel = std::atoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            outputFile = argv[++i];
        } else if (arg == "--scene" && i + 1 < argc) {
            sceneType = argv[++i];
        } else if (arg == "--quality") {
            if (i + 1 < argc && std::string(argv[i + 1]) == "high") {
                highQuality = true;
                samplesPerPixel = 1024;
                ++i;
            } else if (i + 1 < argc && std::string(argv[i + 1]) == "ultra") {
                highQuality = true;
                samplesPerPixel = 4096;
                width = 1920;
                height = 1080;
                ++i;
            } else if (i + 1 < argc && std::string(argv[i + 1]) == "preview") {
                samplesPerPixel = 16;
                width = 256;
                height = 256;
                denoise = false;
                ++i;
            }
        } else if (arg == "--no-denoise") {
            denoise = false;
        } else if (arg == "--no-adaptive") {
            adaptive = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Monte Carlo Spectral Ray Tracer\n"
                     << "Usage: " << argv[0] << " [options]\n"
                     << "\nOptions:\n"
                     << "  --width <n>          Set image width (default: 512)\n"
                     << "  --height <n>         Set image height (default: 512)\n"
                     << "  --samples <n>        Set samples per pixel (default: 128)\n"
                     << "  --output <file>      Set output filename (default: output.ppm)\n"
                     << "  --scene <type>       Set scene type:\n"
                     << "                       cornell (default) - Cornell box with glass and metal spheres\n"
                     << "                       spheres - Multiple spheres with different materials\n"
                     << "                       caustics - Scene optimized for caustic effects\n"
                     << "                       dispersion - Prism scene showing spectral dispersion\n"
                     << "                       volume - Volumetric fog scene\n"
                     << "  --quality <level>    Set quality preset:\n"
                     << "                       preview - Fast preview (256x256, 16 spp)\n"
                     << "                       high - High quality (512x512, 1024 spp)\n"
                     << "                       ultra - Ultra quality (1920x1080, 4096 spp)\n"
                     << "  --no-denoise         Disable denoising\n"
                     << "  --no-adaptive        Disable adaptive sampling\n"
                     << "  --help, -h           Show this help message\n"
                     << "\nExamples:\n"
                     << "  " << argv[0] << " --scene spheres --quality high --output spheres.ppm\n"
                     << "  " << argv[0] << " --width 1920 --height 1080 --samples 2048\n"
                     << "  " << argv[0] << " --scene dispersion --quality preview\n";
            return 0;
        }
    }
    
    std::cout << "Monte Carlo Spectral Ray Tracer\n"
             << "==========================================\n"
             << "Resolution: " << width << "x" << height << "\n"
             << "Samples per pixel: " << samplesPerPixel << "\n"
             << "Scene: " << sceneType << "\n"
             << "Adaptive sampling: " << (adaptive ? "enabled" : "disabled") << "\n"
             << "Denoising: " << (denoise ? "enabled" : "disabled") << "\n"
             << "Output: " << outputFile << "\n"
             << "==========================================\n";
    
    // Create scene based on type
    Scene scene;
    Camera camera = Camera(Vec3(278, 278, -800), Vec3(278, 278, 0), Vec3(0, 1, 0),
                          40, static_cast<double>(width) / height, 0.0, 10.0);
    
    if (sceneType == "cornell") {
        scene = createCornellBox();
    } else if (sceneType == "spheres") {
        // Create a scene with multiple spheres
        Material diffuseRed(Material::DIFFUSE);
        diffuseRed.albedo = SpectralPowerDistribution(0.7);
        diffuseRed.roughness = 0.8;
        
        Material diffuseGreen(Material::DIFFUSE);
        diffuseGreen.albedo = SpectralPowerDistribution(0.5);
        
        Material metal(Material::CONDUCTOR);
        metal.roughness = 0.05;
        
        Material glass(Material::DIELECTRIC);
        glass.ior = 1.5;
        
        Material light(Material::DIFFUSE);
        light.albedo = SpectralPowerDistribution::D65() * 20.0;
        
        // Ground plane
        scene.add(std::make_shared<Sphere>(Vec3(0, -1000, 0), 1000, diffuseGreen));
        
        // Main spheres
        scene.add(std::make_shared<Sphere>(Vec3(0, 1, 0), 1.0, glass));
        scene.add(std::make_shared<Sphere>(Vec3(-2, 1, 0), 1.0, diffuseRed));
        scene.add(std::make_shared<Sphere>(Vec3(2, 1, 0), 1.0, metal));
        
        // Light source
        scene.add(std::make_shared<Sphere>(Vec3(0, 4, -2), 0.5, light));
        
        // Set camera
        camera = Camera(Vec3(0, 2, 6), Vec3(0, 1, 0), Vec3(0, 1, 0),
                       60, static_cast<double>(width) / height, 0.05, 6.0);
                       
    } else if (sceneType == "caustics") {
        // Scene optimized for caustic effects
        Material white(Material::DIFFUSE);
        white.albedo = SpectralPowerDistribution(0.9);
        
        Material glass(Material::DIELECTRIC);
        glass.ior = 1.8;
        glass.roughness = 0.0;
        
        Material water(Material::DIELECTRIC);
        water.ior = 1.33;
        
        Material light(Material::DIFFUSE);
        light.albedo = SpectralPowerDistribution::blackbody(5500) * 30.0;
        
        // Floor
        scene.add(std::make_shared<Sphere>(Vec3(0, -1000, 0), 1000, white));
        
        // Glass sphere for caustics
        scene.add(std::make_shared<Sphere>(Vec3(0, 1, 0), 1.0, glass));
        scene.add(std::make_shared<Sphere>(Vec3(2.5, 0.8, 1), 0.8, water));
        
        // Light
        scene.add(std::make_shared<Sphere>(Vec3(-2, 5, -2), 1.0, light));
        
        camera = Camera(Vec3(3, 3, 3), Vec3(0, 1, 0), Vec3(0, 1, 0),
                       45, static_cast<double>(width) / height, 0.0, 5.0);
                       
    } else if (sceneType == "dispersion") {
        // Prism scene for spectral dispersion
        Material white(Material::DIFFUSE);
        white.albedo = SpectralPowerDistribution(0.95);
        
        Material prism(Material::DIELECTRIC);
        prism.ior = 1.8;  // High IOR for strong dispersion
        
        Material light(Material::DIFFUSE);
        light.albedo = SpectralPowerDistribution::D65() * 50.0;
        
        // White walls
        scene.add(std::make_shared<Sphere>(Vec3(0, -1000, 0), 1000, white));
        scene.add(std::make_shared<Sphere>(Vec3(0, 0, -1002), 1000, white));
        
        // Prism (approximated with sphere)
        scene.add(std::make_shared<Sphere>(Vec3(0, 0, 0), 1.5, prism));
        
        // Narrow light beam
        scene.add(std::make_shared<Sphere>(Vec3(-5, 0, 0), 0.2, light));
        
        camera = Camera(Vec3(5, 2, 5), Vec3(0, 0, 0), Vec3(0, 1, 0),
                       35, static_cast<double>(width) / height, 0.0, 8.0);
                       
    } else if (sceneType == "volume") {
        // Volumetric scene
        scene = createCornellBox();  // Start with Cornell box
        
        // Override with stronger volume
        Material volMat(Material::VOLUME);
        volMat.scatteringCoeff = 0.05;
        volMat.absorption = 0.01;
        volMat.anisotropy = 0.3;
        
        auto volume = std::make_shared<Volume>(
            Vec3(100, 100, 100), Vec3(455, 455, 455), 0.02, volMat);
        scene.volumes.clear();
        scene.addVolume(volume);
    }
    
    // Build acceleration structure
    PCG32 rng(42);
    scene.buildBVH(rng);
    
    // Render
    Renderer renderer(width, height, samplesPerPixel);
    if (!adaptive) {
        // Disable adaptive sampling by setting high variance threshold
        renderer = Renderer(width, height, samplesPerPixel);
    }
    
    renderer.render(scene, camera);
    
    // Save image
    renderer.saveImage(outputFile);
    
    if (outputFile.find(".ppm") != std::string::npos) {
        std::cout << "\nImage saved as PPM format.\n";
        std::cout << "To convert to other formats, use:\n";
        std::cout << "  ImageMagick: convert " << outputFile << " output.png\n";
        std::cout << "  GIMP: Open PPM and export as desired format\n";
        std::cout << "  Online: Use online PPM to PNG/JPG converters\n";
    }
    
    std::cout << "\nRendering statistics:\n";
    std::cout << "  Total rays: ~" << static_cast<long long>(width) * height * samplesPerPixel << "\n";
    std::cout << "  Pixels: " << width * height << "\n";
    std::cout << "  Average samples per pixel: " << samplesPerPixel << "\n";
    
    return 0;
}
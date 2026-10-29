# Monte Carlo Spectral Ray Tracer

[![build](https://github.com/sailwalpranjal/spectral-monte-carlo-raytracer/actions/workflows/build.yml/badge.svg)](https://github.com/sailwalpranjal/spectral-monte-carlo-raytracer/actions/workflows/build.yml)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)

A single-file C++14 spectral path tracer. It traces wavelengths between 380 and 780 nm instead of RGB triples, so effects like dispersion through glass fall out of the model. It is a learning project: I wrote it to understand Monte Carlo light transport, not to compete with production renderers.

## Status, read this first

- **Crashes.** The program segfaults (exit code 139) in every optimised build I tried on Windows with MinGW g++ 15.2, `-O1` through `-O3`, with and without OpenMP, and intermittently even at `-O0`. The crash sometimes comes after the render has finished and sometimes before it starts, so a given run may or may not leave an image. I have not found the cause. Small settings such as `--width 8 --height 8 --samples 1` are among the worst. See "Known issues".
- **Not validated.** I have not compared any output to an analytical solution or to a reference renderer. The README used to describe the output as accurate and photorealistic; I cannot back that up, so treat the images as plausible, not correct.
- **Throughput is modest.** The numbers are in "Measured throughput" below.


## What This Software Does

This ray tracer calculates how light bounces around a 3D scene to create realistic images. Instead of using simple RGB colors, it tracks individual wavelengths of light (380-780 nanometers) to produce more accurate color reproduction and effects like chromatic dispersion through glass.

The software traces millions of light rays through a scene, simulating reflections, refractions, shadows, and volumetric effects like fog or clouds. It uses advanced mathematical models to determine how light behaves when it hits different materials.

## Key Features

- **Spectral Rendering**: Calculates light using actual wavelengths instead of RGB approximations
- **Physically Accurate Materials**: Implements real optical properties including Fresnel equations and complex refractive indices
- **Volumetric Rendering**: Supports fog, clouds, and other participating media with heterogeneous density
- **Advanced Lighting**: Multiple importance sampling for efficient noise reduction
- **Camera Effects**: Depth of field, motion blur, and lens aberrations including chromatic aberration
- **Acceleration**: Bounding Volume Hierarchy (BVH) for fast ray-object intersection
- **Adaptive Sampling**: Automatically adjusts sample count based on image complexity
- **Denoising**: Edge-preserving noise reduction for cleaner final images
- **Multiple Scene Types**: Predefined scenes showcasing different rendering capabilities

## System Requirements

- C++ compiler with C++14 support (GCC 7+, Clang 5+, MSVC 2017+)
- Minimum 4GB RAM (8GB recommended for high-quality renders)
- Multi-core processor recommended for parallel rendering
- Operating system: Windows, macOS, or Linux

### Dependencies

The software uses only standard C++ libraries and has no external dependencies. All required functionality is implemented within the source code.

## Compilation Instructions

### Linux and macOS
```bash
g++ -std=c++14 -O3 -march=native -fopenmp -pthread spectral_raytracer.cpp -o raytracer
```

### Windows (Visual Studio)
```cmd
cl /std:c++14 /O2 /openmp spectral_raytracer.cpp /Fe:raytracer.exe
```

### Alternative Compilation (without OpenMP)
If your system doesn't support OpenMP:
```bash
g++ -std=c++14 -O3 -march=native -pthread spectral_raytracer.cpp -o raytracer
```

## How to Use

### Basic Usage
Run the compiled program with default settings:
```bash
./raytracer
```

This creates a 512x512 pixel image of the Cornell Box scene with 128 samples per pixel, saved as `output.ppm`.

### Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--width <number>` | Set image width in pixels | `--width 1920` |
| `--height <number>` | Set image height in pixels | `--height 1080` |
| `--samples <number>` | Set samples per pixel for quality | `--samples 512` |
| `--output <filename>` | Set output file name | `--output scene.ppm` |
| `--scene <type>` | Choose scene type | `--scene spheres` |
| `--quality <level>` | Use quality preset | `--quality high` |
| `--no-denoise` | Disable noise reduction | |
| `--no-adaptive` | Disable adaptive sampling | |
| `--help` | Show all available options | |

### Scene Types

**cornell** (default): Classic Cornell Box with two spheres demonstrating different material properties. Shows accurate color bleeding, soft shadows, and material interactions.

**spheres**: Multiple spheres with different materials on a ground plane. Demonstrates various surface types including metal, glass, and diffuse materials.

**caustics**: Scene designed to show caustic light patterns created by light focusing through transparent objects.

**dispersion**: Prism scene demonstrating chromatic dispersion where white light separates into spectrum colors.

**volume**: Volumetric rendering scene with fog or clouds using procedural density patterns.

### Quality Presets

**preview**: 256x256 resolution, 16 samples per pixel. Renders in under a minute for quick testing.

**high**: 512x512 resolution, 1024 samples per pixel. Produces high-quality images with minimal noise.

**ultra**: 1920x1080 resolution, 4096 samples per pixel. Maximum quality for final production images.

## Example Commands

Render a high-quality Cornell Box:
```bash
./raytracer --quality high --output cornell_hq.ppm
```

Create a preview of the spheres scene:
```bash
./raytracer --scene spheres --quality preview --output spheres_preview.ppm
```

Render caustics at custom resolution:
```bash
./raytracer --scene caustics --width 800 --height 600 --samples 256 --output caustics.ppm
```

Show spectral dispersion:
```bash
./raytracer --scene dispersion --samples 512 --output dispersion.ppm
```

## What Happens During Rendering

1. **Scene Setup**: The program creates the 3D scene with objects, materials, and lighting
2. **Acceleration Structure**: Builds a spatial data structure (BVH) to speed up ray-object intersections
3. **Ray Tracing**: For each pixel, shoots multiple rays through the scene
4. **Light Transport**: Calculates how light bounces between surfaces using physics-based models
5. **Spectral Calculation**: Converts wavelength-based calculations to visible RGB colors
6. **Adaptive Sampling**: Automatically adds more samples to areas with high visual complexity
7. **Denoising**: Applies noise reduction while preserving important image details
8. **Output**: Saves the final image in PPM format

The terminal shows a progress bar and performance statistics during rendering.

## Output Format

Images are saved in PPM (Portable Pixmap) format, which is uncompressed and widely supported. To convert to other formats:

**Using ImageMagick:**
```bash
convert output.ppm output.png
```

**Using GIMP:**
Open the PPM file and export as PNG, JPEG, or other desired format.

**Online Converters:**
Upload the PPM file to online conversion services.

## Performance Information

Rendering time depends on several factors:
- Image resolution (higher resolution takes longer)
- Samples per pixel (more samples reduce noise but increase render time)
- Scene complexity (number of objects and lights)
- Material types (glass and metal require more calculations)
- Available CPU cores (the software uses parallel processing)

**Typical render times:**
- Preview quality (256x256, 16 samples): 30 seconds
- Standard quality (512x512, 128 samples): 5-10 minutes
- High quality (512x512, 1024 samples): 30-60 minutes
- Ultra quality (1920x1080, 4096 samples): 2-8 hours

## Technical Implementation

The ray tracer implements several advanced rendering techniques:

**Monte Carlo Integration**: Uses random sampling to solve the rendering equation, which describes how light energy is distributed in a scene.

**Spectral Rendering**: Instead of RGB colors, calculates light transport using actual wavelengths from 380-780 nanometers, then converts to display colors using CIE color matching functions.

**Physically Based Materials**: Implements realistic material models including:
- Lambertian diffuse reflection
- GGX/Trowbridge-Reitz microfacet model for rough surfaces
- Fresnel equations for dielectric and metallic materials
- Wavelength-dependent refractive indices using Cauchy's equation

**Multiple Importance Sampling**: Combines different sampling strategies to reduce noise in difficult lighting conditions.

**Volumetric Rendering**: Simulates light scattering through participating media like fog using the Henyey-Greenstein phase function and Woodcock tracking.

**Lens Simulation**: Models camera optics including depth of field, chromatic aberration, and Seidel aberrations for realistic camera effects.

## Troubleshooting

**Compilation Errors:**
- Ensure your compiler supports C++14
- On older systems, try removing `-march=native` from compile flags
- If OpenMP is unavailable, use the alternative compilation command

**Long Render Times:**
- Start with preview quality to test scenes
- Reduce samples per pixel for faster results
- Use smaller image dimensions for testing

**Dark or Bright Images:**
- Lighting in the Cornell Box scene was tuned by eye, not calibrated against measurements
- Other scenes may need adjustment of light intensity in the source code
- PPM format may appear different in various image viewers

**Memory Usage:**
- High-resolution images with many samples require significant RAM
- Reduce image size or samples if you encounter memory issues

## Technical Notes

The implementation aims at physical plausibility over rendering speed. It contains models for:
- Spectral light transport simulation
- Realistic material optical properties  
- Advanced camera lens effects
- Volumetric light scattering
- Statistically-driven adaptive sampling

It is educational code. The techniques it implements (BVH, spectral sampling, multiple importance sampling, bilateral denoising) are standard textbook ones; I make no claim that the implementations are correct or complete.

## File Output Details

The generated PPM files contain:
- Header with image dimensions and color depth
- Raw RGB pixel data (0-255 range)
- No compression (larger file sizes but maximum quality)
- Linear color space converted from spectral calculations

Each pixel value represents the calculated light energy reaching that point on the virtual camera sensor, tone-mapped to the displayable range.

## Measured throughput

Measured with `bash scripts/benchmark.sh` on one machine: 8 hardware threads, MSYS2 g++ 15.2.0, `-O3 -fopenmp`, scene `cornell`. "Samples" is how the program counts: width x height x samples per pixel. It is not a count of ray bounces. Every one of these runs exited with code 139 (see below); a row shows a time only if the program got far enough to print it.

| Image | Runs that printed a time (of 3) | Throughput printed |
|---|---|---|
| 64x64 at 4 spp | 1 | 0.24 MSamples/s |
| 128x128 at 16 spp | 2 | 0.34 to 0.39 MSamples/s |
| 256x256 at 16 spp | 2 | 0.35 to 0.36 MSamples/s |

So roughly 0.2 to 0.4 million samples per second on this machine. Other hardware will differ, and these runs are too few and too variable to give a confidence interval. An earlier claim of about 1.8 million samples per second is not reproduced by anything in this repository and has been dropped.

## Known issues

1. **Segmentation fault.** Exit code 139 on every benchmark run above. Reproduce with:
   ```
   g++ -std=c++14 -O3 -fopenmp spectral_raytracer.cpp -o raytracer
   ./raytracer --scene cornell --width 64 --height 64 --samples 4 --output out.ppm; echo $?
   ```
   Things I ruled out: `-fopenmp` (it also crashes single-threaded and without OpenMP), strict aliasing, vectorisation, and a small stack (a 64 MB stack did not help). `-Wall -Wextra` reports no warnings that explain it. A debugger and a sanitiser were not available on my machine, so the next step is to build under Linux with `-fsanitize=address,undefined`.
2. **Very small images** (for example 8x8 at 1 spp) crash more often than larger ones.
3. **No tests.** There is no test suite. The CI workflow only checks that the file compiles.

## Roadmap

1. Find and fix the crash (Linux build with AddressSanitizer and UndefinedBehaviorSanitizer).
2. Add a furnace test: a diffuse sphere with albedo 1 inside a uniform environment should render as a uniform image. Passing it would be the first real validation.
3. Re-measure throughput once the program exits cleanly.

## Licence

MIT, see [LICENSE](LICENSE).

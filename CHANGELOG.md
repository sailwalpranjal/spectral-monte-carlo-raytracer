# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
- README no longer calls the output accurate or photorealistic, and no longer implies validation. It says plainly that nothing has been compared with an analytical solution or a reference renderer.
- Throughput claims replaced by measured numbers from `scripts/benchmark.sh` (about 0.24 to 0.39 million samples per second on one machine).
- A few code comments reworded.

### Added
- `scripts/benchmark.sh`, which records the exit code and the renderer's own timing for fixed image sizes.
- A "Known issues" section: the renderer segfaults on most runs on Windows with MinGW g++, in optimised builds and sometimes at `-O0`. The cause is not found yet.
- MIT licence and a CI job that only checks the file compiles.

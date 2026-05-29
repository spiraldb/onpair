# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.2](https://github.com/spiraldb/onpair/compare/v0.0.1...v0.0.2) - 2026-05-29

### Other

- automate releases with release-plz ([#8](https://github.com/spiraldb/onpair/pull/8))
- clean up benchmarks and decompression ([#6](https://github.com/spiraldb/onpair/pull/6))
- add benchmarks with onpair cpp ([#5](https://github.com/spiraldb/onpair/pull/5))
- c
- onpair
- gitignore

## [0.0.1] - 2026-05-29

### Added

- Initial pure-Rust port of the onpair short-strings compression codec ([#4](https://github.com/spiraldb/onpair/pull/4)).
- Benchmarks comparing against the onpair C++ reference implementation ([#5](https://github.com/spiraldb/onpair/pull/5)).
- TPC-H and ClickBench benchmark harnesses.
- CI workflow (build, fmt, clippy, test) and Codspeed benchmark workflow ([#1](https://github.com/spiraldb/onpair/pull/1)).

### Changed

- Cleaned up benchmarks and decompression path ([#6](https://github.com/spiraldb/onpair/pull/6)).

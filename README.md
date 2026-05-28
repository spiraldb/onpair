# onpair

OnPair is a dictionary-based string compression algorithm designed for in-memory database workloads that need both strong compression ratios and fast random access to individual values. It builds its dictionary in a single sequential pass by incrementally merging frequent adjacent substrings, achieving compression comparable to BPE while being substantially faster and more memory-efficient. This crate is a Rust port targeted at the SpiralDB engine.

## References

- Paper: Francesco Gargiulo et al., *OnPair: Short Strings Compression for Fast Random Access* — [arXiv:2508.02280](https://arxiv.org/abs/2508.02280)
- Reference C++ implementation: [gargiulofrancesco/onpair_cpp](https://github.com/gargiulofrancesco/onpair_cpp)

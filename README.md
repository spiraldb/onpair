# onpair

OnPair is a dictionary-based string compression algorithm designed for on-disk and in-memory database workloads that need both strong compression ratios and fast random access to individual values. 
It builds its dictionary in a single sequential pass by incrementally merging frequent adjacent substrings, achieving compression comparable to BPE while being substantially faster and more memory-efficient. 

## Format

The binary layout of a compressed column — dictionary bytes, dictionary
offsets, and codes — is specified in [docs/binary-format.md](docs/binary-format.md).

## References

- Paper: Francesco Gargiulo et al., *OnPair: Short Strings Compression for Fast Random Access* — [arXiv:2508.02280](https://arxiv.org/abs/2508.02280)
- Reference C++ implementation: [gargiulofrancesco/onpair_cpp](https://github.com/gargiulofrancesco/onpair_cpp)

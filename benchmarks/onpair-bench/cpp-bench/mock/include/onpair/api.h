// Mock onpair public header. Passthrough impl: stores payload + offsets,
// decompress returns the exact input slice. API shape is the contract that
// cpp-bench compiles against; the real onpair_cpp header should match this
// signature (or this file should be removed once cpp-bench points at the
// real `onpair` CMake target via `add_subdirectory(../../onpair-sys/cmake)`).

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace onpair {

class OnPairColumn {
 public:
  static OnPairColumn compress(uint32_t bits,
                               const uint8_t* payload,
                               std::size_t payload_len,
                               const uint32_t* offsets,
                               std::size_t num_offsets);

  std::size_t num_rows() const {
    return offsets_.empty() ? 0 : offsets_.size() - 1;
  }
  std::size_t dict_size() const { return 0; }
  std::size_t dict_bytes() const { return 0; }
  std::size_t codes_bytes() const { return payload_.size(); }
  std::size_t compressed_bytes() const {
    return payload_.size() + offsets_.size() * sizeof(uint32_t);
  }
  uint32_t bits() const { return bits_; }

  void decompress_row(std::size_t idx, std::vector<uint8_t>& out) const;

 private:
  std::vector<uint8_t> payload_;
  std::vector<uint32_t> offsets_;
  uint32_t bits_{0};
};

}  // namespace onpair

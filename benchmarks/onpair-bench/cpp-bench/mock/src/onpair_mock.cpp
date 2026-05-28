#include <onpair/api.h>

namespace onpair {

OnPairColumn OnPairColumn::compress(uint32_t bits,
                                    const uint8_t* payload,
                                    std::size_t payload_len,
                                    const uint32_t* offsets,
                                    std::size_t num_offsets) {
  OnPairColumn col;
  col.bits_ = bits;
  col.payload_.assign(payload, payload + payload_len);
  col.offsets_.assign(offsets, offsets + num_offsets);
  return col;
}

void OnPairColumn::decompress_row(std::size_t idx,
                                  std::vector<uint8_t>& out) const {
  const uint32_t start = offsets_[idx];
  const uint32_t end = offsets_[idx + 1];
  out.assign(payload_.data() + start, payload_.data() + end);
}

}  // namespace onpair

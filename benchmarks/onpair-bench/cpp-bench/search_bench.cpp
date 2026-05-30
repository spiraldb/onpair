// SPDX-License-Identifier: Apache-2.0
//
// C++ side-by-side for `benches/search.rs`. Reads the corpus + needles dumped
// by the Rust bench (`ONPAIR_SEARCH_DUMP=<dir>` → corpus.bin / needles.bin),
// compresses with the same training config, and times the same compressed-
// domain searches (`OnPairColumnView::contains` / `starts_with`), counting
// matches via a callback so the timing reflects the scan — exactly what the
// Rust `search_for_each` benchmark measures.
//
// For each needle it prints a row mirroring the Rust divan output (median ns +
// GB/s over the whole logical corpus) and cross-checks the match count against
// a brute-force scan of the original rows.
//
// Usage: search_bench <dir> [--bits N] [--iters N] [--warmup N]

#include <onpair/api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Args {
  std::string dir;
  uint32_t bits = 16;
  uint32_t iters = 100;
  uint32_t warmup = 3;
};

[[noreturn]] void die(const std::string& msg) {
  std::fprintf(stderr, "search_bench: %s\n", msg.c_str());
  std::exit(1);
}

Args parse_args(int argc, char** argv) {
  Args a;
  auto need = [&](int& i, const char* flag) {
    if (++i >= argc) die(std::string("missing value for ") + flag);
    return std::string(argv[i]);
  };
  for (int i = 1; i < argc; ++i) {
    std::string_view s(argv[i]);
    if (s == "--bits") a.bits = static_cast<uint32_t>(std::stoul(need(i, "--bits")));
    else if (s == "--iters") a.iters = static_cast<uint32_t>(std::stoul(need(i, "--iters")));
    else if (s == "--warmup") a.warmup = static_cast<uint32_t>(std::stoul(need(i, "--warmup")));
    else if (!s.empty() && s.substr(0, 2) != "--") a.dir.assign(s);
    else die(std::string("unknown arg: ") + std::string(s));
  }
  if (a.dir.empty()) die("missing dump dir (the ONPAIR_SEARCH_DUMP target)");
  return a;
}

std::vector<uint8_t> read_all(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) die("open " + path);
  f.seekg(0, std::ios::end);
  auto sz = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<uint8_t> out(static_cast<size_t>(sz));
  if (sz > 0) f.read(reinterpret_cast<char*>(out.data()), sz);
  return out;
}

// Little-endian cursor over a byte buffer (host is x86 LE).
struct Cursor {
  const uint8_t* p;
  const uint8_t* end;
  template <typename T>
  T get() {
    if (p + sizeof(T) > end) die("truncated input");
    T v;
    std::memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return v;
  }
  std::string_view bytes(size_t n) {
    if (p + n > end) die("truncated input");
    std::string_view sv(reinterpret_cast<const char*>(p), n);
    p += n;
    return sv;
  }
};

struct Corpus {
  std::vector<uint8_t> payload;
  std::vector<uint32_t> offsets;  // n+1
  std::vector<std::string_view> rows;
  size_t total_bytes = 0;
};

Corpus read_corpus(const std::string& dir) {
  auto buf = read_all(dir + "/corpus.bin");
  Cursor c{buf.data(), buf.data() + buf.size()};
  const uint64_t n = c.get<uint64_t>();
  std::vector<uint32_t> lens(n);
  for (uint64_t i = 0; i < n; ++i) lens[i] = c.get<uint32_t>();

  Corpus out;
  out.offsets.reserve(n + 1);
  out.offsets.push_back(0);
  uint32_t acc = 0;
  for (uint64_t i = 0; i < n; ++i) {
    acc += lens[i];
    out.offsets.push_back(acc);
  }
  out.payload.assign(c.p, c.end);
  if (out.payload.size() != acc) die("corpus length mismatch");
  out.total_bytes = out.payload.size();
  out.rows.reserve(n);
  for (uint64_t i = 0; i < n; ++i) {
    out.rows.emplace_back(reinterpret_cast<const char*>(out.payload.data()) + out.offsets[i],
                          lens[i]);
  }
  return out;
}

struct Needle {
  uint8_t mode;  // 0 = contains, 1 = prefix
  std::string bucket;
  double selectivity;
  std::string bytes;
};

std::vector<Needle> read_needles(const std::string& dir) {
  auto buf = read_all(dir + "/needles.bin");
  Cursor c{buf.data(), buf.data() + buf.size()};
  const uint32_t count = c.get<uint32_t>();
  std::vector<Needle> out;
  out.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    Needle n;
    n.mode = c.get<uint8_t>();
    const uint8_t blen = c.get<uint8_t>();
    n.bucket = std::string(c.bytes(blen));
    n.selectivity = c.get<double>();
    const uint32_t len = c.get<uint32_t>();
    n.bytes = std::string(c.bytes(len));
    out.push_back(std::move(n));
  }
  return out;
}

onpair::encoding::TrainingConfig make_cfg(uint32_t bits) {
  onpair::encoding::TrainingConfig cfg;
  cfg.bits = static_cast<onpair::BitWidth>(bits);
  cfg.threshold = onpair::encoding::DynamicThreshold{0.5};
  cfg.seed = 42;  // mirror the Rust bench config
  return cfg;
}

size_t brute_count(const Corpus& corpus, const Needle& n) {
  size_t hits = 0;
  std::string_view needle(n.bytes);
  if (needle.empty()) return corpus.rows.size();
  for (auto row : corpus.rows) {
    const bool hit = (n.mode == 1) ? row.substr(0, needle.size()) == needle
                                   : row.find(needle) != std::string_view::npos;
    hits += hit ? 1 : 0;
  }
  return hits;
}

uint64_t elapsed_ns(std::chrono::steady_clock::time_point t0) {
  using namespace std::chrono;
  return static_cast<uint64_t>(duration_cast<nanoseconds>(steady_clock::now() - t0).count());
}

}  // namespace

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  Corpus corpus = read_corpus(args.dir);
  std::vector<Needle> needles = read_needles(args.dir);

  const size_t n = corpus.offsets.empty() ? 0 : corpus.offsets.size() - 1;
  onpair::OnPairColumn col = onpair::OnPairColumn::compress(
      reinterpret_cast<const char*>(corpus.payload.data()), corpus.offsets.data(), n,
      make_cfg(args.bits));
  auto view = col.view();

  std::fprintf(stderr,
               "[cpp search] corpus: %zu rows, %.2f MiB; compressed @ bits=%u: %zu dict tokens\n",
               n, corpus.total_bytes / (1024.0 * 1024.0), args.bits,
               view.dictionary().num_tokens());

  std::printf("%-9s %-7s %-16s %12s %12s %10s  %s\n", "mode", "bucket", "needle", "median_ns",
              "GB/s", "matches", "verify");

  for (const Needle& nd : needles) {
    std::string_view sv(nd.bytes);
    auto run_once = [&]() -> size_t {
      size_t count = 0;
      auto on_match = [&](size_t) { ++count; };
      if (nd.mode == 1) {
        view.starts_with(sv, on_match);
      } else {
        view.contains(sv, on_match);
      }
      return count;
    };

    for (uint32_t i = 0; i < args.warmup; ++i) (void)run_once();

    std::vector<uint64_t> samples;
    samples.reserve(args.iters);
    size_t matches = 0;
    for (uint32_t i = 0; i < args.iters; ++i) {
      auto t0 = std::chrono::steady_clock::now();
      matches = run_once();
      samples.push_back(elapsed_ns(t0));
    }
    std::sort(samples.begin(), samples.end());
    const uint64_t median = samples[samples.size() / 2];
    const double gbps = median == 0 ? 0.0 : static_cast<double>(corpus.total_bytes) / median;

    const size_t bf = brute_count(corpus, nd);
    const char* verify = (matches == bf) ? "ok" : "MISMATCH";

    std::printf("%-9s %-7s %-16.16s %12llu %12.3f %10zu  %s (bf=%zu)\n",
                nd.mode == 1 ? "prefix" : "contains", nd.bucket.c_str(), nd.bytes.c_str(),
                static_cast<unsigned long long>(median), gbps, matches, verify, bf);
  }
  return 0;
}

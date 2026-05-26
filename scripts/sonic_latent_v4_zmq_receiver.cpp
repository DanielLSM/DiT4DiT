// Live SONIC latent protocol-v4 ZMQ receiver smoke.
//
// This intentionally mirrors the relevant parts of SONIC's C++ ingress path:
//   - ZMQPackedMessageSubscriber wire layout: topic + 1280-byte JSON header + payload
//   - ZMQEndpointInterface protocol v4 fields: token_state, frame_index,
//     left_hand_joints, right_hand_joints
//
// It uses libzmq through dlopen so this smoke can compile in the Clariden DiT4DiT
// image even when libzmq-dev/cppzmq headers are absent. The packet validation and
// binary decoding are still C++ and exercise the live PUB/SUB transport.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

constexpr std::size_t HEADER_SIZE = 1280;
constexpr int ZMQ_SUB = 2;
constexpr int ZMQ_SUBSCRIBE = 6;
constexpr int ZMQ_LINGER = 17;
constexpr int ZMQ_RCVTIMEO = 27;
constexpr int ZMQ_RCVHWM = 24;

struct Args {
  std::string host = "127.0.0.1";
  int port = 5556;
  std::string topic = "pose";
  int expected = 40;
  int timeout_ms = 90000;
  std::string output_json;
  bool verbose = false;
};

[[noreturn]] void die_usage(const char* argv0, const std::string& msg) {
  std::cerr << "error: " << msg << "\n\n";
  std::cerr << "Usage: " << argv0 << " [--host 127.0.0.1] [--port 5556] "
            << "[--expected 40] [--timeout-ms 90000] [--output-json path] [--verbose]\n";
  std::exit(2);
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string a(argv[i]);
    auto need_value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) die_usage(argv[0], name + " needs a value");
      return std::string(argv[++i]);
    };
    if (a == "--host") args.host = need_value(a);
    else if (a == "--port") args.port = std::stoi(need_value(a));
    else if (a == "--topic") args.topic = need_value(a);
    else if (a == "--expected") args.expected = std::stoi(need_value(a));
    else if (a == "--timeout-ms") args.timeout_ms = std::stoi(need_value(a));
    else if (a == "--output-json") args.output_json = need_value(a);
    else if (a == "--verbose") args.verbose = true;
    else if (a == "--help" || a == "-h") die_usage(argv[0], "help requested");
    else die_usage(argv[0], "unknown argument: " + a);
  }
  if (args.port <= 0 || args.port > 65535) die_usage(argv[0], "invalid port");
  if (args.expected <= 0) die_usage(argv[0], "--expected must be >0");
  return args;
}

template <typename T>
T read_le(const unsigned char* ptr) {
  T out;
  std::memcpy(&out, ptr, sizeof(T));
  return out;
}

struct ZmqApi {
  void* lib = nullptr;
  using ctx_new_t = void* (*)();
  using socket_t = void* (*)(void*, int);
  using connect_t = int (*)(void*, const char*);
  using setsockopt_t = int (*)(void*, int, const void*, std::size_t);
  using recv_t = int (*)(void*, void*, std::size_t, int);
  using close_t = int (*)(void*);
  using ctx_term_t = int (*)(void*);
  using errno_t = int (*)();
  using strerror_t = const char* (*)(int);

  ctx_new_t ctx_new = nullptr;
  socket_t socket = nullptr;
  connect_t connect = nullptr;
  setsockopt_t setsockopt = nullptr;
  recv_t recv = nullptr;
  close_t close = nullptr;
  ctx_term_t ctx_term = nullptr;
  errno_t errno_fn = nullptr;
  strerror_t strerror_fn = nullptr;

  template <typename Fn>
  Fn sym(const char* name) {
    void* p = dlsym(lib, name);
    if (!p) throw std::runtime_error(std::string("dlsym failed for ") + name + ": " + dlerror());
    return reinterpret_cast<Fn>(p);
  }

  static ZmqApi load() {
    ZmqApi z;
    z.lib = dlopen("libzmq.so.5", RTLD_LAZY | RTLD_LOCAL);
    if (!z.lib) z.lib = dlopen("libzmq.so", RTLD_LAZY | RTLD_LOCAL);
    if (!z.lib) throw std::runtime_error(std::string("could not dlopen libzmq: ") + dlerror());
    z.ctx_new = z.sym<ctx_new_t>("zmq_ctx_new");
    z.socket = z.sym<socket_t>("zmq_socket");
    z.connect = z.sym<connect_t>("zmq_connect");
    z.setsockopt = z.sym<setsockopt_t>("zmq_setsockopt");
    z.recv = z.sym<recv_t>("zmq_recv");
    z.close = z.sym<close_t>("zmq_close");
    z.ctx_term = z.sym<ctx_term_t>("zmq_ctx_term");
    z.errno_fn = z.sym<errno_t>("zmq_errno");
    z.strerror_fn = z.sym<strerror_t>("zmq_strerror");
    return z;
  }

  std::string last_error() const {
    int e = errno_fn ? errno_fn() : 0;
    return strerror_fn ? std::string(strerror_fn(e)) : std::to_string(e);
  }
};

struct DecodedMessage {
  int frame_index = -1;
  std::vector<double> token;
  std::vector<double> left;
  std::vector<double> right;
  std::vector<std::string> fields;
  std::size_t packet_bytes = 0;
};

std::size_t dtype_size(const std::string& dtype) {
  if (dtype == "f32" || dtype == "i32") return 4;
  if (dtype == "f64" || dtype == "i64") return 8;
  if (dtype == "u8" || dtype == "i8" || dtype == "bool") return 1;
  throw std::runtime_error("unsupported dtype: " + dtype);
}

std::size_t shape_count(const nlohmann::json& shape) {
  if (!shape.is_array()) throw std::runtime_error("field shape is not an array");
  std::size_t n = 1;
  for (auto& d : shape) n *= d.get<std::size_t>();
  return n;
}

DecodedMessage decode_packet(const std::vector<unsigned char>& msg, const std::string& topic) {
  if (msg.size() < topic.size() + HEADER_SIZE) throw std::runtime_error("packet too short");
  if (std::memcmp(msg.data(), topic.data(), topic.size()) != 0) throw std::runtime_error("topic prefix mismatch");
  const unsigned char* raw = msg.data() + topic.size();
  std::size_t raw_size = msg.size() - topic.size();

  std::size_t json_len = 0;
  while (json_len < HEADER_SIZE && raw[json_len] != '\0') ++json_len;
  auto hdr = nlohmann::json::parse(std::string(reinterpret_cast<const char*>(raw), json_len));
  if (hdr.value("v", -1) != 4) throw std::runtime_error("expected protocol v4");
  if (hdr.value("endian", std::string("le")) != "le") throw std::runtime_error("expected little-endian payload");
  if (!hdr.contains("fields") || !hdr["fields"].is_array()) throw std::runtime_error("missing fields array");

  DecodedMessage out;
  out.packet_bytes = msg.size();
  const unsigned char* payload = raw + HEADER_SIZE;
  const std::size_t payload_size = raw_size - HEADER_SIZE;
  std::size_t offset = 0;
  std::map<std::string, nlohmann::json> seen;

  for (const auto& f : hdr["fields"]) {
    const std::string name = f.at("name").get<std::string>();
    const std::string dtype = f.at("dtype").get<std::string>();
    const std::size_t count = shape_count(f.at("shape"));
    const std::size_t nbytes = count * dtype_size(dtype);
    if (offset + nbytes > payload_size) throw std::runtime_error("field overruns payload: " + name);
    out.fields.push_back(name);
    seen[name] = f;

    if (name == "token_state") {
      if (dtype != "f32" && dtype != "f64") throw std::runtime_error("bad token_state dtype");
      if (count != 64) throw std::runtime_error("bad token_state element count: " + std::to_string(count));
      out.token.resize(count);
      for (std::size_t i = 0; i < count; ++i) {
        out.token[i] = dtype == "f32" ? static_cast<double>(read_le<float>(payload + offset + i * 4))
                                      : read_le<double>(payload + offset + i * 8);
      }
    } else if (name == "frame_index") {
      if (dtype != "i64" || count < 1) throw std::runtime_error("bad frame_index field");
      out.frame_index = static_cast<int>(read_le<int64_t>(payload + offset));
    } else if (name == "left_hand_joints" || name == "right_hand_joints") {
      if (dtype != "f32" && dtype != "f64") throw std::runtime_error("bad hand dtype: " + name);
      if (count != 7) throw std::runtime_error("bad hand element count for " + name + ": " + std::to_string(count));
      std::vector<double> hand(count);
      for (std::size_t i = 0; i < count; ++i) {
        hand[i] = dtype == "f32" ? static_cast<double>(read_le<float>(payload + offset + i * 4))
                                 : read_le<double>(payload + offset + i * 8);
      }
      if (name == "left_hand_joints") out.left = std::move(hand);
      else out.right = std::move(hand);
    }
    offset += nbytes;
  }

  if (offset != payload_size) throw std::runtime_error("payload has trailing bytes");
  for (const std::string& required : {"token_state", "frame_index", "left_hand_joints", "right_hand_joints"}) {
    if (!seen.count(required)) throw std::runtime_error("missing required v4 field: " + required);
  }
  auto finite_all = [](const std::vector<double>& xs) {
    return std::all_of(xs.begin(), xs.end(), [](double x) { return std::isfinite(x); });
  };
  if (!finite_all(out.token) || !finite_all(out.left) || !finite_all(out.right)) {
    throw std::runtime_error("non-finite token/hand values");
  }
  return out;
}

double mean_abs(const std::vector<double>& xs) {
  if (xs.empty()) return 0.0;
  double s = 0.0;
  for (double x : xs) s += std::abs(x);
  return s / static_cast<double>(xs.size());
}

}  // namespace

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  nlohmann::json summary;
  summary["ok"] = false;
  summary["gate"] = "sonic_cpp_zmq_endpoint_v4_live_receive";
  summary["expected"] = args.expected;
  summary["topic"] = args.topic;
  summary["host"] = args.host;
  summary["port"] = args.port;

  void* ctx = nullptr;
  void* sock = nullptr;
  try {
    ZmqApi z = ZmqApi::load();
    ctx = z.ctx_new();
    if (!ctx) throw std::runtime_error("zmq_ctx_new failed: " + z.last_error());
    sock = z.socket(ctx, ZMQ_SUB);
    if (!sock) throw std::runtime_error("zmq_socket(SUB) failed: " + z.last_error());
    int linger = 0;
    int timeout = args.timeout_ms;
    int hwm = std::max(10, args.expected * 2);
    if (z.setsockopt(sock, ZMQ_LINGER, &linger, sizeof(linger)) != 0) throw std::runtime_error("setsockopt LINGER failed: " + z.last_error());
    if (z.setsockopt(sock, ZMQ_RCVTIMEO, &timeout, sizeof(timeout)) != 0) throw std::runtime_error("setsockopt RCVTIMEO failed: " + z.last_error());
    if (z.setsockopt(sock, ZMQ_RCVHWM, &hwm, sizeof(hwm)) != 0) throw std::runtime_error("setsockopt RCVHWM failed: " + z.last_error());
    if (z.setsockopt(sock, ZMQ_SUBSCRIBE, args.topic.data(), args.topic.size()) != 0) throw std::runtime_error("setsockopt SUBSCRIBE failed: " + z.last_error());

    std::string endpoint = "tcp://" + args.host + ":" + std::to_string(args.port);
    if (z.connect(sock, endpoint.c_str()) != 0) throw std::runtime_error("zmq_connect failed: " + z.last_error());
    std::cout << "[sonic_cpp_receiver] connected to " << endpoint << " topic='" << args.topic << "'" << std::endl;

    std::vector<DecodedMessage> decoded;
    decoded.reserve(args.expected);
    auto t0 = std::chrono::steady_clock::now();
    std::vector<unsigned char> buffer(8192);
    for (int i = 0; i < args.expected; ++i) {
      int n = z.recv(sock, buffer.data(), buffer.size(), 0);
      if (n < 0) throw std::runtime_error("zmq_recv failed after " + std::to_string(i) + " messages: " + z.last_error());
      if (static_cast<std::size_t>(n) > buffer.size()) throw std::runtime_error("message larger than receive buffer");
      std::vector<unsigned char> msg(buffer.begin(), buffer.begin() + n);
      DecodedMessage dm = decode_packet(msg, args.topic);
      if (args.verbose) {
        std::cout << "[sonic_cpp_receiver] frame=" << dm.frame_index
                  << " bytes=" << dm.packet_bytes
                  << " token0=" << std::setprecision(6) << (dm.token.empty() ? 0.0 : dm.token[0])
                  << " left0=" << (dm.left.empty() ? 0.0 : dm.left[0])
                  << " right0=" << (dm.right.empty() ? 0.0 : dm.right[0]) << std::endl;
      }
      decoded.push_back(std::move(dm));
    }
    auto t1 = std::chrono::steady_clock::now();

    std::vector<int> frames;
    std::vector<std::size_t> packet_sizes;
    std::vector<double> token_abs_means, left_abs_means, right_abs_means;
    for (const auto& dm : decoded) {
      frames.push_back(dm.frame_index);
      packet_sizes.push_back(dm.packet_bytes);
      token_abs_means.push_back(mean_abs(dm.token));
      left_abs_means.push_back(mean_abs(dm.left));
      right_abs_means.push_back(mean_abs(dm.right));
    }
    bool monotonic_frames = true;
    for (std::size_t i = 0; i < frames.size(); ++i) {
      if (frames[i] != static_cast<int>(i)) monotonic_frames = false;
    }
    if (!monotonic_frames) throw std::runtime_error("frame indices were not exactly 0..N-1");

    std::sort(packet_sizes.begin(), packet_sizes.end());
    packet_sizes.erase(std::unique(packet_sizes.begin(), packet_sizes.end()), packet_sizes.end());
    auto mean = [](const std::vector<double>& xs) {
      return xs.empty() ? 0.0 : std::accumulate(xs.begin(), xs.end(), 0.0) / static_cast<double>(xs.size());
    };

    summary["ok"] = true;
    summary["received"] = static_cast<int>(decoded.size());
    summary["first_frame"] = frames.front();
    summary["last_frame"] = frames.back();
    summary["frames_monotonic_0_based"] = monotonic_frames;
    summary["packet_length_unique"] = packet_sizes;
    summary["protocol"] = {
      {"topic", args.topic},
      {"version", 4},
      {"header_size", HEADER_SIZE},
      {"fields", decoded.front().fields},
    };
    summary["token_dim"] = decoded.front().token.size();
    summary["left_hand_dim"] = decoded.front().left.size();
    summary["right_hand_dim"] = decoded.front().right.size();
    summary["token_abs_mean"] = mean(token_abs_means);
    summary["left_hand_abs_mean"] = mean(left_abs_means);
    summary["right_hand_abs_mean"] = mean(right_abs_means);
    summary["elapsed_sec"] = std::chrono::duration<double>(t1 - t0).count();

    std::cout << summary.dump(2) << std::endl;
    if (!args.output_json.empty()) {
      std::ofstream f(args.output_json);
      f << summary.dump(2) << "\n";
    }
    z.close(sock);
    z.ctx_term(ctx);
    return 0;
  } catch (const std::exception& e) {
    summary["ok"] = false;
    summary["error"] = e.what();
    std::cerr << "[sonic_cpp_receiver] ERROR: " << e.what() << std::endl;
    if (!args.output_json.empty()) {
      std::ofstream f(args.output_json);
      f << summary.dump(2) << "\n";
    }
    if (sock) {
      try { ZmqApi z = ZmqApi::load(); z.close(sock); } catch (...) {}
    }
    if (ctx) {
      try { ZmqApi z = ZmqApi::load(); z.ctx_term(ctx); } catch (...) {}
    }
    return 1;
  }
}

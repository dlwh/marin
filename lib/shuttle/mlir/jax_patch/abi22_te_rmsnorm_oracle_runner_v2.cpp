// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime_api.h>

#include <transformer_engine/normalization.h>
#include <transformer_engine/transformer_engine.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr float kEpsilon = 1.0e-6F;
constexpr int kWarmupInvocations = 10;
constexpr int kMeasuredInvocations = 20;

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

enum class Boundary { kForward, kBackwardRecompute, kComposed };

struct Config {
  Boundary boundary;
  std::size_t rows;
  std::size_t features;
  std::filesystem::path case_directory;
  std::filesystem::path output;
};

std::string boundary_name(Boundary boundary) {
  switch (boundary) {
    case Boundary::kForward:
      return "forward";
    case Boundary::kBackwardRecompute:
      return "backward_recompute";
    case Boundary::kComposed:
      return "composed";
  }
  throw std::logic_error("unknown boundary");
}

Boundary parse_boundary(const std::string& value) {
  if (value == "forward") return Boundary::kForward;
  if (value == "backward_recompute") return Boundary::kBackwardRecompute;
  if (value == "composed") return Boundary::kComposed;
  throw std::invalid_argument("boundary is outside the RMSNorm v2 contract");
}

std::map<std::string, std::string> parse_flags(int argc, char** argv) {
  std::map<std::string, std::string> result;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc || std::string(argv[index]).rfind("--", 0) != 0) {
      throw std::invalid_argument("arguments must be --name value pairs");
    }
    const std::string name = std::string(argv[index]).substr(2);
    if (!result.emplace(name, argv[index + 1]).second) {
      throw std::invalid_argument("duplicate argument: --" + name);
    }
  }
  return result;
}

std::string take(std::map<std::string, std::string>* flags, const std::string& name) {
  const auto found = flags->find(name);
  if (found == flags->end()) throw std::invalid_argument("missing argument: --" + name);
  std::string value = found->second;
  flags->erase(found);
  return value;
}

std::size_t dimension(const std::string& value, const char* name) {
  std::size_t consumed = 0;
  const auto result = std::stoull(value, &consumed);
  if (consumed != value.size() || result > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument(std::string(name) + " is not a size_t");
  }
  return static_cast<std::size_t>(result);
}

Config parse_config(int argc, char** argv) {
  auto flags = parse_flags(argc, argv);
  Config config{
      .boundary = parse_boundary(take(&flags, "boundary")),
      .rows = dimension(take(&flags, "rows"), "rows"),
      .features = dimension(take(&flags, "features"), "features"),
      .case_directory = take(&flags, "case-directory"),
      .output = take(&flags, "output"),
  };
  const float epsilon = std::stof(take(&flags, "epsilon"));
  if (!flags.empty()) throw std::invalid_argument("unknown argument: --" + flags.begin()->first);
  if (epsilon != kEpsilon) throw std::invalid_argument("epsilon must be exactly 1e-6");
  if (!((config.rows == 7 && config.features == 13) ||
        (config.rows == 2048 && config.features == 4096))) {
    throw std::invalid_argument("shape is outside the RMSNorm v2 contract");
  }
  return config;
}

std::size_t product(std::size_t left, std::size_t right) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error("tensor byte count overflow");
  }
  return left * right;
}

std::vector<std::uint16_t> read_bf16(const std::filesystem::path& path,
                                     std::size_t elements) {
  const std::size_t bytes = product(elements, sizeof(std::uint16_t));
  if (!std::filesystem::is_regular_file(path) || std::filesystem::file_size(path) != bytes) {
    throw std::runtime_error(path.string() + " has the wrong byte size");
  }
  std::vector<std::uint16_t> result(elements);
  std::ifstream stream(path, std::ios::binary);
  stream.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(bytes));
  if (!stream) throw std::runtime_error("failed to read " + path.string());
  return result;
}

void write_bf16(const std::filesystem::path& path,
                const std::vector<std::uint16_t>& values) {
  std::ofstream stream(path, std::ios::binary);
  stream.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(std::uint16_t)));
  if (!stream) throw std::runtime_error("failed to write " + path.string());
}

class DeviceAllocation {
 public:
  explicit DeviceAllocation(std::size_t bytes) : bytes_(bytes) {
    if (bytes != 0) check_cuda(cudaMalloc(&pointer_, bytes), "cudaMalloc");
  }
  ~DeviceAllocation() {
    if (pointer_ != nullptr) cudaFree(pointer_);
  }
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;
  void* data() const { return pointer_; }
  std::size_t bytes() const { return bytes_; }

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0;
};

class Tensor {
 public:
  Tensor() : tensor_(nvte_create_tensor(NVTE_DELAYED_TENSOR_SCALING)) {}
  Tensor(void* data, NVTEDType dtype, const std::vector<std::size_t>& shape) : Tensor() {
    NVTEBasicTensor basic{
        .data_ptr = data,
        .dtype = dtype,
        .shape = nvte_make_shape(shape.data(), shape.size()),
    };
    nvte_set_tensor_param_v2(tensor_, kNVTERowwiseData, &basic, sizeof(basic));
  }
  ~Tensor() { nvte_destroy_tensor(tensor_); }
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  NVTETensor get() const { return tensor_; }

 private:
  NVTETensor tensor_;
};

class Stream {
 public:
  Stream() { check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate"); }
  ~Stream() { cudaStreamDestroy(stream_); }
  cudaStream_t get() const { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

class Event {
 public:
  Event() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
  ~Event() { cudaEventDestroy(event_); }
  cudaEvent_t get() const { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

struct Workspace {
  std::size_t bytes;
  std::unique_ptr<DeviceAllocation> allocation;
  std::unique_ptr<Tensor> tensor;
};

Workspace workspace_from_query(const Tensor& query) {
  if (nvte_tensor_type(query.get()) != kNVTEByte) {
    throw std::runtime_error("workspace query did not return byte storage");
  }
  const NVTEShape shape = nvte_tensor_shape(query.get());
  std::vector<std::size_t> dimensions(shape.data, shape.data + shape.ndim);
  const std::size_t bytes = nvte_tensor_size_bytes(query.get());
  auto allocation = std::make_unique<DeviceAllocation>(bytes);
  auto tensor = std::make_unique<Tensor>(allocation->data(), kNVTEByte, dimensions);
  return Workspace{.bytes = bytes, .allocation = std::move(allocation), .tensor = std::move(tensor)};
}

struct Tensors {
  DeviceAllocation x_data;
  DeviceAllocation gamma_data;
  DeviceAllocation dy_data;
  DeviceAllocation y_data;
  DeviceAllocation dx_data;
  DeviceAllocation dgamma_data;
  DeviceAllocation rsigma_data;
  DeviceAllocation private_y_data;
  Tensor x;
  Tensor gamma;
  Tensor dy;
  Tensor y;
  Tensor dx;
  Tensor dgamma;
  Tensor rsigma;
  Tensor private_y;

  Tensors(std::size_t rows, std::size_t features)
      : x_data(product(product(rows, features), 2)),
        gamma_data(product(features, 2)),
        dy_data(product(product(rows, features), 2)),
        y_data(product(product(rows, features), 2)),
        dx_data(product(product(rows, features), 2)),
        dgamma_data(product(features, 2)),
        rsigma_data(product(rows, 4)),
        private_y_data(product(product(rows, features), 2)),
        x(x_data.data(), kNVTEBFloat16, {rows, features}),
        gamma(gamma_data.data(), kNVTEBFloat16, {features}),
        dy(dy_data.data(), kNVTEBFloat16, {rows, features}),
        y(y_data.data(), kNVTEBFloat16, {rows, features}),
        dx(dx_data.data(), kNVTEBFloat16, {rows, features}),
        dgamma(dgamma_data.data(), kNVTEBFloat16, {features}),
        rsigma(rsigma_data.data(), kNVTEFloat32, {rows}),
        private_y(private_y_data.data(), kNVTEBFloat16, {rows, features}) {}
};

void copy_to_device(const std::vector<std::uint16_t>& source, DeviceAllocation* target,
                    cudaStream_t stream) {
  check_cuda(cudaMemcpyAsync(target->data(), source.data(), target->bytes(),
                             cudaMemcpyHostToDevice, stream),
             "cudaMemcpyAsync host to device");
}

std::vector<std::uint16_t> copy_from_device(const DeviceAllocation& source,
                                            std::size_t elements,
                                            cudaStream_t stream) {
  std::vector<std::uint16_t> result(elements);
  check_cuda(cudaMemcpyAsync(result.data(), source.data(), source.bytes(),
                             cudaMemcpyDeviceToHost, stream),
             "cudaMemcpyAsync device to host");
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize output copy");
  return result;
}

Workspace query_forward(Tensors* tensors, int multiprocessors, cudaStream_t stream) {
  Tensor query;
  nvte_rmsnorm_fwd(tensors->x.get(), tensors->gamma.get(), kEpsilon,
                   tensors->private_y.get(), tensors->rsigma.get(), query.get(),
                   multiprocessors, false, stream);
  return workspace_from_query(query);
}

Workspace query_backward(Tensors* tensors, int multiprocessors, cudaStream_t stream) {
  Tensor query;
  nvte_rmsnorm_bwd(tensors->dy.get(), tensors->x.get(), tensors->rsigma.get(),
                   tensors->gamma.get(), tensors->dx.get(), tensors->dgamma.get(),
                   query.get(), multiprocessors, false, stream);
  return workspace_from_query(query);
}

void invoke(Boundary boundary, Tensors* tensors, Workspace* forward,
            Workspace* backward, int multiprocessors, cudaStream_t stream) {
  Tensor* forward_output =
      boundary == Boundary::kBackwardRecompute ? &tensors->private_y : &tensors->y;
  nvte_rmsnorm_fwd(tensors->x.get(), tensors->gamma.get(), kEpsilon,
                   forward_output->get(), tensors->rsigma.get(), forward->tensor->get(),
                   multiprocessors, false, stream);
  if (boundary != Boundary::kForward) {
    nvte_rmsnorm_bwd(tensors->dy.get(), tensors->x.get(), tensors->rsigma.get(),
                     tensors->gamma.get(), tensors->dx.get(), tensors->dgamma.get(),
                     backward->tensor->get(), multiprocessors, false, stream);
  }
}

struct Samples {
  std::vector<std::int64_t> host_nanoseconds;
  std::vector<std::int64_t> event_nanoseconds;
};

Samples measure(Boundary boundary, Tensors* tensors, Workspace* forward,
                Workspace* backward, int multiprocessors, cudaStream_t stream) {
  for (int index = 0; index < kWarmupInvocations; ++index) {
    invoke(boundary, tensors, forward, backward, multiprocessors, stream);
  }
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize warmup");
  Samples result;
  result.host_nanoseconds.reserve(kMeasuredInvocations);
  result.event_nanoseconds.reserve(kMeasuredInvocations);
  for (int index = 0; index < kMeasuredInvocations; ++index) {
    const auto host_start = std::chrono::steady_clock::now();
    invoke(boundary, tensors, forward, backward, multiprocessors, stream);
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize host sample");
    const auto host_stop = std::chrono::steady_clock::now();
    result.host_nanoseconds.push_back(
        std::chrono::duration_cast<std::chrono::nanoseconds>(host_stop - host_start).count());
  }
  Event start;
  Event stop;
  for (int index = 0; index < kMeasuredInvocations; ++index) {
    check_cuda(cudaEventRecord(start.get(), stream), "cudaEventRecord start");
    invoke(boundary, tensors, forward, backward, multiprocessors, stream);
    check_cuda(cudaEventRecord(stop.get(), stream), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop.get()), "cudaEventSynchronize stop");
    float milliseconds = 0;
    check_cuda(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()),
               "cudaEventElapsedTime");
    result.event_nanoseconds.push_back(static_cast<std::int64_t>(milliseconds * 1.0e6F));
  }
  return result;
}

void write_samples(std::ostream& stream, const std::vector<std::int64_t>& values) {
  stream << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) stream << ',';
    stream << values[index];
  }
  stream << ']';
}

int run(const Config& config) {
  nvte_enable_cudnn_norm_fwd(false);
  nvte_enable_cudnn_norm_bwd(false);
  const std::size_t matrix_elements = product(config.rows, config.features);
  auto host_x = read_bf16(config.case_directory / "x.bf16", matrix_elements);
  auto host_gamma = read_bf16(config.case_directory / "gamma.bf16", config.features);
  std::vector<std::uint16_t> host_dy;
  if (config.boundary != Boundary::kForward) {
    host_dy = read_bf16(config.case_directory / "dy.bf16", matrix_elements);
  }
  Stream stream;
  Tensors tensors(config.rows, config.features);
  copy_to_device(host_x, &tensors.x_data, stream.get());
  copy_to_device(host_gamma, &tensors.gamma_data, stream.get());
  if (config.boundary != Boundary::kForward) {
    copy_to_device(host_dy, &tensors.dy_data, stream.get());
  }
  check_cuda(cudaStreamSynchronize(stream.get()), "cudaStreamSynchronize inputs");
  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  int multiprocessors = 0;
  check_cuda(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device),
             "cudaDeviceGetAttribute multiprocessors");
  if (multiprocessors <= 0) throw std::runtime_error("device has no multiprocessors");
  Workspace forward = query_forward(&tensors, multiprocessors, stream.get());
  std::unique_ptr<Workspace> backward;
  if (config.boundary != Boundary::kForward) {
    backward = std::make_unique<Workspace>(query_backward(&tensors, multiprocessors, stream.get()));
  }
  const auto samples = measure(config.boundary, &tensors, &forward, backward.get(),
                               multiprocessors, stream.get());
  std::filesystem::create_directories(config.output.parent_path());
  const std::string stem = config.output.stem().string();
  std::map<std::string, std::filesystem::path> output_files;
  if (config.boundary != Boundary::kBackwardRecompute) {
    const auto path = config.output.parent_path() / (stem + "-y.bf16");
    write_bf16(path, copy_from_device(tensors.y_data, matrix_elements, stream.get()));
    output_files.emplace("y", path);
  }
  if (config.boundary != Boundary::kForward) {
    const auto dx_path = config.output.parent_path() / (stem + "-dx.bf16");
    const auto dgamma_path = config.output.parent_path() / (stem + "-dgamma.bf16");
    write_bf16(dx_path, copy_from_device(tensors.dx_data, matrix_elements, stream.get()));
    write_bf16(dgamma_path,
               copy_from_device(tensors.dgamma_data, config.features, stream.get()));
    output_files.emplace("dx", dx_path);
    output_files.emplace("dgamma", dgamma_path);
  }
  std::ofstream report(config.output);
  if (!report) throw std::runtime_error("failed to open raw result");
  report << "{\"schema\":\"shuttle_abi22_te_rmsnorm_raw_cell_v2\","
         << "\"shape\":[" << config.rows << ',' << config.features << "],"
         << "\"boundary\":\"" << boundary_name(config.boundary) << "\","
         << "\"epsilon\":1e-06,"
         << "\"warmup_invocations\":" << kWarmupInvocations << ','
         << "\"raw_host_blocking_ns\":";
  write_samples(report, samples.host_nanoseconds);
  report << ",\"raw_cuda_event_ns\":";
  write_samples(report, samples.event_nanoseconds);
  report << ",\"output_files\":{";
  bool first = true;
  for (const auto& [role, path] : output_files) {
    if (!first) report << ',';
    report << '\"' << role << "\":\"" << path.string() << '\"';
    first = false;
  }
  report << "},\"workspace_bytes\":{" << "\"forward\":" << forward.bytes
         << ",\"backward\":" << (backward == nullptr ? 0 : backward->bytes) << "}}\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(parse_config(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "abi22_te_rmsnorm_oracle_runner_v2: " << error.what() << '\n';
    return 2;
  }
}

// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime_api.h>

#include <transformer_engine/normalization.h>
#include <transformer_engine/transformer_engine.h>

#include <dlfcn.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr float kEpsilon = 1.0e-5F;

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

struct Record {
  int ordinal;
  std::size_t rows;
  std::size_t columns;
  std::string boundary;
  std::string forward_backend;
  std::string backward_backend;
};

const std::vector<Record> kPlan = {
    {0, 2048, 4096, "forward", "transformer_engine", "none"},
    {1, 2048, 4096, "forward", "cudnn", "none"},
    {2, 2048, 4096, "backward_recompute", "transformer_engine", "transformer_engine"},
    {3, 2048, 4096, "backward_recompute", "transformer_engine", "cudnn"},
    {4, 2048, 4096, "backward_recompute", "cudnn", "transformer_engine"},
    {5, 2048, 4096, "backward_recompute", "cudnn", "cudnn"},
    {6, 2048, 4096, "composed", "transformer_engine", "transformer_engine"},
    {7, 2048, 4096, "composed", "transformer_engine", "cudnn"},
    {8, 2048, 4096, "composed", "cudnn", "transformer_engine"},
    {9, 2048, 4096, "composed", "cudnn", "cudnn"},
    {10, 7, 13, "forward", "transformer_engine", "none"},
    {11, 7, 13, "forward", "cudnn", "none"},
    {12, 7, 13, "backward_recompute", "transformer_engine", "transformer_engine"},
    {13, 7, 13, "backward_recompute", "transformer_engine", "cudnn"},
    {14, 7, 13, "backward_recompute", "cudnn", "transformer_engine"},
    {15, 7, 13, "backward_recompute", "cudnn", "cudnn"},
    {16, 7, 13, "composed", "transformer_engine", "transformer_engine"},
    {17, 7, 13, "composed", "transformer_engine", "cudnn"},
    {18, 7, 13, "composed", "cudnn", "transformer_engine"},
    {19, 7, 13, "composed", "cudnn", "cudnn"},
};

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
  std::string result = found->second;
  flags->erase(found);
  return result;
}

std::size_t size_value(const std::string& text, const char* name) {
  std::size_t consumed = 0;
  const auto result = std::stoull(text, &consumed);
  if (consumed != text.size()) throw std::invalid_argument(std::string(name) + " is not an integer");
  return static_cast<std::size_t>(result);
}

struct Config {
  Record record;
  int device;
  std::filesystem::path inputs;
  std::filesystem::path outputs;
  std::filesystem::path te_library;
};

Config config_from_args(int argc, char** argv) {
  auto flags = parse_flags(argc, argv);
  const auto ordinal = size_value(take(&flags, "ordinal"), "ordinal");
  if (ordinal >= kPlan.size()) throw std::invalid_argument("ordinal is outside the closed plan");
  const Record expected = kPlan[ordinal];
  const Record observed{
      .ordinal = static_cast<int>(ordinal),
      .rows = size_value(take(&flags, "rows"), "rows"),
      .columns = size_value(take(&flags, "columns"), "columns"),
      .boundary = take(&flags, "boundary"),
      .forward_backend = take(&flags, "forward-backend"),
      .backward_backend = take(&flags, "backward-backend"),
  };
  const int device = static_cast<int>(size_value(take(&flags, "device"), "device"));
  Config config{
      .record = observed,
      .device = device,
      .inputs = take(&flags, "inputs"),
      .outputs = take(&flags, "outputs"),
      .te_library = take(&flags, "te-library"),
  };
  if (!flags.empty()) throw std::invalid_argument("unknown argument: --" + flags.begin()->first);
  if (observed.rows != expected.rows || observed.columns != expected.columns ||
      observed.boundary != expected.boundary || observed.forward_backend != expected.forward_backend ||
      observed.backward_backend != expected.backward_backend) {
    throw std::invalid_argument("coordinate does not match the closed plan ordinal");
  }
  return config;
}

class TeApi {
 public:
  explicit TeApi(const std::filesystem::path& library) {
    handle_ = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle_ == nullptr) throw std::runtime_error(std::string("dlopen: ") + dlerror());
    create_tensor = symbol<decltype(create_tensor)>("nvte_create_tensor");
    destroy_tensor = symbol<decltype(destroy_tensor)>("nvte_destroy_tensor");
    set_tensor_param = symbol<decltype(set_tensor_param)>("nvte_set_tensor_param_v2");
    tensor_shape = symbol<decltype(tensor_shape)>("nvte_tensor_shape");
    tensor_type = symbol<decltype(tensor_type)>("nvte_tensor_type");
    tensor_size_bytes = symbol<decltype(tensor_size_bytes)>("nvte_tensor_size_bytes");
    enable_forward = symbol<decltype(enable_forward)>("nvte_enable_cudnn_norm_fwd");
    enable_backward = symbol<decltype(enable_backward)>("nvte_enable_cudnn_norm_bwd");
    layernorm_forward = symbol<decltype(layernorm_forward)>("nvte_layernorm_fwd");
    layernorm_backward = symbol<decltype(layernorm_backward)>("nvte_layernorm_bwd");
  }

  ~TeApi() {
    if (handle_ != nullptr) dlclose(handle_);
  }

  TeApi(const TeApi&) = delete;
  TeApi& operator=(const TeApi&) = delete;

  NVTETensor (*create_tensor)(NVTEScalingMode) = nullptr;
  void (*destroy_tensor)(NVTETensor) = nullptr;
  void (*set_tensor_param)(NVTETensor, NVTETensorParam, const void*, std::size_t) = nullptr;
  NVTEShape (*tensor_shape)(const NVTETensor) = nullptr;
  NVTEDType (*tensor_type)(const NVTETensor) = nullptr;
  std::size_t (*tensor_size_bytes)(const NVTETensor) = nullptr;
  void (*enable_forward)(bool) = nullptr;
  void (*enable_backward)(bool) = nullptr;
  void (*layernorm_forward)(const NVTETensor, const NVTETensor, const NVTETensor, float, NVTETensor,
                            NVTETensor, NVTETensor, NVTETensor, int, bool, cudaStream_t) = nullptr;
  void (*layernorm_backward)(const NVTETensor, const NVTETensor, const NVTETensor, const NVTETensor,
                             const NVTETensor, NVTETensor, NVTETensor, NVTETensor, NVTETensor, int,
                             bool, cudaStream_t) = nullptr;

 private:
  template <typename Function>
  Function symbol(const char* name) {
    dlerror();
    void* address = dlsym(handle_, name);
    if (const char* error = dlerror(); error != nullptr) throw std::runtime_error(error);
    return reinterpret_cast<Function>(address);
  }

  void* handle_ = nullptr;
};

class DeviceAllocation {
 public:
  explicit DeviceAllocation(std::size_t bytes) : bytes_(bytes) {
    if (bytes != 0) check_cuda(cudaMalloc(&data_, bytes), "cudaMalloc");
  }
  ~DeviceAllocation() {
    if (data_ != nullptr) cudaFree(data_);
  }
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;
  void* data() const { return data_; }
  std::size_t bytes() const { return bytes_; }

 private:
  void* data_ = nullptr;
  std::size_t bytes_;
};

NVTEShape make_shape(const std::vector<std::size_t>& shape) {
  NVTEShape result{};
  constexpr std::size_t capacity = sizeof(result.data) / sizeof(result.data[0]);
  if (shape.size() > capacity) throw std::invalid_argument("tensor rank exceeds NVTEShape capacity");
  result.ndim = shape.size();
  for (std::size_t index = 0; index < shape.size(); ++index) result.data[index] = shape[index];
  return result;
}

class Tensor {
 public:
  explicit Tensor(TeApi* api) : api_(api), tensor_(api->create_tensor(NVTE_DELAYED_TENSOR_SCALING)) {}
  Tensor(TeApi* api, void* data, NVTEDType dtype, const std::vector<std::size_t>& shape) : Tensor(api) {
    NVTEBasicTensor basic{.data_ptr = data, .dtype = dtype, .shape = make_shape(shape)};
    api_->set_tensor_param(tensor_, kNVTERowwiseData, &basic, sizeof(basic));
  }
  ~Tensor() { api_->destroy_tensor(tensor_); }
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  NVTETensor get() const { return tensor_; }

 private:
  TeApi* api_;
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

struct Workspace {
  std::vector<std::size_t> shape;
  std::unique_ptr<DeviceAllocation> allocation;
  std::unique_ptr<Tensor> tensor;
};

Workspace materialize_workspace(TeApi* api, const Tensor& query) {
  const auto shape_value = api->tensor_shape(query.get());
  std::vector<std::size_t> shape(shape_value.data, shape_value.data + shape_value.ndim);
  if (api->tensor_type(query.get()) != kNVTEByte) throw std::runtime_error("workspace dtype is not kNVTEByte");
  auto allocation = std::make_unique<DeviceAllocation>(api->tensor_size_bytes(query.get()));
  auto tensor = std::make_unique<Tensor>(api, allocation->data(), kNVTEByte, shape);
  return {.shape = std::move(shape), .allocation = std::move(allocation), .tensor = std::move(tensor)};
}

std::vector<std::uint8_t> read_exact(const std::filesystem::path& path, std::size_t bytes) {
  if (std::filesystem::file_size(path) != bytes) throw std::runtime_error(path.string() + " has wrong size");
  std::vector<std::uint8_t> result(bytes);
  std::ifstream stream(path, std::ios::binary);
  stream.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(bytes));
  if (!stream) throw std::runtime_error("failed to read " + path.string());
  return result;
}

void copy_to_device(const std::vector<std::uint8_t>& source, DeviceAllocation* destination, cudaStream_t stream) {
  check_cuda(cudaMemcpyAsync(destination->data(), source.data(), source.size(), cudaMemcpyHostToDevice, stream),
             "cudaMemcpyAsync host to device");
}

void write_from_device(const std::filesystem::path& path, const DeviceAllocation& source, cudaStream_t stream) {
  std::vector<std::uint8_t> result(source.bytes());
  check_cuda(cudaMemcpyAsync(result.data(), source.data(), result.size(), cudaMemcpyDeviceToHost, stream),
             "cudaMemcpyAsync device to host");
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize output");
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(result.data()), static_cast<std::streamsize>(result.size()));
  if (!output) throw std::runtime_error("failed to write " + path.string());
}

struct Tensors {
  DeviceAllocation x_data, gamma_data, beta_data, dy_data, y_data, dx_data, dgamma_data, dbeta_data;
  DeviceAllocation mu_data, rsigma_data, zero_beta_data, throwaway_data;
  Tensor x, gamma, beta, dy, y, dx, dgamma, dbeta, mu, rsigma, zero_beta, throwaway;

  Tensors(TeApi* api, std::size_t rows, std::size_t columns)
      : x_data(rows * columns * 2), gamma_data(columns * 2), beta_data(columns * 2),
        dy_data(rows * columns * 2), y_data(rows * columns * 2), dx_data(rows * columns * 2),
        dgamma_data(columns * 2), dbeta_data(columns * 2), mu_data(rows * 4), rsigma_data(rows * 4),
        zero_beta_data(columns * 2), throwaway_data(rows * columns * 2),
        x(api, x_data.data(), kNVTEBFloat16, {rows, columns}),
        gamma(api, gamma_data.data(), kNVTEBFloat16, {columns}),
        beta(api, beta_data.data(), kNVTEBFloat16, {columns}),
        dy(api, dy_data.data(), kNVTEBFloat16, {rows, columns}),
        y(api, y_data.data(), kNVTEBFloat16, {rows, columns}),
        dx(api, dx_data.data(), kNVTEBFloat16, {rows, columns}),
        dgamma(api, dgamma_data.data(), kNVTEBFloat16, {columns}),
        dbeta(api, dbeta_data.data(), kNVTEBFloat16, {columns}),
        mu(api, mu_data.data(), kNVTEFloat32, {rows}), rsigma(api, rsigma_data.data(), kNVTEFloat32, {rows}),
        zero_beta(api, zero_beta_data.data(), kNVTEBFloat16, {columns}),
        throwaway(api, throwaway_data.data(), kNVTEBFloat16, {rows, columns}) {}
};

Workspace query_forward(TeApi* api, const Record& record, Tensors* tensors, int sms, cudaStream_t stream) {
  Tensor empty(api);
  NVTETensor beta = record.boundary == "backward_recompute" ? tensors->zero_beta.get() : tensors->beta.get();
  NVTETensor output = record.boundary == "backward_recompute" ? tensors->throwaway.get() : tensors->y.get();
  api->layernorm_forward(tensors->x.get(), tensors->gamma.get(), beta, kEpsilon, output, tensors->mu.get(),
                         tensors->rsigma.get(), empty.get(), sms, false, stream);
  return materialize_workspace(api, empty);
}

Workspace query_backward(TeApi* api, Tensors* tensors, int sms, cudaStream_t stream) {
  Tensor empty(api);
  api->layernorm_backward(tensors->dy.get(), tensors->x.get(), tensors->mu.get(), tensors->rsigma.get(),
                          tensors->gamma.get(), tensors->dx.get(), tensors->dgamma.get(), tensors->dbeta.get(),
                          empty.get(), sms, false, stream);
  return materialize_workspace(api, empty);
}

void invoke(const Record& record, TeApi* api, Tensors* tensors, Workspace* forward, Workspace* backward, int sms,
            cudaStream_t stream) {
  NVTETensor beta = record.boundary == "backward_recompute" ? tensors->zero_beta.get() : tensors->beta.get();
  NVTETensor output = record.boundary == "backward_recompute" ? tensors->throwaway.get() : tensors->y.get();
  api->layernorm_forward(tensors->x.get(), tensors->gamma.get(), beta, kEpsilon, output, tensors->mu.get(),
                         tensors->rsigma.get(), forward->tensor->get(), sms, false, stream);
  if (record.boundary != "forward") {
    api->layernorm_backward(tensors->dy.get(), tensors->x.get(), tensors->mu.get(), tensors->rsigma.get(),
                            tensors->gamma.get(), tensors->dx.get(), tensors->dgamma.get(), tensors->dbeta.get(),
                            backward->tensor->get(), sms, false, stream);
  }
}

void write_shape(std::ostream& stream, const std::vector<std::size_t>& shape) {
  stream << '[';
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (index != 0) stream << ',';
    stream << shape[index];
  }
  stream << ']';
}

int run(const Config& config) {
  check_cuda(cudaSetDevice(config.device), "cudaSetDevice");
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, config.device), "cudaGetDeviceProperties");
  if (properties.multiProcessorCount <= 0) throw std::runtime_error("physical SM count must be positive");

  TeApi api(config.te_library);
  api.enable_forward(config.record.forward_backend == "cudnn");
  api.enable_backward(config.record.backward_backend == "cudnn");
  Stream stream;
  Tensors tensors(&api, config.record.rows, config.record.columns);
  copy_to_device(read_exact(config.inputs / "x.bf16", tensors.x_data.bytes()), &tensors.x_data, stream.get());
  copy_to_device(read_exact(config.inputs / "gamma.bf16", tensors.gamma_data.bytes()), &tensors.gamma_data,
                 stream.get());
  if (config.record.boundary != "backward_recompute") {
    copy_to_device(read_exact(config.inputs / "beta.bf16", tensors.beta_data.bytes()), &tensors.beta_data,
                   stream.get());
  }
  if (config.record.boundary != "forward") {
    copy_to_device(read_exact(config.inputs / "dy.bf16", tensors.dy_data.bytes()), &tensors.dy_data, stream.get());
  }
  check_cuda(cudaMemsetAsync(tensors.zero_beta_data.data(), 0, tensors.zero_beta_data.bytes(), stream.get()),
             "cudaMemsetAsync zero_beta");
  check_cuda(cudaStreamSynchronize(stream.get()), "cudaStreamSynchronize inputs");

  Workspace forward = query_forward(&api, config.record, &tensors, properties.multiProcessorCount, stream.get());
  std::unique_ptr<Workspace> backward;
  if (config.record.boundary != "forward") {
    backward = std::make_unique<Workspace>(
        query_backward(&api, &tensors, properties.multiProcessorCount, stream.get()));
  }
  invoke(config.record, &api, &tensors, &forward, backward.get(), properties.multiProcessorCount, stream.get());
  check_cuda(cudaStreamSynchronize(stream.get()), "cudaStreamSynchronize boundary");

  if (config.record.boundary != "backward_recompute") {
    write_from_device(config.outputs / "y.bf16", tensors.y_data, stream.get());
  }
  if (config.record.boundary != "forward") {
    write_from_device(config.outputs / "dx.bf16", tensors.dx_data, stream.get());
    write_from_device(config.outputs / "dgamma.bf16", tensors.dgamma_data, stream.get());
    write_from_device(config.outputs / "dbeta.bf16", tensors.dbeta_data, stream.get());
  }

  std::ofstream metadata(config.outputs / "metadata.json");
  if (!metadata) throw std::runtime_error("failed to open metadata output");
  metadata << "{\"schema_version\":1,\"status\":\"unsealed_local_adapter_report\","
              "\"authority_sealed\":false,\"coordinate\":{\"ordinal\":"
           << config.record.ordinal
           << ",\"shape\":[" << config.record.rows << ',' << config.record.columns << "],\"boundary\":\""
           << config.record.boundary << "\",\"forward_backend\":\"" << config.record.forward_backend
           << "\",\"backward_backend\":";
  if (config.record.backward_backend == "none") metadata << "null";
  else metadata << '\"' << config.record.backward_backend << '\"';
  metadata << "},\"epsilon\":1e-05,\"scaling_mode\":\"NVTE_DELAYED_TENSOR_SCALING\","
              "\"layout\":\"row_major_contiguous_c_order\",\"physical_sm_count\":"
           << properties.multiProcessorCount << ",\"workspace_queries\":{\"forward\":{\"shape\":";
  write_shape(metadata, forward.shape);
  metadata << ",\"dtype\":\"kNVTEByte\"},\"backward\":";
  if (backward == nullptr) metadata << "null";
  else {
    metadata << "{\"shape\":";
    write_shape(metadata, backward->shape);
    metadata << ",\"dtype\":\"kNVTEByte\"}";
  }
  metadata << "},\"timing_collected\":false}\n";
  if (!metadata) throw std::runtime_error("failed to write metadata output");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(config_from_args(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "abi11_layernorm_te_oracle_adapter: " << error.what() << '\n';
    return 2;
  }
}

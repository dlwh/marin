// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include "mlir/IR/BuiltinTypes.h"
#include "shuttle/Runtime/GpuExecutable.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  if (argc != 2) {
    llvm::errs() << "usage: shuttle-gpu-inspect transport.bin\n";
    return 2;
  }
  auto file = llvm::MemoryBuffer::getFile(argv[1]);
  if (!file) {
    llvm::errs() << file.getError().message() << '\n';
    return 1;
  }
  auto bytes = (*file)->getBuffer();
  auto executable = mlir::shuttle::GpuExecutable::Load(
      llvm::ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(bytes.data()),
                              bytes.size()));
  if (!executable.ok()) {
    llvm::errs() << executable.status().ToString() << '\n';
    return 1;
  }
  llvm::json::Array images;
  for (const auto &image : (*executable)->codeImages())
    images.push_back(llvm::json::Object{
        {"size", static_cast<int64_t>(image.size())},
        {"digest", mlir::shuttle::gpuExecutableBundleDigest(image)}});
  llvm::json::Array launches;
  for (const auto &launch : (*executable)->launches())
    launches.push_back(llvm::json::Object{
        {"ordinal", launch.taskOrdinal},
        {"image_index", static_cast<int64_t>(launch.imageIndex)},
        {"symbol", launch.kernelName},
        {"image_digest", launch.codeDigest}});
  llvm::json::Array slots;
  for (const auto &slot : (*executable)->slots()) {
    llvm::json::Array shape;
    for (int64_t extent : mlir::cast<mlir::RankedTensorType>(slot.tensorType).getShape())
      shape.push_back(extent);
    slots.push_back(llvm::json::Object{{"ordinal", slot.ordinal},
                                      {"shape", std::move(shape)},
                                      {"required_bytes", slot.requiredBytes}});
  }
  llvm::outs() << llvm::json::Value(llvm::json::Object{
      {"images", std::move(images)}, {"launches", std::move(launches)},
      {"slots", std::move(slots)},
      {"total_code_bytes", static_cast<int64_t>((*executable)->totalCodeBytes())}})
               << '\n';
  return 0;
}

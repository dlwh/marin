// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "shuttle/IR/ShuttleDialect.h"
#include "shuttle/IR/ShuttleOps.h"
#include "shuttle/Runtime/GpuExecutable.h"
#include "shuttle/Transforms/Passes.h"
#include "stablehlo/dialect/Register.h"
#include "tools/cpp/runfiles/runfiles.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

namespace {

constexpr llvm::StringLiteral kAutomaticTuning =
    R"json({"cluster_shape":[],"materialization":"automatic","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]})json";
constexpr llvm::StringLiteral kFusionTuning =
    R"json({"cluster_shape":[],"materialization":"prefer_fusion","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]})json";
constexpr llvm::StringLiteral kMaterializationTuning =
    R"json({"cluster_shape":[],"materialization":"prefer_materialization","maximum_candidates":1,"pipeline_stages":1,"tile_sizes":[]})json";

struct FixtureExpectation {
  const char *filename;
  int64_t maps;
  int64_t folds;
  int64_t materializedTasks;
  int64_t materializedBuffers;
  int64_t fusedTasks;
  int64_t fusedBuffers;
};

// Counts are consequences of the anonymous algebra SSA graph. They are test
// oracles and are never compiler selectors.
constexpr std::array<FixtureExpectation, 12> kFixtures = {{
    {"jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-forward.mlir", 18, 1,
     19, 21, 11, 13},
    {"jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-backward.mlir", 43,
     5, 48, 51, 30, 33},
    {"jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-composed.mlir", 46,
     5, 51, 54, 33, 36},
    {"jax-0.10.1-bf16-projected_parameter_broadcast-"
     "3x2x5x11x263-forward.mlir",
     18, 1, 19, 21, 11, 13},
    {"jax-0.10.1-bf16-projected_parameter_broadcast-"
     "3x2x5x11x263-backward.mlir",
     43, 5, 48, 51, 30, 33},
    {"jax-0.10.1-bf16-projected_parameter_broadcast-"
     "3x2x5x11x263-composed.mlir",
     46, 5, 51, 54, 33, 36},
    {"jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-forward.mlir", 29, 2, 31, 34,
     18, 21},
    {"jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-backward.mlir", 63, 10, 73,
     76, 49, 52},
    {"jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-composed.mlir", 70, 10, 80,
     84, 54, 58},
    {"jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-forward.mlir", 29, 2, 31, 34,
     18, 21},
    {"jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-backward.mlir", 63, 10, 73,
     76, 49, 52},
    {"jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-composed.mlir", 70, 10, 80,
     84, 54, 58},
}};

struct BuiltModule {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

struct PlanSummary {
  int64_t maps = 0;
  int64_t folds = 0;
  int64_t contracts = 0;
  int64_t tasks = 0;
  int64_t buffers = 0;
  int64_t mapTasks = 0;
  int64_t foldTasks = 0;
  int64_t contractTasks = 0;
  int64_t eligibleFoldMapEdges = 0;
  std::vector<int64_t> fusedMapGroupSizes;
  std::vector<int64_t> fusedFoldMapGroupSizes;
  bool hasFoldRegion = false;
  bool hasContractRegion = false;
  bool exactSourceCoverage = false;
  std::string fingerprint;
};

int64_t staticElementCount(mlir::RankedTensorType type) {
  int64_t count = 1;
  for (int64_t extent : type.getShape())
    count *= extent;
  return count;
}

bool isOneToOneFoldMapEdge(mlir::shuttle::FoldOp fold, mlir::shuttle::MapOp map,
                           unsigned operandNumber) {
  if (fold.getNumResults() != 1 || !fold.getResult(0).hasOneUse() ||
      fold.getInputs().size() != 1 || fold.getInitializers().size() != 1 ||
      !fold.getOrderFree() || map.getNumResults() != 1 ||
      map.getIndexingMaps().size() != map.getInputs().size() + 1)
    return false;
  auto foldInput = mlir::dyn_cast<mlir::RankedTensorType>(
      fold.getInputs().front().getType());
  auto initializer = mlir::dyn_cast<mlir::RankedTensorType>(
      fold.getInitializers().front().getType());
  auto folded =
      mlir::dyn_cast<mlir::RankedTensorType>(fold.getResult(0).getType());
  auto output =
      mlir::dyn_cast<mlir::RankedTensorType>(map.getResult(0).getType());
  auto edge =
      mlir::dyn_cast<mlir::AffineMapAttr>(map.getIndexingMaps()[operandNumber]);
  auto resultMap = mlir::dyn_cast<mlir::AffineMapAttr>(
      map.getIndexingMaps().getValue().back());
  if (!foldInput || !initializer || !folded || !output ||
      !foldInput.hasStaticShape() || !foldInput.getElementType().isF32() ||
      initializer.getRank() != 0 || !initializer.getElementType().isF32() ||
      !folded.hasStaticShape() || !output.hasStaticShape() ||
      !folded.getElementType().isF32() || !output.getElementType().isF32() ||
      !edge || !resultMap || !resultMap.getValue().isIdentity() ||
      staticElementCount(folded) != staticElementCount(output) ||
      edge.getValue().getNumDims() != output.getRank() ||
      edge.getValue().getNumSymbols() != 0 ||
      edge.getValue().getNumResults() != folded.getRank())
    return false;
  for (mlir::Type type : map.getOperandTypes()) {
    auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type);
    if (!tensor || !tensor.hasStaticShape() || !tensor.getElementType().isF32())
      return false;
  }
  llvm::SmallDenseSet<unsigned> mapped;
  std::optional<unsigned> previousDimension;
  for (auto [axis, expression] :
       llvm::enumerate(edge.getValue().getResults())) {
    auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
    if (!dimension || !mapped.insert(dimension.getPosition()).second ||
        (previousDimension && dimension.getPosition() <= *previousDimension) ||
        folded.getDimSize(axis) != output.getDimSize(dimension.getPosition()))
      return false;
    previousDimension = dimension.getPosition();
  }
  return llvm::all_of(llvm::enumerate(output.getShape()), [&](auto item) {
    return mapped.contains(item.index()) || item.value() == 1;
  });
}

struct FusionAudit {
  int64_t groups = 0;
  bool sawComposedProjectedInput = false;
};

bool sourcePrecedes(mlir::shuttle::SourceRefAttr left,
                    mlir::shuttle::SourceRefAttr right) {
  return std::tuple(left.getFunctionOrdinal(), left.getBlockOrdinal(),
                    left.getOperationOrdinal(), left.getResultOrdinal()) <
         std::tuple(right.getFunctionOrdinal(), right.getBlockOrdinal(),
                    right.getOperationOrdinal(), right.getResultOrdinal());
}

std::string printType(mlir::Type type) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  type.print(stream);
  return result;
}

struct PtxStore {
  std::string pointer;
  std::string value;
  bool bf16;
};

// This parser is a pre-hardware structural oracle. It binds bounded code shape,
// reaching definitions, and shared reduction staging, but does not model PTX
// control-flow execution or establish numerical equivalence. The hardware-gated
// forward/VJP runner test is the authoritative semantic oracle.
struct PtxDataflow {
  struct Definition {
    std::string opcode;
    std::vector<std::string> operands;
    std::string instruction;
    bool predicated = false;
  };
  std::map<std::string, Definition> definitions;
  std::map<std::string, std::string> currentValues;
  std::vector<PtxStore> stores;
  std::vector<std::string> sharedStores;
};

PtxDataflow ptxDataflow(llvm::StringRef ptx);
std::set<std::string> ptxDependencyClosure(const PtxDataflow &flow,
                                           llvm::StringRef root);
const PtxDataflow::Definition *ptxDefinition(const PtxDataflow &flow,
                                             llvm::StringRef value);
bool ptxDependsOn(const PtxDataflow &flow, llvm::StringRef value,
                  llvm::StringRef dependency);
std::string ptxSemanticFingerprint(const PtxDataflow &flow,
                                   llvm::StringRef value);

std::vector<std::string> ptxTokens(llvm::StringRef text) {
  std::vector<std::string> tokens;
  for (size_t index = 0; index < text.size();) {
    const bool registerToken = text[index] == '%';
    const bool parameterToken = text[index] == 'p' && index + 1 < text.size() &&
                                llvm::isDigit(text[index + 1]);
    const bool constantToken = text[index] == '0' && index + 2 < text.size() &&
                               text[index + 1] == 'f' &&
                               llvm::isHexDigit(text[index + 2]);
    if (!registerToken && !parameterToken && !constantToken) {
      ++index;
      continue;
    }
    size_t end = index + (constantToken ? 2 : 1);
    while (end < text.size() && (llvm::isAlnum(text[end]) || text[end] == '_'))
      ++end;
    tokens.push_back(text.slice(index, end).str());
    index = end;
  }
  return tokens;
}

std::string ptxInstructionSkeleton(llvm::StringRef instruction) {
  std::string result;
  size_t cursor = 0;
  for (llvm::StringRef token : ptxTokens(instruction)) {
    size_t position = instruction.find(token, cursor);
    if (position == llvm::StringRef::npos)
      return {};
    result += instruction.slice(cursor, position).str();
    result += "$";
    cursor = position + token.size();
  }
  result += instruction.drop_front(cursor).str();
  return result;
}

bool hasDisjointContiguousFoldLoop(llvm::StringRef ptx, int64_t outputExtent,
                                   int64_t reductionExtent,
                                   int64_t serialTiles) {
  llvm::SmallVector<llvm::StringRef> lines;
  ptx.split(lines, '\n');
  for (llvm::StringRef &line : lines)
    line = line.trim();
  auto defines = [](llvm::StringRef line, llvm::StringRef reg) {
    if (line.starts_with("@")) {
      size_t instruction = line.find(' ');
      if (instruction == llvm::StringRef::npos)
        return false;
      line = line.drop_front(instruction + 1);
    }
    if (line.empty() || line.starts_with("st.") || line.starts_with("bra ") ||
        line.starts_with("bar.") || line == "ret;" || line.ends_with(":"))
      return false;
    std::vector<std::string> tokens = ptxTokens(line);
    return !tokens.empty() && tokens.front() == reg;
  };
  for (size_t laneLine = 0; laneLine < lines.size(); ++laneLine) {
    std::vector<std::string> laneTokens = ptxTokens(lines[laneLine]);
    if (!lines[laneLine].starts_with("mov.u32 ") ||
        !lines[laneLine].ends_with(", %tid.x;") || laneTokens.size() != 2 ||
        laneTokens[1] != "%tid")
      continue;
    llvm::StringRef lane = laneTokens[0];
    if (laneLine + 4 >= lines.size())
      continue;
    std::vector<std::string> rowTokens = ptxTokens(lines[laneLine + 1]);
    if (!lines[laneLine + 1].starts_with("mov.u32 ") ||
        !lines[laneLine + 1].ends_with(", %ctaid.x;") ||
        rowTokens.size() != 2 || rowTokens[1] != "%ctaid")
      continue;
    llvm::StringRef row = rowTokens[0];
    if (row == lane)
      continue;
    std::vector<std::string> guardTokens = ptxTokens(lines[laneLine + 2]);
    if (guardTokens.size() != 2 || guardTokens[1] != row ||
        lines[laneLine + 2] !=
            (llvm::Twine("setp.ge.u32 ") + guardTokens[0] + ", " + row + ", " +
             llvm::Twine(outputExtent) + ";")
                .str())
      continue;
    llvm::StringRef outputGuard = guardTokens[0];
    const std::string guardPrefix =
        (llvm::Twine("@") + outputGuard + " bra ").str();
    if (!lines[laneLine + 3].starts_with(guardPrefix) ||
        !lines[laneLine + 3].ends_with(";"))
      continue;
    llvm::StringRef outputExit =
        lines[laneLine + 3].drop_front(guardPrefix.size()).drop_back();
    for (size_t startLine = laneLine + 4; startLine < lines.size();
         ++startLine) {
      if (lines[startLine].starts_with("bra ") ||
          lines[startLine].contains(" bra ") || lines[startLine] == "ret;" ||
          lines[startLine].contains(" ret;") || lines[startLine] == "exit;" ||
          lines[startLine].contains(" exit;") ||
          lines[startLine].ends_with(":"))
        break;
      if (defines(lines[startLine], lane))
        break;
      std::vector<std::string> startTokens = ptxTokens(lines[startLine]);
      const std::string serialSuffix =
          (llvm::Twine(", ") + llvm::Twine(serialTiles) + ";").str();
      if (!lines[startLine].starts_with("mul.lo.u32 ") ||
          !lines[startLine].ends_with(serialSuffix) ||
          startTokens.size() != 2 || startTokens[1] != lane)
        continue;
      llvm::StringRef start = startTokens[0];
      if (startLine + 5 >= lines.size())
        return false;
      std::vector<std::string> endTokens = ptxTokens(lines[startLine + 1]);
      if (!lines[startLine + 1].starts_with("add.u32 ") ||
          !lines[startLine + 1].ends_with(serialSuffix) ||
          endTokens.size() != 2 || endTokens[1] != start)
        return false;
      llvm::StringRef end = endTokens[0];
      if (end == start)
        return false;
      const std::string extentSuffix =
          (llvm::Twine(", ") + llvm::Twine(reductionExtent) + ";").str();
      std::vector<std::string> minTokens = ptxTokens(lines[startLine + 2]);
      if (!lines[startLine + 2].starts_with("min.u32 ") ||
          !lines[startLine + 2].ends_with(extentSuffix) ||
          minTokens.size() != 2 || minTokens[0] != end ||
          minTokens[1] != end)
        return false;
      size_t loopLine = startLine + 3;
      while (loopLine < lines.size() && !lines[loopLine].ends_with(":"))
        ++loopLine;
      if (loopLine + 2 >= lines.size())
        return false;
      std::string accumulator;
      for (size_t line = startLine + 3; line < loopLine; ++line) {
        if (defines(lines[line], start) || defines(lines[line], end) ||
            lines[line].starts_with("bra ") || lines[line].contains(" bra ") ||
            lines[line] == "ret;" || lines[line].contains(" ret;") ||
            lines[line] == "exit;" || lines[line].contains(" exit;"))
          return false;
        std::vector<std::string> tokens = ptxTokens(lines[line]);
        if (lines[line].starts_with("mov.f32 ") &&
            lines[line].ends_with(", 0f00000000;") && tokens.size() == 2)
          accumulator = tokens[0];
      }
      if (accumulator.empty())
        return false;
      PtxDataflow preheaderFlow = ptxDataflow(
          llvm::join(llvm::ArrayRef(lines).take_front(loopLine), "\n"));
      auto reachingAccumulator = preheaderFlow.currentValues.find(accumulator);
      if (reachingAccumulator == preheaderFlow.currentValues.end())
        return false;
      const PtxDataflow::Definition *accumulatorDefinition =
          ptxDefinition(preheaderFlow, reachingAccumulator->second);
      if (!accumulatorDefinition ||
          accumulatorDefinition->opcode != "mov.f32" ||
          accumulatorDefinition->operands !=
              std::vector<std::string>{"0f00000000"})
        return false;
      llvm::StringRef loop = lines[loopLine].drop_back();
      std::vector<std::string> predicateTokens =
          ptxTokens(lines[loopLine + 1]);
      if (!lines[loopLine + 1].starts_with("setp.ge.u32 ") ||
          predicateTokens.size() != 3 || predicateTokens[1] != start ||
          predicateTokens[2] != end)
        return false;
      llvm::StringRef predicate = predicateTokens[0];
      const std::string branchPrefix =
          (llvm::Twine("@") + predicate + " bra ").str();
      if (!lines[loopLine + 2].starts_with(branchPrefix) ||
          !lines[loopLine + 2].ends_with(";"))
        return false;
      llvm::StringRef exit = lines[loopLine + 2]
                                 .drop_front(branchPrefix.size())
                                 .drop_back();
      const std::string increment =
          (llvm::Twine("add.u32 ") + start + ", " + start + ", 1;").str();
      const std::string backedge = (llvm::Twine("bra ") + loop + ";").str();
      const std::string exitLabel = (llvm::Twine(exit) + ":").str();
      bool accumulatedIndexedLeaf = false;
      int64_t accumulatedLeafUpdates = 0;
      for (size_t line = loopLine + 3; line + 2 < lines.size(); ++line) {
        if (lines[line] == exitLabel || lines[line].starts_with("bra ") ||
            lines[line].contains(" bra ") || lines[line] == "ret;" ||
            lines[line].contains(" ret;") || lines[line] == "exit;" ||
            lines[line].contains(" exit;") || lines[line].ends_with(":") ||
            defines(lines[line], end) ||
            (defines(lines[line], start) && lines[line] != increment))
          return false;
        std::vector<std::string> tokens = ptxTokens(lines[line]);
        if (lines[line].starts_with("add.rn.f32 ") && tokens.size() == 3 &&
            (tokens[1] == accumulator || tokens[2] == accumulator)) {
          llvm::StringRef leaf =
              tokens[1] == accumulator ? tokens[2] : tokens[1];
          const std::string update =
              (llvm::Twine("mov.f32 ") + accumulator + ", " + tokens[0] + ";")
                  .str();
          if (lines[line + 1] != update)
            return false;
          PtxDataflow bodyFlow = ptxDataflow(llvm::join(
              llvm::ArrayRef(lines).slice(loopLine + 3, line - loopLine - 3),
              "\n"));
          auto reachingLeaf = bodyFlow.currentValues.find(leaf.str());
          if (reachingLeaf != bodyFlow.currentValues.end()) {
            std::set<std::string> closure =
                ptxDependencyClosure(bodyFlow, reachingLeaf->second);
            const bool unpredicatedLeaf =
                !closure.count(accumulator) &&
                llvm::all_of(closure, [&](const std::string &value) {
                  const PtxDataflow::Definition *definition =
                      ptxDefinition(bodyFlow, value);
                  return !definition || !definition->predicated;
                });
            if (!unpredicatedLeaf)
              return false;
            for (const auto &[loadValue, load] : bodyFlow.definitions) {
              if (!closure.count(loadValue) ||
                  !llvm::StringRef(load.opcode).starts_with("ld.global.") ||
                  load.predicated)
                continue;
              std::set<std::string> loadClosure =
                  ptxDependencyClosure(bodyFlow, loadValue);
              bool unpredicatedStartConversion = false;
              for (const std::string &value : loadClosure) {
                const PtxDataflow::Definition *definition =
                    ptxDefinition(bodyFlow, value);
                unpredicatedStartConversion |=
                    definition && definition->opcode == "cvt.u64.u32" &&
                    !definition->predicated &&
                    ptxDependencyClosure(bodyFlow, value).count(start.str());
              }
              accumulatedIndexedLeaf |= unpredicatedStartConversion;
            }
          }
          ++accumulatedLeafUpdates;
          ++line;
          continue;
        }
        if (defines(lines[line], accumulator))
          return false;
        if (lines[line] == increment)
          return accumulatedIndexedLeaf && accumulatedLeafUpdates == 1 &&
                 outputExit != exit && lines[line + 1] == backedge &&
                 lines[line + 2] == exitLabel &&
                 llvm::is_contained(llvm::ArrayRef(lines).drop_front(line + 3),
                                    (llvm::Twine(outputExit) + ":").str());
      }
      return false;
    }
  }
  return false;
}

bool hasDependentRowFoldPhaseDataflow(llvm::StringRef ptx,
                                      int64_t reductionExtent,
                                      int64_t serialTiles) {
  constexpr llvm::StringLiteral loadLabel = "$fold_region_phase_1_load:";
  constexpr llvm::StringLiteral loadedLabel = "$fold_region_phase_1_loaded:";
  size_t load = ptx.find(loadLabel);
  size_t loaded = ptx.find(loadedLabel);
  if (load == llvm::StringRef::npos || loaded == llvm::StringRef::npos ||
      load >= loaded)
    return false;
  const std::string preheaderPrefix =
      (llvm::Twine("  bar.sync 0;\n  mul.lo.u32 %r4, %r0, ") +
       llvm::Twine(serialTiles) + ";")
          .str();
  size_t preheader = ptx.take_front(load).rfind(preheaderPrefix);
  size_t sharedResultLoad =
      preheader == llvm::StringRef::npos
          ? llvm::StringRef::npos
          : ptx.take_front(preheader).rfind("  ld.shared.f32 ");
  size_t sharedStore = ptx.find("  st.shared.f32", loaded);
  size_t sharedStoreEnd = sharedStore == llvm::StringRef::npos
                              ? llvm::StringRef::npos
                              : ptx.find('\n', sharedStore);
  if (preheader == llvm::StringRef::npos ||
      sharedResultLoad == llvm::StringRef::npos ||
      sharedStoreEnd == llvm::StringRef::npos)
    return false;
  llvm::StringRef phase = ptx.slice(sharedResultLoad, sharedStoreEnd);
  const std::string end =
      (llvm::Twine("  add.u32 %r7, %r4, ") + llvm::Twine(serialTiles) +
       ";\n  min.u32 %r7, %r7, " + llvm::Twine(reductionExtent) + ";")
          .str();
  const std::string rowMajorIndex = (llvm::Twine("  cvt.u64.u32 %rd0, %r4;\n"
                                                 "  cvt.u64.u32 %rd1, %r1;\n"
                                                 "  mad.lo.u64 %rd0, %rd1, ") +
                                     llvm::Twine(reductionExtent) + ", %rd0;")
                                        .str();
  if (!phase.contains(end) || !phase.contains(rowMajorIndex) ||
      !phase.contains("  add.u32 %r4, %r4, 1;\n"
                      "  bra $fold_region_phase_1_load;"))
    return false;
  PtxDataflow flow = ptxDataflow(phase);
  if (flow.sharedStores.size() != 1)
    return false;
  auto rowMajorFound = flow.currentValues.find("%rd0");
  if (rowMajorFound == flow.currentValues.end())
    return false;
  llvm::StringRef rowMajorValue = rowMajorFound->second;
  const PtxDataflow::Definition *rowMajor = ptxDefinition(flow, rowMajorValue);
  const std::string expectedRowMajor =
      (llvm::Twine("mad.lo.u64 %rd0, %rd1, ") + llvm::Twine(reductionExtent) +
       ", %rd0;")
          .str();
  if (!rowMajor || rowMajor->opcode != "mad.lo.u64" ||
      rowMajor->instruction != expectedRowMajor ||
      rowMajor->operands.size() != 2 ||
      !ptxDependsOn(flow, rowMajor->operands[0], "%r1") ||
      !ptxDependsOn(flow, rowMajor->operands[1], "%r0"))
    return false;

  llvm::StringRef storedValue = flow.sharedStores.front();
  const PtxDataflow::Definition *stored = ptxDefinition(flow, storedValue);
  if (!stored || stored->opcode != "mov.f32" || stored->operands.size() != 1)
    return false;
  const PtxDataflow::Definition *combined =
      ptxDefinition(flow, stored->operands.front());
  if (!combined || combined->opcode != "add.rn.f32" ||
      combined->operands.size() != 2)
    return false;
  int accumulatorOperand = -1;
  for (int operand : {0, 1}) {
    const PtxDataflow::Definition *candidate =
        ptxDefinition(flow, combined->operands[operand]);
    if (candidate && candidate->opcode == "mov.f32" &&
        candidate->operands == std::vector<std::string>{"0f00000000"})
      accumulatorOperand = operand;
  }
  if (accumulatorOperand < 0)
    return false;
  const PtxDataflow::Definition *leaf =
      ptxDefinition(flow, combined->operands[1 - accumulatorOperand]);
  if (!leaf || leaf->opcode != "mul.rn.f32" || leaf->operands.size() != 2)
    return false;
  auto isCenteredInput = [&](llvm::StringRef value) {
    const PtxDataflow::Definition *centered = ptxDefinition(flow, value);
    if (!centered || centered->opcode != "sub.rn.f32" ||
        centered->operands.size() != 2)
      return false;
    std::set<std::string> input =
        ptxDependencyClosure(flow, centered->operands[0]);
    std::set<std::string> mean =
        ptxDependencyClosure(flow, centered->operands[1]);
    return input.count("$ld.global.u16") && input.count("p0") &&
           input.count(rowMajorValue.str()) && mean.count("$shared");
  };
  if (!isCenteredInput(leaf->operands[0]) ||
      !isCenteredInput(leaf->operands[1]))
    return false;
  if (ptxSemanticFingerprint(flow, leaf->operands[0]) !=
      ptxSemanticFingerprint(flow, leaf->operands[1]))
    return false;

  llvm::StringRef accumulator = storedValue.take_front(storedValue.find('@'));
  llvm::StringRef combinedValue = stored->operands.front();
  llvm::StringRef combinedRegister =
      combinedValue.take_front(combinedValue.find('@'));
  int64_t accumulatorDefinitions =
      llvm::count_if(flow.definitions, [&](const auto &definition) {
        return llvm::StringRef(definition.first)
            .starts_with((llvm::Twine(accumulator) + "@").str());
      });
  const std::string recurrence =
      (llvm::Twine("  mov.f32 ") + accumulator + ", " + combinedRegister +
       ";\n  add.u32 %r4, %r4, 1;")
          .str();
  return accumulatorDefinitions == 2 && phase.contains(recurrence);
}

PtxDataflow ptxDataflow(llvm::StringRef ptx) {
  PtxDataflow result;
  std::map<std::string, std::string> currentRegister;
  std::map<std::string, int64_t> generations;
  std::vector<std::string> sharedValues;
  auto current = [&](const std::string &token) {
    if (token.empty() || token.front() != '%')
      return token;
    auto found = currentRegister.find(token);
    return found == currentRegister.end() ? token : found->second;
  };
  while (!ptx.empty()) {
    auto [line, remainder] = ptx.split('\n');
    ptx = remainder;
    line = line.trim();
    if (line.empty() || line.starts_with("//") || line.ends_with(":"))
      continue;
    size_t opcodeStart = line.find_first_not_of(" @%p0123456789");
    if (opcodeStart == llvm::StringRef::npos)
      continue;
    llvm::StringRef instruction = line.drop_front(opcodeStart);
    auto [opcode, operands] = instruction.split(' ');
    std::vector<std::string> tokens = ptxTokens(operands);
    if (opcode.starts_with("st.global.")) {
      if (tokens.size() >= 2)
        result.stores.push_back({current(tokens[0]), current(tokens[1]),
                                 opcode == "st.global.u16"});
      continue;
    }
    if (opcode.starts_with("st.shared.")) {
      if (tokens.size() >= 2) {
        sharedValues.push_back(current(tokens[1]));
        result.sharedStores.push_back(current(tokens[1]));
      }
      continue;
    }
    if (tokens.empty() || tokens.front().front() != '%')
      continue;
    std::vector<std::string> dependencies;
    for (const std::string &token : llvm::drop_begin(tokens))
      dependencies.push_back(current(token));
    if (opcode.starts_with("ld.global."))
      dependencies.push_back((llvm::Twine("$") + opcode).str());
    if (opcode.starts_with("ld.shared.")) {
      dependencies.push_back("$shared");
      llvm::append_range(dependencies, sharedValues);
    }
    const std::string destination = (llvm::Twine(tokens.front()) + "@" +
                                     llvm::Twine(generations[tokens.front()]++))
                                        .str();
    result.definitions[destination] = {opcode.str(), std::move(dependencies),
                                       instruction.str(),
                                       line.starts_with("@")};
    currentRegister[tokens.front()] = destination;
  }
  result.currentValues = std::move(currentRegister);
  return result;
}

std::set<std::string> ptxDependencyClosure(const PtxDataflow &flow,
                                           llvm::StringRef root) {
  std::set<std::string> closure;
  std::vector<std::string> worklist{root.str()};
  while (!worklist.empty()) {
    std::string value = std::move(worklist.back());
    worklist.pop_back();
    if (!closure.insert(value).second)
      continue;
    auto found = flow.definitions.find(value);
    if (found != flow.definitions.end()) {
      closure.insert((llvm::Twine("$op:") + found->second.opcode).str());
      worklist.insert(worklist.end(), found->second.operands.begin(),
                      found->second.operands.end());
    }
  }
  return closure;
}

const PtxDataflow::Definition *ptxDefinition(const PtxDataflow &flow,
                                             llvm::StringRef value) {
  auto found = flow.definitions.find(value.str());
  return found == flow.definitions.end() ? nullptr : &found->second;
}

bool ptxDependsOn(const PtxDataflow &flow, llvm::StringRef value,
                  llvm::StringRef dependency) {
  auto closure = ptxDependencyClosure(flow, value);
  return closure.count(dependency.str()) != 0;
}

std::string ptxSemanticFingerprint(const PtxDataflow &flow,
                                   llvm::StringRef value) {
  std::set<std::string> active;
  auto fingerprint = [&](auto &self, llvm::StringRef current) -> std::string {
    const PtxDataflow::Definition *definition = ptxDefinition(flow, current);
    if (!definition)
      return current.str();
    if (!active.insert(current.str()).second)
      return "$cycle";
    std::string result = definition->predicated ? "@" : "";
    result += ptxInstructionSkeleton(definition->instruction);
    result += "(";
    for (llvm::StringRef operand : definition->operands) {
      result += self(self, operand);
      result += ",";
    }
    active.erase(current.str());
    result += ")";
    return result;
  };
  return fingerprint(fingerprint, value);
}

bool containsRmsNormReciprocalSubgraph(const PtxDataflow &flow,
                                       llvm::StringRef value,
                                       llvm::StringRef extent,
                                       llvm::StringRef meanDependency) {
  std::set<std::string> closure = ptxDependencyClosure(flow, value);
  if (closure.count("$op:neg.f32"))
    return false;
  for (const std::string &candidate : closure) {
    const auto *reciprocal = ptxDefinition(flow, candidate);
    if (!reciprocal || reciprocal->opcode != "div.rn.f32" ||
        reciprocal->operands.size() < 2 ||
        !ptxDependsOn(flow, reciprocal->operands[0], "0f3F800000"))
      continue;
    const auto *root = ptxDefinition(flow, reciprocal->operands[1]);
    if (!root || root->opcode != "sqrt.rn.f32" || root->operands.empty())
      continue;
    const auto *epsilonAdd = ptxDefinition(flow, root->operands[0]);
    if (!epsilonAdd || epsilonAdd->opcode != "add.rn.f32" ||
        epsilonAdd->operands.size() < 2)
      continue;
    for (int meanOperand : {0, 1}) {
      const int epsilonOperand = meanOperand == 0 ? 1 : 0;
      if (!ptxDependsOn(flow, epsilonAdd->operands[epsilonOperand],
                        "0f358637BD"))
        continue;
      const auto *mean = ptxDefinition(flow, epsilonAdd->operands[meanOperand]);
      if (!mean || mean->opcode != "div.rn.f32" || mean->operands.size() < 2)
        continue;
      if (ptxDependsOn(flow, mean->operands[0], meanDependency) &&
          ptxDependsOn(flow, mean->operands[1], extent))
        return true;
    }
  }
  return false;
}

bool containsStagedSharedAdd(const PtxDataflow &flow) {
  if (flow.sharedStores.empty())
    return false;
  for (llvm::StringRef stored : llvm::drop_begin(flow.sharedStores)) {
    const auto *add = ptxDefinition(flow, stored);
    if (!add || add->opcode != "add.rn.f32" || add->operands.size() < 2)
      return false;
    const auto *left = ptxDefinition(flow, add->operands[0]);
    const auto *right = ptxDefinition(flow, add->operands[1]);
    if (!left || !right ||
        !llvm::StringRef(left->opcode).starts_with("ld.shared.") ||
        !llvm::StringRef(right->opcode).starts_with("ld.shared."))
      return false;
  }
  return true;
}

std::vector<int64_t>
expectedFusionGroupSizes(const FixtureExpectation &fixture) {
  if (fixture.maps == 18 && fixture.folds == 1)
    return {6, 4};
  if (fixture.maps == 43 && fixture.folds == 5)
    return {5, 7, 3, 2, 6};
  if (fixture.maps == 46 && fixture.folds == 5)
    return {5, 7, 2, 2, 2, 6};
  if (fixture.maps == 29 && fixture.folds == 2)
    return {3, 2, 6, 6};
  if (fixture.maps == 63 && fixture.folds == 10)
    return {3, 2, 5, 7, 3, 2, 6, 3, 2};
  if (fixture.maps == 70 && fixture.folds == 10)
    return {3, 2, 5, 7, 2, 4, 2, 6, 3, 2};
  return {};
}

std::string options(mlir::shuttle::NumericalPolicy policy,
                    llvm::StringRef tuning) {
  const llvm::StringRef numerics =
      policy == mlir::shuttle::NumericalPolicy::Fast ? "fast"
                                                     : "source_ordered";
  return (llvm::Twine("{\"execution_mode\":\"gpu_executable_bundle\","
                      "\"numerics\":\"") +
          numerics +
          "\",\"pipeline_abi_version\":22,\"schema_version\":1,"
          "\"tuning\":" +
          tuning + "}")
      .str();
}

std::optional<std::string> fixtureSource(llvm::StringRef filename) {
  std::string error;
  auto runfiles = bazel::tools::cpp::runfiles::Runfiles::CreateForTest(&error);
  if (!runfiles)
    return std::nullopt;
  std::ifstream input(runfiles->Rlocation(
      (llvm::Twine("shuttle_mlir/test/Inputs/") + filename).str()));
  std::ostringstream source;
  source << input.rdbuf();
  if (!input)
    return std::nullopt;
  return source.str();
}

void renameSymbols(mlir::ModuleOp module) {
  module.setSymName("anonymous_module");
  int64_t ordinal = 0;
  for (mlir::func::FuncOp function : module.getOps<mlir::func::FuncOp>())
    function.setSymName(
        (llvm::Twine("anonymous_function_") + llvm::Twine(ordinal++)).str());
}

std::unique_ptr<BuiltModule> build(llvm::StringRef source,
                                   mlir::shuttle::NumericalPolicy policy,
                                   llvm::StringRef tuning, bool renamed,
                                   bool executable) {
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect, mlir::shuttle::ShuttleDialect>();
  auto result = std::make_unique<BuiltModule>();
  result->context = std::make_unique<mlir::MLIRContext>(registry);
  result->module =
      mlir::parseSourceString<mlir::ModuleOp>(source, result->context.get());
  if (!result->module)
    return {};
  if (renamed)
    renameSymbols(*result->module);

  mlir::PassManager manager(result->context.get());
  manager.addPass(mlir::shuttle::createAnnotateSourcePass());
  manager.addPass(mlir::shuttle::createFormStructuralRegionsPass(
      policy, options(policy, tuning), tuning.str()));
  manager.addPass(mlir::shuttle::createConvertStablehloToAlgebraPass());
  manager.addPass(mlir::shuttle::createVerifySourceCoveragePass());
  manager.addPass(mlir::shuttle::createVerifySemanticErasurePass());
  manager.addPass(mlir::shuttle::createPlanRowFoldMaterializationPass());
  manager.addPass(mlir::shuttle::createVerifyMaterializationPlanPass());
  if (executable) {
    manager.addPass(mlir::shuttle::createPlanSimt32RowFoldSchedulePass());
    manager.addPass(mlir::shuttle::createVerifySimt32RowFoldSchedulePass());
    manager.addPass(mlir::shuttle::createBuildGpuExecutableBundlePass());
    manager.addPass(mlir::shuttle::createVerifyGpuExecutableBundlePass());
  }
  if (mlir::failed(manager.run(*result->module)))
    return {};
  return result;
}

std::unique_ptr<BuiltModule>
buildAlgebra(llvm::StringRef source, mlir::shuttle::NumericalPolicy policy,
             bool executable = true, bool renamed = false) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect, mlir::shuttle::ShuttleDialect>();
  auto result = std::make_unique<BuiltModule>();
  result->context = std::make_unique<mlir::MLIRContext>(registry);
  result->module =
      mlir::parseSourceString<mlir::ModuleOp>(source, result->context.get());
  if (!result->module)
    return {};
  if (renamed)
    renameSymbols(*result->module);
  result->module->walk(
      [&](mlir::shuttle::RegionOp region) { region.setPolicy(policy); });
  mlir::PassManager manager(result->context.get());
  manager.addPass(mlir::shuttle::createPlanRowFoldMaterializationPass());
  manager.addPass(mlir::shuttle::createVerifyMaterializationPlanPass());
  if (executable) {
    manager.addPass(mlir::shuttle::createPlanSimt32RowFoldSchedulePass());
    manager.addPass(mlir::shuttle::createVerifySimt32RowFoldSchedulePass());
    manager.addPass(mlir::shuttle::createBuildGpuExecutableBundlePass());
    manager.addPass(mlir::shuttle::createVerifyGpuExecutableBundlePass());
  }
  if (mlir::failed(manager.run(*result->module)))
    return {};
  return result;
}

std::optional<PlanSummary> summary(mlir::ModuleOp module) {
  PlanSummary result;
  std::vector<std::pair<mlir::shuttle::SourceRefAttr, int64_t>> fusedGroups;
  llvm::SmallDenseSet<mlir::Attribute> algebraSources;
  module.walk([&](mlir::shuttle::MapOp) { ++result.maps; });
  module.walk([&](mlir::shuttle::FoldOp) { ++result.folds; });
  module.walk([&](mlir::shuttle::ContractOp) { ++result.contracts; });
  module.walk([&](mlir::Operation *operation) {
    if (auto map = mlir::dyn_cast<mlir::shuttle::MapOp>(operation))
      algebraSources.insert(map.getSource());
    else if (auto fold = mlir::dyn_cast<mlir::shuttle::FoldOp>(operation))
      algebraSources.insert(fold.getSource());
    else if (auto contract =
                 mlir::dyn_cast<mlir::shuttle::ContractOp>(operation))
      algebraSources.insert(contract.getSource());
  });
  module.walk([&](mlir::shuttle::FoldOp fold) {
    if (fold.getNumResults() != 1 || !fold.getResult(0).hasOneUse())
      return;
    mlir::OpOperand &use = *fold.getResult(0).getUses().begin();
    auto map = mlir::dyn_cast<mlir::shuttle::MapOp>(use.getOwner());
    if (map && isOneToOneFoldMapEdge(fold, map, use.getOperandNumber()))
      ++result.eligibleFoldMapEdges;
  });
  llvm::SmallVector<mlir::shuttle::MaterializationPlanOp> plans(
      module.getOps<mlir::shuttle::MaterializationPlanOp>());
  if (plans.size() != 1)
    return std::nullopt;
  result.fingerprint = plans.front().getFingerprint().str();
  llvm::SmallDenseSet<mlir::Attribute> plannedSources;
  for (mlir::Operation &operation : plans.front().getBody().front()) {
    if (mlir::isa<mlir::shuttle::MaterializationBufferOp>(operation)) {
      ++result.buffers;
      continue;
    }
    auto task = mlir::dyn_cast<mlir::shuttle::MaterializationTaskOp>(operation);
    if (!task)
      continue;
    auto fusedSources = task->getAttrOfType<mlir::ArrayAttr>("fused_sources");
    auto foldEpilogueSources =
        task->getAttrOfType<mlir::ArrayAttr>("fold_epilogue_sources");
    auto foldRegionSources =
        task->getAttrOfType<mlir::ArrayAttr>("fold_region_sources");
    auto contractRegionSources =
        task->getAttrOfType<mlir::ArrayAttr>("contract_region_sources");
    if (static_cast<int>(static_cast<bool>(fusedSources)) +
            static_cast<int>(static_cast<bool>(foldEpilogueSources)) +
            static_cast<int>(static_cast<bool>(foldRegionSources)) +
            static_cast<int>(static_cast<bool>(contractRegionSources)) >
        1)
      return std::nullopt;
    if (contractRegionSources) {
      result.hasContractRegion = true;
      if (task.getKind() != mlir::shuttle::MaterializationTaskKind::Contract ||
          contractRegionSources.empty())
        return std::nullopt;
      for (mlir::Attribute source : contractRegionSources) {
        if (!mlir::isa<mlir::shuttle::SourceRefAttr>(source))
          return std::nullopt;
        plannedSources.insert(source);
      }
    } else if (foldRegionSources) {
      result.hasFoldRegion = true;
      if (task.getKind() != mlir::shuttle::MaterializationTaskKind::Fold ||
          foldRegionSources.empty())
        return std::nullopt;
      for (mlir::Attribute source : foldRegionSources) {
        if (!mlir::isa<mlir::shuttle::SourceRefAttr>(source))
          return std::nullopt;
        plannedSources.insert(source);
      }
    } else if (foldEpilogueSources) {
      if (task.getKind() != mlir::shuttle::MaterializationTaskKind::Fold ||
          foldEpilogueSources.empty() ||
          !plannedSources.insert(task.getSource()).second)
        return std::nullopt;
      auto firstSource = mlir::dyn_cast<mlir::shuttle::SourceRefAttr>(
          foldEpilogueSources.getValue().front());
      if (!firstSource)
        return std::nullopt;
      if (foldEpilogueSources.size() > 1)
        fusedGroups.emplace_back(firstSource, foldEpilogueSources.size());
      result.fusedFoldMapGroupSizes.push_back(foldEpilogueSources.size());
      for (mlir::Attribute source : foldEpilogueSources) {
        if (!mlir::isa<mlir::shuttle::SourceRefAttr>(source) ||
            !plannedSources.insert(source).second)
          return std::nullopt;
      }
    } else if (fusedSources) {
      if (fusedSources.size() < 2 ||
          fusedSources.getValue().back() != task.getSource())
        return std::nullopt;
      auto firstSource = mlir::dyn_cast<mlir::shuttle::SourceRefAttr>(
          fusedSources.getValue().front());
      if (!firstSource)
        return std::nullopt;
      fusedGroups.emplace_back(firstSource, fusedSources.size());
      for (mlir::Attribute source : fusedSources) {
        if (!mlir::isa<mlir::shuttle::SourceRefAttr>(source) ||
            !plannedSources.insert(source).second)
          return std::nullopt;
      }
    } else if (!plannedSources.insert(task.getSource()).second) {
      return std::nullopt;
    }
    ++result.tasks;
    switch (task.getKind()) {
    case mlir::shuttle::MaterializationTaskKind::Map:
      ++result.mapTasks;
      break;
    case mlir::shuttle::MaterializationTaskKind::Fold:
      ++result.foldTasks;
      break;
    case mlir::shuttle::MaterializationTaskKind::Contract:
      ++result.contractTasks;
      break;
    }
  }
  result.exactSourceCoverage =
      plannedSources.size() == algebraSources.size() &&
      llvm::all_of(plannedSources, [&](mlir::Attribute source) {
        return algebraSources.contains(source);
      });
  llvm::sort(fusedGroups, [](const auto &left, const auto &right) {
    return sourcePrecedes(left.first, right.first);
  });
  for (const auto &group : fusedGroups)
    result.fusedMapGroupSizes.push_back(group.second);
  return result;
}

std::optional<llvm::SmallVector<int64_t>>
mapDomainShape(mlir::shuttle::MapOp map) {
  if (map.getNumResults() != 1)
    return std::nullopt;
  auto result =
      mlir::dyn_cast<mlir::RankedTensorType>(map.getResult(0).getType());
  if (!result || !result.hasStaticShape())
    return std::nullopt;
  auto resultMap =
      mlir::cast<mlir::AffineMapAttr>(map.getIndexingMaps().getValue().back())
          .getValue();
  if (!resultMap.isIdentity())
    return std::nullopt;
  return llvm::SmallVector<int64_t>(result.getShape());
}

std::optional<FusionAudit> auditFusionGroups(mlir::ModuleOp module) {
  llvm::SmallDenseMap<mlir::Attribute, mlir::shuttle::MapOp> maps;
  module.walk([&](mlir::shuttle::MapOp map) {
    maps.try_emplace(map.getSource(), map);
  });
  llvm::SmallVector<mlir::shuttle::MaterializationPlanOp> plans(
      module.getOps<mlir::shuttle::MaterializationPlanOp>());
  if (plans.size() != 1)
    return std::nullopt;

  FusionAudit audit;
  for (mlir::shuttle::MaterializationTaskOp task :
       plans.front()
           .getBody()
           .front()
           .getOps<mlir::shuttle::MaterializationTaskOp>()) {
    auto sources = task->getAttrOfType<mlir::ArrayAttr>("fused_sources");
    if (!sources)
      continue;
    ++audit.groups;
    llvm::SmallDenseSet<mlir::Attribute> group;
    std::optional<mlir::shuttle::SourceRefAttr> previousSource;
    for (mlir::Attribute source : sources) {
      auto sourceRef = mlir::dyn_cast<mlir::shuttle::SourceRefAttr>(source);
      auto found = maps.find(source);
      if (!sourceRef || found == maps.end() || !group.insert(source).second ||
          (previousSource && !sourcePrecedes(*previousSource, sourceRef)))
        return std::nullopt;
      previousSource = sourceRef;
      mlir::shuttle::MapOp map = found->second;
      std::optional<llvm::SmallVector<int64_t>> domain = mapDomainShape(map);
      if (!domain)
        return std::nullopt;
      llvm::SmallVector<mlir::Type> boundaryTypes(map.getOperandTypes());
      llvm::append_range(boundaryTypes, map.getResultTypes());
      for (mlir::Type type : boundaryTypes) {
        auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type);
        if (!tensor || !tensor.hasStaticShape() ||
            !tensor.getElementType().isF32())
          return std::nullopt;
      }
    }
    for (mlir::Attribute source : sources.getValue().drop_back()) {
      mlir::shuttle::MapOp producer = maps.lookup(source);
      if (producer.getNumResults() != 1 || !producer.getResult(0).hasOneUse())
        return std::nullopt;
      mlir::OpOperand &edge = *producer.getResult(0).getUses().begin();
      auto consumer = mlir::dyn_cast<mlir::shuttle::MapOp>(edge.getOwner());
      if (!consumer || !group.contains(consumer.getSource()))
        return std::nullopt;
      auto edgeMap = mlir::cast<mlir::AffineMapAttr>(
                         consumer.getIndexingMaps()[edge.getOperandNumber()])
                         .getValue();
      if (!edgeMap.isIdentity() ||
          mapDomainShape(producer) != mapDomainShape(consumer))
        return std::nullopt;
      for (auto [operand, indexing] :
           llvm::zip_equal(producer.getInputs(),
                           producer.getIndexingMaps().getValue().take_front(
                               producer.getInputs().size()))) {
        auto inputMap = mlir::cast<mlir::AffineMapAttr>(indexing).getValue();
        const bool projected = llvm::any_of(
            inputMap.getResults(), [](mlir::AffineExpr expression) {
              return expression.getKind() == mlir::AffineExprKind::FloorDiv;
            });
        auto inputProducer = operand.getDefiningOp<mlir::shuttle::MapOp>();
        if (projected &&
            (!inputProducer || !group.contains(inputProducer.getSource())))
          audit.sawComposedProjectedInput = true;
      }
    }
  }
  return audit;
}

bool mutateFusedTask(mlir::ModuleOp module, llvm::StringRef mutation) {
  llvm::SmallVector<mlir::shuttle::MaterializationPlanOp> plans(
      module.getOps<mlir::shuttle::MaterializationPlanOp>());
  if (plans.size() != 1)
    return false;
  for (mlir::shuttle::MaterializationTaskOp task :
       plans.front()
           .getBody()
           .front()
           .getOps<mlir::shuttle::MaterializationTaskOp>()) {
    auto sources = task->getAttrOfType<mlir::ArrayAttr>("fused_sources");
    auto maps = task->getAttrOfType<mlir::ArrayAttr>("composed_input_maps");
    if (!sources || sources.size() < 2 || !maps || maps.size() < 2)
      continue;
    mlir::Builder builder(module.getContext());
    llvm::SmallVector<mlir::Attribute> changed;
    if (mutation == "duplicate-source") {
      llvm::append_range(changed, sources);
      changed[1] = changed[0];
      task->setAttr("fused_sources", builder.getArrayAttr(changed));
    } else if (mutation == "reordered-source") {
      llvm::append_range(changed, llvm::reverse(sources.getValue()));
      task->setAttr("fused_sources", builder.getArrayAttr(changed));
    } else if (mutation == "stale-source") {
      llvm::append_range(changed, sources);
      changed[0] = mlir::shuttle::SourceRefAttr::get(module.getContext(), 99,
                                                     99, 99, 99);
      task->setAttr("fused_sources", builder.getArrayAttr(changed));
    } else if (mutation == "missing-map") {
      task->setAttr("composed_input_maps",
                    builder.getArrayAttr(maps.getValue().drop_back()));
    } else if (mutation == "reordered-map") {
      llvm::append_range(changed, llvm::reverse(maps.getValue()));
      task->setAttr("composed_input_maps", builder.getArrayAttr(changed));
    } else {
      return false;
    }
    plans.front().setFingerprint(
        mlir::shuttle::materializationPlanFingerprint(plans.front()));
    return true;
  }
  return false;
}

bool mutateFoldEpilogueTask(mlir::ModuleOp module, llvm::StringRef mutation) {
  llvm::SmallVector<mlir::shuttle::MaterializationPlanOp> plans(
      module.getOps<mlir::shuttle::MaterializationPlanOp>());
  if (plans.size() != 1)
    return false;
  for (mlir::shuttle::MaterializationTaskOp task :
       plans.front()
           .getBody()
           .front()
           .getOps<mlir::shuttle::MaterializationTaskOp>()) {
    auto sources =
        task->getAttrOfType<mlir::ArrayAttr>("fold_epilogue_sources");
    auto maps = task->getAttrOfType<mlir::ArrayAttr>("epilogue_input_maps");
    if (!sources || sources.size() < 2 || !maps || maps.size() < 2)
      continue;
    mlir::Builder builder(module.getContext());
    llvm::SmallVector<mlir::Attribute> changed;
    if (mutation == "duplicate-source") {
      llvm::append_range(changed, sources);
      changed[1] = changed[0];
      task->setAttr("fold_epilogue_sources", builder.getArrayAttr(changed));
    } else if (mutation == "reordered-source") {
      llvm::append_range(changed, llvm::reverse(sources.getValue()));
      task->setAttr("fold_epilogue_sources", builder.getArrayAttr(changed));
    } else if (mutation == "stale-source") {
      llvm::append_range(changed, sources);
      changed[0] = mlir::shuttle::SourceRefAttr::get(module.getContext(), 99,
                                                     99, 99, 99);
      task->setAttr("fold_epilogue_sources", builder.getArrayAttr(changed));
    } else if (mutation == "missing-map") {
      task->setAttr("epilogue_input_maps",
                    builder.getArrayAttr(maps.getValue().drop_back()));
    } else if (mutation == "reordered-map") {
      llvm::append_range(changed, llvm::reverse(maps.getValue()));
      if (maps.getValue() == llvm::ArrayRef<mlir::Attribute>(changed))
        continue;
      task->setAttr("epilogue_input_maps", builder.getArrayAttr(changed));
    } else {
      return false;
    }
    plans.front().setFingerprint(
        mlir::shuttle::materializationPlanFingerprint(plans.front()));
    return true;
  }
  return false;
}

bool mutateFoldRegionTask(mlir::ModuleOp module, llvm::StringRef mutation) {
  auto plan = *module.getOps<mlir::shuttle::MaterializationPlanOp>().begin();
  for (mlir::shuttle::MaterializationTaskOp task :
       plan.getBody().front().getOps<mlir::shuttle::MaterializationTaskOp>()) {
    auto sources = task->getAttrOfType<mlir::ArrayAttr>("fold_region_sources");
    auto phases = task->getAttrOfType<mlir::ArrayAttr>("fold_region_phases");
    auto boundaryUses =
        task->getAttrOfType<mlir::ArrayAttr>("fold_region_boundary_uses");
    if (!sources || !phases || !boundaryUses || boundaryUses.empty())
      continue;
    mlir::Builder builder(module.getContext());
    auto replaceField = [&](mlir::DictionaryAttr dictionary,
                            llvm::StringRef name,
                            mlir::Attribute value) -> mlir::DictionaryAttr {
      llvm::SmallVector<mlir::NamedAttribute> fields(dictionary.begin(),
                                                     dictionary.end());
      auto field = llvm::find_if(fields, [&](mlir::NamedAttribute candidate) {
        return candidate.getName() == name;
      });
      if (field == fields.end())
        return {};
      *field = builder.getNamedAttr(name, value);
      return builder.getDictionaryAttr(fields);
    };
    if (mutation == "boundary-buffer") {
      if (task.getInputBuffers().size() < 3)
        continue;
      llvm::SmallVector<mlir::Attribute> changed(boundaryUses.getValue());
      auto use = mlir::cast<mlir::DictionaryAttr>(changed.front());
      llvm::SmallVector<mlir::NamedAttribute> fields(use.begin(), use.end());
      auto inputBuffer = llvm::find_if(fields, [](mlir::NamedAttribute field) {
        return field.getName() == "input_buffer";
      });
      if (inputBuffer == fields.end())
        return false;
      *inputBuffer =
          builder.getNamedAttr("input_buffer", builder.getI64IntegerAttr(2));
      changed.front() = builder.getDictionaryAttr(fields);
      task->setAttr("fold_region_boundary_uses", builder.getArrayAttr(changed));
    } else if (mutation == "missing-boundary") {
      if (boundaryUses.size() < 2)
        continue;
      task->setAttr("fold_region_boundary_uses",
                    builder.getArrayAttr(boundaryUses.getValue().drop_back()));
    } else if (mutation == "duplicate-boundary") {
      llvm::SmallVector<mlir::Attribute> changed(boundaryUses.getValue());
      changed.push_back(changed.front());
      task->setAttr("fold_region_boundary_uses", builder.getArrayAttr(changed));
    } else if (mutation == "nonlocal-phase") {
      if (phases.size() < 2)
        continue;
      llvm::SmallVector<mlir::Attribute> changed(phases.getValue());
      auto phase = mlir::cast<mlir::DictionaryAttr>(changed[1]);
      llvm::SmallVector<mlir::NamedAttribute> fields(phase.begin(),
                                                     phase.end());
      auto transition = llvm::find_if(fields, [](mlir::NamedAttribute field) {
        return field.getName() == "transition";
      });
      if (transition == fields.end())
        return false;
      *transition = builder.getNamedAttr(
          "transition", builder.getStringAttr("launch_boundary"));
      changed[1] = builder.getDictionaryAttr(fields);
      task->setAttr("fold_region_phases", builder.getArrayAttr(changed));
    } else if (mutation == "missing-phase") {
      if (phases.size() < 2)
        continue;
      task->setAttr("fold_region_phases",
                    builder.getArrayAttr(phases.getValue().drop_back()));
    } else if (mutation == "duplicate-phase") {
      if (phases.size() < 2)
        continue;
      llvm::SmallVector<mlir::Attribute> changed(phases.getValue());
      changed.push_back(changed[1]);
      task->setAttr("fold_region_phases", builder.getArrayAttr(changed));
    } else if (mutation == "reordered-phase") {
      if (phases.size() < 2)
        continue;
      llvm::SmallVector<mlir::Attribute> changed(
          llvm::reverse(phases.getValue()));
      for (auto [index, phase] : llvm::enumerate(changed)) {
        changed[index] =
            replaceField(mlir::cast<mlir::DictionaryAttr>(phase), "transition",
                         builder.getStringAttr(index == 0 ? "launch_boundary"
                                                          : "cta_local"));
      }
      task->setAttr("fold_region_phases", builder.getArrayAttr(changed));
    } else if (mutation == "phase-domain-shape" ||
               mutation == "phase-reduction-dimensions" ||
               mutation == "phase-order-free") {
      llvm::SmallVector<mlir::Attribute> changed(phases.getValue());
      auto phase = mlir::cast<mlir::DictionaryAttr>(changed.front());
      mlir::Attribute replacement;
      llvm::StringRef field;
      if (mutation == "phase-domain-shape") {
        field = "domain_shape";
        replacement = builder.getDenseI64ArrayAttr({1});
      } else if (mutation == "phase-reduction-dimensions") {
        field = "reduction_dimensions";
        replacement = builder.getDenseI64ArrayAttr({0});
      } else {
        field = "order_free";
        auto orderFree = mlir::cast<mlir::BoolAttr>(phase.get(field));
        replacement = builder.getBoolAttr(!orderFree.getValue());
      }
      changed.front() = replaceField(phase, field, replacement);
      if (!changed.front())
        return false;
      task->setAttr("fold_region_phases", builder.getArrayAttr(changed));
    } else if (mutation == "wrong-valid-phase") {
      if (phases.size() < 2)
        continue;
      llvm::SmallVector<mlir::Attribute> changed(boundaryUses.getValue());
      auto use = mlir::cast<mlir::DictionaryAttr>(changed.front());
      llvm::SmallVector<mlir::NamedAttribute> fields(use.begin(), use.end());
      auto phase = llvm::find_if(fields, [](mlir::NamedAttribute field) {
        return field.getName() == "phase";
      });
      if (phase == fields.end())
        return false;
      *phase = builder.getNamedAttr("phase", builder.getI64IntegerAttr(1));
      changed.front() = builder.getDictionaryAttr(fields);
      task->setAttr("fold_region_boundary_uses", builder.getArrayAttr(changed));
    } else if (mutation == "boundary-indexing-map") {
      llvm::SmallVector<mlir::Attribute> changed(boundaryUses.getValue());
      auto use = mlir::cast<mlir::DictionaryAttr>(changed.front());
      auto source = mlir::cast<mlir::shuttle::SourceRefAttr>(use.get("source"));
      auto operand = mlir::cast<mlir::IntegerAttr>(use.get("operand"));
      mlir::shuttle::MapOp map;
      module.walk([&](mlir::shuttle::MapOp candidate) {
        if (candidate.getSource() == source)
          map = candidate;
      });
      if (!map)
        return false;
      {
        auto original = mlir::cast<mlir::AffineMapAttr>(
            use.get("indexing_map")).getValue();
        if (original.getNumResults() == 0)
          continue;
        llvm::SmallVector<mlir::AffineExpr> results(original.getResults());
        results.back() = builder.getAffineConstantExpr(0);
        auto replacement = mlir::AffineMap::get(
            original.getNumDims(), original.getNumSymbols(), results,
            module.getContext());
        if (replacement == original)
          continue;
        use = replaceField(use, "indexing_map",
                           mlir::AffineMapAttr::get(replacement));
      }
      if (!use)
        return false;
      changed.front() = use;
      task->setAttr("fold_region_boundary_uses", builder.getArrayAttr(changed));
    } else if (mutation == "dangling-recomputation") {
      llvm::SmallVector<mlir::Attribute> changed;
      if (auto recomputed = task->getAttrOfType<mlir::ArrayAttr>(
              "fold_region_recomputed_sources"))
        llvm::append_range(changed, recomputed);
      changed.push_back(mlir::shuttle::SourceRefAttr::get(module.getContext(),
                                                          99, 99, 99, 99));
      task->setAttr("fold_region_recomputed_sources",
                    builder.getArrayAttr(changed));
    } else if (mutation == "duplicate-recomputation") {
      auto recomputed = task->getAttrOfType<mlir::ArrayAttr>(
          "fold_region_recomputed_sources");
      if (!recomputed || recomputed.empty())
        continue;
      llvm::SmallVector<mlir::Attribute> changed(recomputed.getValue());
      changed.push_back(changed.front());
      task->setAttr("fold_region_recomputed_sources",
                    builder.getArrayAttr(changed));
    } else if (mutation == "missing-recomputation") {
      auto recomputed = task->getAttrOfType<mlir::ArrayAttr>(
          "fold_region_recomputed_sources");
      if (!recomputed || recomputed.empty())
        continue;
      task->setAttr("fold_region_recomputed_sources",
                    builder.getArrayAttr(recomputed.getValue().drop_back()));
    } else if (mutation == "reordered-source") {
      llvm::SmallVector<mlir::Attribute> changed(
          llvm::reverse(sources.getValue()));
      task->setAttr("fold_region_sources", builder.getArrayAttr(changed));
    } else {
      return false;
    }
    plan.setFingerprint(mlir::shuttle::materializationPlanFingerprint(plan));
    return true;
  }
  return false;
}

bool mutateContractRegionTask(mlir::ModuleOp module,
                              llvm::StringRef mutation) {
  auto plan = *module.getOps<mlir::shuttle::MaterializationPlanOp>().begin();
  mlir::shuttle::MaterializationTaskOp contractTask;
  for (auto task : plan.getBody().front().getOps<
           mlir::shuttle::MaterializationTaskOp>()) {
    if (task->getAttr("contract_region_sources")) {
      contractTask = task;
      break;
    }
  }
  if (!contractTask)
    return false;
  mlir::Builder builder(module.getContext());
  if (mutation == "strategy") {
    auto strategy = contractTask->getAttrOfType<mlir::StringAttr>(
        "contract_region_strategy");
    if (!strategy)
      return false;
    contractTask->setAttr(
        "contract_region_strategy",
        builder.getStringAttr(strategy.getValue() == "inline_fold"
                                  ? "compact_stat"
                                  : "inline_fold"));
  } else if (mutation == "cost") {
    auto cost = contractTask->getAttrOfType<mlir::DictionaryAttr>(
        "contract_region_cost");
    if (!cost)
      return false;
    llvm::SmallVector<mlir::NamedAttribute> fields(cost.begin(), cost.end());
    auto rows = llvm::find_if(fields, [](mlir::NamedAttribute field) {
      return field.getName() == "rows";
    });
    if (rows == fields.end())
      return false;
    int64_t value = mlir::cast<mlir::IntegerAttr>(rows->getValue()).getInt();
    *rows = builder.getNamedAttr("rows", builder.getI64IntegerAttr(value + 1));
    contractTask->setAttr("contract_region_cost",
                          builder.getDictionaryAttr(fields));
  } else if (mutation == "sources") {
    auto sources = contractTask->getAttrOfType<mlir::ArrayAttr>(
        "contract_region_sources");
    if (!sources || sources.size() < 2)
      return false;
    contractTask->setAttr(
        "contract_region_sources",
        builder.getArrayAttr(sources.getValue().drop_back()));
  } else if (mutation == "fold-source" || mutation == "contract-source" ||
             mutation == "boundary-source") {
    auto foldSource =
        contractTask->getAttrOfType<mlir::shuttle::SourceRefAttr>(
            "contract_region_fold_source");
    auto contractSource =
        contractTask->getAttrOfType<mlir::shuttle::SourceRefAttr>(
            "contract_region_contract_source");
    auto epilogue = contractTask->getAttrOfType<mlir::ArrayAttr>(
        "contract_region_epilogue_sources");
    if (!foldSource || !contractSource || !epilogue || epilogue.empty())
      return false;
    contractTask->setAttr(
        mutation == "fold-source"
            ? "contract_region_fold_source"
            : mutation == "contract-source"
                  ? "contract_region_contract_source"
                  : "contract_region_bf16_boundary_source",
        mutation == "fold-source" ? mlir::Attribute(contractSource)
                                   : mutation == "contract-source"
                                         ? mlir::Attribute(foldSource)
                                         : epilogue.getValue().front());
  } else if (mutation == "normalized-operand") {
    contractTask->setAttr("contract_region_normalized_operand",
                          builder.getI64IntegerAttr(1));
  } else if (mutation == "epilogue") {
    contractTask->setAttr("contract_region_epilogue_sources",
                          builder.getArrayAttr({}));
  } else if (mutation == "recomputation") {
    auto recomputed = contractTask->getAttrOfType<mlir::ArrayAttr>(
        "contract_region_recomputed_sources");
    if (!recomputed || recomputed.empty())
      return false;
    contractTask->setAttr(
        "contract_region_recomputed_sources",
        builder.getArrayAttr(recomputed.getValue().drop_back()));
  } else if (mutation == "buffers") {
    llvm::SmallVector<int64_t> inputs(contractTask.getInputBuffers());
    if (inputs.size() < 2)
      return false;
    std::swap(inputs[0], inputs[1]);
    contractTask->setAttr("input_buffers",
                          builder.getDenseI64ArrayAttr(inputs));
  } else if (mutation == "ssa-edge") {
    mlir::shuttle::MapOp replacement;
    mlir::shuttle::MapOp target;
    module.walk([&](mlir::shuttle::MapOp map) {
      if (map.getSource().getOperationOrdinal() == 1)
        replacement = map;
      if (map.getSource().getOperationOrdinal() == 13)
        target = map;
    });
    if (!replacement || !target || target.getInputs().empty() ||
        replacement.getNumResults() != 1 ||
        replacement.getResult(0).getType() != target.getInputs()[0].getType())
      return false;
    target->setOperand(0, replacement.getResult(0));
  } else {
    return false;
  }
  plan.setFingerprint(mlir::shuttle::materializationPlanFingerprint(plan));
  return true;
}

bool mutateHybridNormalizedOutput(mlir::ModuleOp module,
                                  llvm::StringRef mutation) {
  auto plan = *module.getOps<mlir::shuttle::MaterializationPlanOp>().begin();
  llvm::SmallVector<mlir::shuttle::MaterializationBufferOp> buffers(
      plan.getBody().front().getOps<
          mlir::shuttle::MaterializationBufferOp>());
  for (auto task : plan.getBody().front().getOps<
           mlir::shuttle::MaterializationTaskOp>()) {
    if (!task->getAttr("contract_region_sources") ||
        task.getOutputBuffers().size() != 2)
      continue;
    mlir::Builder builder(module.getContext());
    llvm::SmallVector<int64_t> outputs(task.getOutputBuffers());
    if (mutation == "output-order") {
      std::swap(outputs[0], outputs[1]);
      task->setAttr("output_buffers", builder.getDenseI64ArrayAttr(outputs));
    } else {
      auto normalized = llvm::find_if(buffers, [&](auto buffer) {
        return buffer.getOrdinal() == outputs.front();
      });
      if (normalized == buffers.end())
        return false;
      if (mutation == "live-out") {
        (*normalized)->setAttr("live_out", builder.getBoolAttr(true));
      } else if (mutation == "storage") {
        (*normalized)->setAttr(
            "storage", mlir::shuttle::MaterializationStorageAttr::get(
                           module.getContext(),
                           mlir::shuttle::MaterializationStorage::External));
      } else {
        return false;
      }
    }
    plan.setFingerprint(mlir::shuttle::materializationPlanFingerprint(plan));
    return true;
  }
  return false;
}

bool mutateGenericContractFoldComposition(mlir::ModuleOp module,
                                          llvm::StringRef mutation) {
  auto plans =
      llvm::to_vector(module.getOps<mlir::shuttle::MaterializationPlanOp>());
  if (plans.size() != 1)
    return false;
  auto tasks =
      llvm::to_vector(plans.front()
                          .getBody()
                          .front()
                          .getOps<mlir::shuttle::MaterializationTaskOp>());
  auto buffers =
      llvm::to_vector(plans.front()
                          .getBody()
                          .front()
                          .getOps<mlir::shuttle::MaterializationBufferOp>());
  auto byKind = [&](llvm::StringRef kind) {
    return llvm::find_if(tasks, [&](auto task) {
      auto value =
          task->template getAttrOfType<mlir::StringAttr>("composition_kind");
      return value && value.getValue() == kind;
    });
  };
  auto first = byKind("contract_fold");
  auto second = byKind("map_contract");
  if (first == tasks.end() || second == tasks.end() ||
      first->getOutputBuffers().size() != 2)
    return false;
  auto bufferByOrdinal = [&](int64_t ordinal) {
    return llvm::find_if(
        buffers, [&](auto buffer) { return buffer.getOrdinal() == ordinal; });
  };
  auto full = bufferByOrdinal(first->getOutputBuffers().front());
  auto compact = bufferByOrdinal(first->getOutputBuffers().back());
  if (full == buffers.end() || compact == buffers.end())
    return false;
  mlir::Builder builder(module.getContext());
  if (mutation == "output-order") {
    llvm::SmallVector<int64_t> outputs(first->getOutputBuffers());
    std::swap(outputs[0], outputs[1]);
    (*first)->setAttr("output_buffers", builder.getDenseI64ArrayAttr(outputs));
  } else if (mutation == "missing-output") {
    (*first)->setAttr(
        "output_buffers",
        builder.getDenseI64ArrayAttr(first->getOutputBuffers().drop_back()));
  } else if (mutation == "compact-type") {
    (*compact)->setAttr("tensor_type",
                        mlir::TypeAttr::get(full->getTensorType()));
  } else if (mutation == "compact-element-type") {
    auto type = mlir::cast<mlir::RankedTensorType>(compact->getTensorType());
    (*compact)->setAttr("tensor_type",
                        mlir::TypeAttr::get(mlir::RankedTensorType::get(
                            type.getShape(), builder.getF16Type())));
  } else if (mutation == "extra-public-result") {
    (*compact)->setAttr("live_out", builder.getBoolAttr(true));
  } else if (mutation == "full-live-out") {
    (*full)->setAttr("live_out", builder.getBoolAttr(true));
  } else if (mutation == "boundary-storage") {
    (*compact)->setAttr("storage",
                        mlir::shuttle::MaterializationStorageAttr::get(
                            module.getContext(),
                            mlir::shuttle::MaterializationStorage::External));
  } else if (mutation == "boundary-producer") {
    (*compact)->setAttr("producer",
                        builder.getI64IntegerAttr(second->getOrdinal()));
  } else if (mutation == "lifetime") {
    (*compact)->setAttr("lifetime_end",
                        builder.getI64IntegerAttr(first->getOrdinal()));
  } else if (mutation == "missing-consumer") {
    (*compact)->setAttr("consumers", builder.getDenseI64ArrayAttr({}));
  } else if (mutation == "missing-second-input") {
    llvm::SmallVector<int64_t> inputs;
    llvm::copy_if(
        second->getInputBuffers(), std::back_inserter(inputs),
        [&](int64_t ordinal) { return ordinal != compact->getOrdinal(); });
    (*second)->setAttr("input_buffers", builder.getDenseI64ArrayAttr(inputs));
  } else if (mutation == "missing-public-input") {
    llvm::SmallVector<int64_t> inputs(first->getInputBuffers());
    auto publicInput = llvm::find_if(inputs, [&](int64_t ordinal) {
      auto buffer = bufferByOrdinal(ordinal);
      return buffer != buffers.end() && buffer->getLiveIn();
    });
    if (publicInput == inputs.end())
      return false;
    inputs.erase(publicInput);
    (*first)->setAttr("input_buffers", builder.getDenseI64ArrayAttr(inputs));
  } else if (mutation == "rebound-public-input") {
    llvm::SmallVector<int64_t> inputs(first->getInputBuffers());
    llvm::SmallVector<size_t> publicPositions;
    for (auto [position, ordinal] : llvm::enumerate(inputs)) {
      auto buffer = bufferByOrdinal(ordinal);
      if (buffer != buffers.end() && buffer->getLiveIn())
        publicPositions.push_back(position);
    }
    if (publicPositions.size() < 2)
      return false;
    std::swap(inputs[publicPositions[0]], inputs[publicPositions[1]]);
    (*first)->setAttr("input_buffers", builder.getDenseI64ArrayAttr(inputs));
  } else if (mutation == "missing-second-public-input" ||
             mutation == "rebound-second-public-input" ||
             mutation == "second-input-order") {
    llvm::SmallVector<int64_t> inputs(second->getInputBuffers());
    auto publicInput = llvm::find_if(inputs, [&](int64_t ordinal) {
      auto buffer = bufferByOrdinal(ordinal);
      return buffer != buffers.end() && buffer->getLiveIn();
    });
    if (publicInput == inputs.end())
      return false;
    if (mutation == "missing-second-public-input") {
      inputs.erase(publicInput);
    } else if (mutation == "rebound-second-public-input") {
      auto replacement = llvm::find_if(buffers, [&](auto buffer) {
        return buffer.getLiveIn() && buffer.getOrdinal() != *publicInput;
      });
      if (replacement == buffers.end())
        return false;
      *publicInput = replacement->getOrdinal();
    } else {
      auto privateInput = llvm::find(inputs, compact->getOrdinal());
      if (privateInput == inputs.end())
        return false;
      std::iter_swap(publicInput, privateInput);
    }
    (*second)->setAttr("input_buffers", builder.getDenseI64ArrayAttr(inputs));
  } else if (mutation == "missing-public-result" ||
             mutation == "rebound-public-result") {
    auto publicResult =
        llvm::find_if(buffers, [](auto buffer) { return buffer.getLiveOut(); });
    if (publicResult == buffers.end())
      return false;
    llvm::SmallVector<int64_t> outputs(second->getOutputBuffers());
    auto output = llvm::find(outputs, publicResult->getOrdinal());
    if (output == outputs.end())
      return false;
    if (mutation == "missing-public-result")
      outputs.erase(output);
    else
      *output = compact->getOrdinal();
    (*second)->setAttr("output_buffers", builder.getDenseI64ArrayAttr(outputs));
  } else if (mutation == "missing-kind") {
    (*first)->removeAttr("composition_kind");
  } else if (mutation == "unknown-kind") {
    (*first)->setAttr("composition_kind",
                      builder.getStringAttr("contract_fold_unknown"));
  } else if (mutation == "source-ownership") {
    auto sources =
        (*second)->getAttrOfType<mlir::ArrayAttr>("composition_sources");
    if (!sources || sources.size() < 2)
      return false;
    (*second)->setAttr("composition_sources",
                       builder.getArrayAttr(sources.getValue().drop_back()));
  } else if (mutation == "duplicate-source-ownership") {
    auto firstSources =
        (*first)->getAttrOfType<mlir::ArrayAttr>("composition_sources");
    auto secondSources =
        (*second)->getAttrOfType<mlir::ArrayAttr>("composition_sources");
    if (!firstSources || firstSources.empty() || !secondSources)
      return false;
    llvm::SmallVector<mlir::Attribute> sources(secondSources.getValue());
    sources.push_back(firstSources[0]);
    (*second)->setAttr("composition_sources", builder.getArrayAttr(sources));
  } else if (mutation == "cost" || mutation == "underreported-cost" ||
             mutation == "missing-cost-field" ||
             mutation == "contract-evaluations") {
    auto cost =
        (*second)->getAttrOfType<mlir::DictionaryAttr>("composition_cost");
    if (!cost)
      return false;
    llvm::SmallVector<mlir::NamedAttribute> fields(cost.begin(), cost.end());
    auto baseline = llvm::find_if(fields, [](mlir::NamedAttribute field) {
      return field.getName() == "baseline_temporary_bytes";
    });
    auto candidate = llvm::find_if(fields, [](mlir::NamedAttribute field) {
      return field.getName() == "candidate_temporary_bytes";
    });
    if (baseline == fields.end() || candidate == fields.end())
      return false;
    if (mutation == "missing-cost-field") {
      fields.erase(candidate);
      (*second)->setAttr("composition_cost", builder.getDictionaryAttr(fields));
    } else if (mutation == "contract-evaluations") {
      auto evaluations = llvm::find_if(fields, [](mlir::NamedAttribute field) {
        return field.getName() == "contract_evaluations";
      });
      if (evaluations == fields.end())
        return false;
      *evaluations = builder.getNamedAttr(
          "contract_evaluations", builder.getI64IntegerAttr(2));
      (*second)->setAttr("composition_cost", builder.getDictionaryAttr(fields));
    } else {
      const int64_t baselineBytes =
          mlir::cast<mlir::IntegerAttr>(baseline->getValue()).getInt();
      *candidate = builder.getNamedAttr(
          "candidate_temporary_bytes",
          builder.getI64IntegerAttr(mutation == "cost" ? baselineBytes + 1
                                                       : baselineBytes - 1));
      (*second)->setAttr("composition_cost",
                         builder.getDictionaryAttr(fields));
    }
  } else {
    return false;
  }
  plans.front().setFingerprint(
      mlir::shuttle::materializationPlanFingerprint(plans.front()));
  return true;
}

bool verifyMaterializationPlan(mlir::ModuleOp module) {
  mlir::PassManager manager(module.getContext());
  manager.addPass(mlir::shuttle::createVerifyMaterializationPlanPass());
  return mlir::succeeded(manager.run(module));
}

void expectCounts(const PlanSummary &actual, const FixtureExpectation &expected,
                  bool fused) {
  EXPECT_EQ(actual.maps, expected.maps);
  EXPECT_EQ(actual.folds, expected.folds);
  EXPECT_EQ(actual.contracts, 0);
  if (actual.hasFoldRegion) {
    EXPECT_TRUE(fused);
    if (expected.maps == 18 && expected.folds == 1) {
      EXPECT_EQ(actual.tasks, 1);
      EXPECT_EQ(actual.buffers, 3);
      EXPECT_EQ(actual.foldTasks, 1);
    } else if (expected.maps == 43 && expected.folds == 5) {
      EXPECT_EQ(actual.tasks, 3);
      EXPECT_EQ(actual.buffers, 6);
      EXPECT_EQ(actual.foldTasks, 3);
    } else if (expected.maps == 46 && expected.folds == 5) {
      EXPECT_EQ(actual.tasks, 4);
      EXPECT_EQ(actual.buffers, 7);
      EXPECT_EQ(actual.foldTasks, 4);
    } else if (expected.maps == 29 && expected.folds == 2) {
      EXPECT_EQ(actual.tasks, 1);
      EXPECT_EQ(actual.buffers, 4);
      EXPECT_EQ(actual.foldTasks, 1);
    } else if (expected.maps == 63 && expected.folds == 10) {
      EXPECT_EQ(actual.tasks, 2);
      EXPECT_EQ(actual.buffers, 8);
      EXPECT_EQ(actual.foldTasks, 2);
    } else if (expected.maps == 70 && expected.folds == 10) {
      EXPECT_EQ(actual.tasks, 2);
      EXPECT_EQ(actual.buffers, 10);
      EXPECT_EQ(actual.foldTasks, 2);
    } else {
      ADD_FAILURE() << "unexpected fixture admitted as a Fold region";
    }
    EXPECT_EQ(actual.mapTasks, 0);
    EXPECT_EQ(actual.contractTasks, 0);
    EXPECT_TRUE(actual.exactSourceCoverage);
    EXPECT_TRUE(actual.fusedMapGroupSizes.empty());
    EXPECT_TRUE(actual.fusedFoldMapGroupSizes.empty());
    return;
  }
  EXPECT_EQ(actual.tasks,
            fused ? expected.fusedTasks - actual.eligibleFoldMapEdges
                  : expected.materializedTasks);
  EXPECT_EQ(actual.buffers,
            fused ? expected.fusedBuffers - actual.eligibleFoldMapEdges
                  : expected.materializedBuffers);
  EXPECT_EQ(actual.foldTasks, expected.folds);
  EXPECT_EQ(actual.contractTasks, 0);
  EXPECT_EQ(actual.mapTasks, actual.tasks - actual.foldTasks);
  EXPECT_TRUE(actual.exactSourceCoverage);
  EXPECT_EQ(actual.fusedMapGroupSizes, fused
                                           ? expectedFusionGroupSizes(expected)
                                           : std::vector<int64_t>{});
  EXPECT_EQ(actual.fusedFoldMapGroupSizes.size(),
            fused ? actual.eligibleFoldMapEdges : 0);
}

TEST(Abi22MapFusionContract,
     PinnedOrdinaryJaxMatrixBindsStructuralTaskAndBufferReductions) {
  constexpr mlir::shuttle::NumericalPolicy policies[] = {
      mlir::shuttle::NumericalPolicy::SourceOrdered,
      mlir::shuttle::NumericalPolicy::Fast,
  };
  for (const FixtureExpectation &fixture : kFixtures) {
    SCOPED_TRACE(fixture.filename);
    std::optional<std::string> source = fixtureSource(fixture.filename);
    ASSERT_TRUE(source);
    std::optional<PlanSummary> sourceOrderedFusion;
    for (mlir::shuttle::NumericalPolicy policy : policies) {
      SCOPED_TRACE(policy == mlir::shuttle::NumericalPolicy::Fast ? "fast"
                                                                  : "source");
      std::unique_ptr<BuiltModule> automatic =
          build(*source, policy, kAutomaticTuning, false, false);
      ASSERT_TRUE(automatic);
      std::optional<PlanSummary> automaticSummary = summary(*automatic->module);
      ASSERT_TRUE(automaticSummary);
      expectCounts(*automaticSummary, fixture, false);

      std::unique_ptr<BuiltModule> materialized =
          build(*source, policy, kMaterializationTuning, false, false);
      ASSERT_TRUE(materialized);
      std::optional<PlanSummary> materializedSummary =
          summary(*materialized->module);
      ASSERT_TRUE(materializedSummary);
      expectCounts(*materializedSummary, fixture, false);

      std::unique_ptr<BuiltModule> fused =
          build(*source, policy, kFusionTuning, false, false);
      ASSERT_TRUE(fused);
      std::optional<PlanSummary> fusedSummary = summary(*fused->module);
      ASSERT_TRUE(fusedSummary);
      expectCounts(*fusedSummary, fixture, true);
      std::optional<FusionAudit> audit = auditFusionGroups(*fused->module);
      ASSERT_TRUE(audit);
      const int64_t epilogueMapGroups =
          llvm::count_if(fusedSummary->fusedFoldMapGroupSizes,
                         [](int64_t size) { return size > 1; });
      if (!fusedSummary->hasFoldRegion) {
        EXPECT_EQ(
            audit->groups + epilogueMapGroups,
            static_cast<int64_t>(expectedFusionGroupSizes(fixture).size()));
        if (llvm::StringRef(fixture.filename)
                .contains("projected_parameter_broadcast"))
          EXPECT_TRUE(audit->sawComposedProjectedInput);
      }
      EXPECT_EQ(fixture.materializedTasks - fixture.fusedTasks,
                fixture.materializedBuffers - fixture.fusedBuffers);

      std::unique_ptr<BuiltModule> renamed =
          build(*source, policy, kFusionTuning, true, false);
      ASSERT_TRUE(renamed);
      std::optional<PlanSummary> renamedSummary = summary(*renamed->module);
      ASSERT_TRUE(renamedSummary);
      expectCounts(*renamedSummary, fixture, true);
      EXPECT_EQ(renamedSummary->fingerprint, fusedSummary->fingerprint);

      if (sourceOrderedFusion) {
        EXPECT_EQ(sourceOrderedFusion->tasks, fusedSummary->tasks);
        EXPECT_EQ(sourceOrderedFusion->buffers, fusedSummary->buffers);
        EXPECT_NE(sourceOrderedFusion->fingerprint, fusedSummary->fingerprint);
      } else {
        sourceOrderedFusion = *fusedSummary;
      }
    }
  }
}

TEST(Abi22MapFusionContract, DependentRowFoldsShareOneGenericFoldRegion) {
  std::optional<std::string> source =
      fixtureSource("jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-forward.mlir");
  ASSERT_TRUE(source);
  std::unique_ptr<BuiltModule> built =
      build(*source, mlir::shuttle::NumericalPolicy::SourceOrdered,
            kFusionTuning, false, true);
  ASSERT_TRUE(built);
  std::optional<PlanSummary> plan = summary(*built->module);
  ASSERT_TRUE(plan);

  EXPECT_TRUE(plan->hasFoldRegion);
  EXPECT_EQ(plan->tasks, 1);
  EXPECT_EQ(plan->buffers, 4);
  EXPECT_EQ(plan->foldTasks, 1);
  EXPECT_EQ(plan->mapTasks, 0);
  EXPECT_TRUE(plan->exactSourceCoverage);

  auto device = *built->module->getOps<mlir::shuttle::DeviceModuleOp>().begin();
  auto entries = llvm::to_vector(
      device.getBody().front().getOps<mlir::shuttle::DeviceEntryOp>());
  ASSERT_EQ(entries.size(), 1u);
  ASSERT_TRUE(entries.front().getBlock());
  EXPECT_EQ(*entries.front().getBlock(), llvm::ArrayRef<int64_t>({256, 1, 1}));
  llvm::StringRef ptx(
      reinterpret_cast<const char *>(device.getCode().data() +
                                     entries.front().getCodeOffset()),
      entries.front().getCodeLength());
  EXPECT_EQ(ptx.count(", [%rd3];\n  bar.sync 0;\n"), 1u);
  EXPECT_TRUE(hasDependentRowFoldPhaseDataflow(ptx, 4096, 16));

  auto replaceAfterPhase = [&](llvm::StringRef oldText,
                               llvm::StringRef newText) {
    std::string mutation = ptx.str();
    size_t phase = mutation.find("$fold_region_phase_1_load:");
    EXPECT_NE(phase, std::string::npos);
    size_t position = mutation.find(oldText.str(), phase);
    EXPECT_NE(position, std::string::npos);
    if (position != std::string::npos)
      mutation.replace(position, oldText.size(), newText.str());
    return mutation;
  };
  EXPECT_FALSE(hasDependentRowFoldPhaseDataflow(
      replaceAfterPhase("cvt.u64.u32 %rd0, %r4;", "cvt.u64.u32 %rd0, %r0;"),
      4096, 16));
  EXPECT_FALSE(hasDependentRowFoldPhaseDataflow(
      replaceAfterPhase("cvt.u64.u32 %rd1, %r1;", "cvt.u64.u32 %rd1, %r0;"),
      4096, 16));

  std::string nonzeroInitializer = ptx.str();
  size_t phase = nonzeroInitializer.find("$fold_region_phase_1_load:");
  ASSERT_NE(phase, std::string::npos);
  size_t initializer = nonzeroInitializer.rfind("  mov.f32 ", phase);
  ASSERT_NE(initializer, std::string::npos);
  size_t zero = nonzeroInitializer.find("0f00000000", initializer);
  ASSERT_LT(zero, phase);
  nonzeroInitializer.replace(zero, strlen("0f00000000"), "0f3F800000");
  EXPECT_FALSE(hasDependentRowFoldPhaseDataflow(nonzeroInitializer, 4096, 16));

  std::string wrongCoordinate = ptx.str();
  phase = wrongCoordinate.find("$fold_region_phase_1_load:");
  size_t coordinate =
      wrongCoordinate.find("  mad.lo.u64 %rd0, %rd1, 4096, %rd0;", phase);
  ASSERT_NE(coordinate, std::string::npos);
  size_t coordinateEnd = wrongCoordinate.find('\n', coordinate);
  ASSERT_NE(coordinateEnd, std::string::npos);
  wrongCoordinate.insert(coordinateEnd + 1, "  mov.u64 %rd0, 0;\n");
  EXPECT_FALSE(hasDependentRowFoldPhaseDataflow(wrongCoordinate, 4096, 16));

  std::string wrongStride = ptx.str();
  phase = wrongStride.find("$fold_region_phase_1_load:");
  coordinate = wrongStride.find("  mad.lo.u64 %rd0, %rd1, 4096, %rd0;", phase);
  ASSERT_NE(coordinate, std::string::npos);
  coordinateEnd = wrongStride.find('\n', coordinate);
  ASSERT_NE(coordinateEnd, std::string::npos);
  wrongStride.insert(coordinateEnd + 1,
                     "  mad.lo.u64 %rd0, %rd1, 8192, %rd0;\n");
  EXPECT_FALSE(hasDependentRowFoldPhaseDataflow(wrongStride, 4096, 16));

  std::string wrongAddressScale = ptx.str();
  phase = wrongAddressScale.find("$fold_region_phase_1_load:");
  size_t firstLoad = wrongAddressScale.find("  ld.global.u16 ", phase);
  ASSERT_NE(firstLoad, std::string::npos);
  size_t secondLoad = wrongAddressScale.find("  ld.global.u16 ", firstLoad + 1);
  ASSERT_NE(secondLoad, std::string::npos);
  size_t addressScale = wrongAddressScale.rfind(", 2, ", secondLoad);
  ASSERT_GT(addressScale, firstLoad);
  wrongAddressScale.replace(addressScale + 2, 1, "4");
  EXPECT_FALSE(hasDependentRowFoldPhaseDataflow(wrongAddressScale, 4096, 16));

  std::string zeroInnerOperand = ptx.str();
  phase = zeroInnerOperand.find("$fold_region_phase_1_load:");
  size_t increment = zeroInnerOperand.find("  add.u32 %r4, %r4, 1;", phase);
  size_t combine = zeroInnerOperand.rfind("  add.rn.f32 ", increment);
  ASSERT_NE(phase, std::string::npos);
  ASSERT_NE(increment, std::string::npos);
  ASSERT_NE(combine, std::string::npos);
  size_t multiply = zeroInnerOperand.rfind("  mul.rn.f32 ", combine);
  ASSERT_NE(multiply, std::string::npos);
  size_t multiplyEnd = zeroInnerOperand.find('\n', multiply);
  std::vector<std::string> multiplyTokens =
      ptxTokens(llvm::StringRef(zeroInnerOperand).slice(multiply, multiplyEnd));
  ASSERT_EQ(multiplyTokens.size(), 3u);
  size_t innerOperand = zeroInnerOperand.find(multiplyTokens[1], multiply);
  ASSERT_LT(innerOperand, multiplyEnd);
  zeroInnerOperand.replace(innerOperand, multiplyTokens[1].size(),
                           "0f00000000");
  EXPECT_FALSE(hasDependentRowFoldPhaseDataflow(zeroInnerOperand, 4096, 16));

  std::string distinctCenteredValue = ptx.str();
  phase = distinctCenteredValue.find("$fold_region_phase_1_load:");
  increment = distinctCenteredValue.find("  add.u32 %r4, %r4, 1;", phase);
  combine = distinctCenteredValue.rfind("  add.rn.f32 ", increment);
  multiply = distinctCenteredValue.rfind("  mul.rn.f32 ", combine);
  multiplyEnd = distinctCenteredValue.find('\n', multiply);
  multiplyTokens = ptxTokens(
      llvm::StringRef(distinctCenteredValue).slice(multiply, multiplyEnd));
  ASSERT_EQ(multiplyTokens.size(), 3u);
  const std::string secondCenteredPrefix =
      (llvm::Twine("  sub.rn.f32 ") + multiplyTokens[2] + ",").str();
  size_t secondCentered =
      distinctCenteredValue.rfind(secondCenteredPrefix, multiply);
  ASSERT_NE(secondCentered, std::string::npos);
  size_t secondCenteredEnd = distinctCenteredValue.find('\n', secondCentered);
  std::vector<std::string> centeredTokens =
      ptxTokens(llvm::StringRef(distinctCenteredValue)
                    .slice(secondCentered, secondCenteredEnd));
  ASSERT_EQ(centeredTokens.size(), 3u);
  const std::string centeredAdjustment = (llvm::Twine("  add.rn.f32 %f127, ") +
                                          centeredTokens[2] + ", 0f3F800000;\n")
                                             .str();
  distinctCenteredValue.insert(secondCentered, centeredAdjustment);
  secondCentered += centeredAdjustment.size();
  secondCenteredEnd += centeredAdjustment.size();
  size_t meanOperand =
      distinctCenteredValue.find(centeredTokens[2], secondCentered);
  ASSERT_LT(meanOperand, secondCenteredEnd);
  distinctCenteredValue.replace(meanOperand, centeredTokens[2].size(), "%f127");
  EXPECT_FALSE(
      hasDependentRowFoldPhaseDataflow(distinctCenteredValue, 4096, 16));
}

TEST(Abi22MapFusionContract,
     FusedPhysicalTasksBindPtxEntriesAndTransportSlots) {
  constexpr std::array<FixtureExpectation, 2> representatives = {kFixtures[0],
                                                                 kFixtures[6]};
  constexpr mlir::shuttle::NumericalPolicy policies[] = {
      mlir::shuttle::NumericalPolicy::SourceOrdered,
      mlir::shuttle::NumericalPolicy::Fast,
  };
  for (const FixtureExpectation &fixture : representatives) {
    SCOPED_TRACE(fixture.filename);
    std::optional<std::string> source = fixtureSource(fixture.filename);
    ASSERT_TRUE(source);
    for (mlir::shuttle::NumericalPolicy policy : policies) {
      std::unique_ptr<BuiltModule> built =
          build(*source, policy, kFusionTuning, false, true);
      ASSERT_TRUE(built);
      std::unique_ptr<BuiltModule> materialized =
          build(*source, policy, kMaterializationTuning, false, true);
      ASSERT_TRUE(materialized);
      std::optional<PlanSummary> plan = summary(*built->module);
      ASSERT_TRUE(plan);
      expectCounts(*plan, fixture, true);

      llvm::SmallVector<mlir::shuttle::DeviceModuleOp> devices(
          built->module->getOps<mlir::shuttle::DeviceModuleOp>());
      llvm::SmallVector<mlir::shuttle::InvocationAbiOp> invocationAbis(
          built->module->getOps<mlir::shuttle::InvocationAbiOp>());
      llvm::SmallVector<mlir::shuttle::ExecutableBundleOp> bundles(
          built->module->getOps<mlir::shuttle::ExecutableBundleOp>());
      ASSERT_EQ(devices.size(), 1);
      ASSERT_EQ(invocationAbis.size(), 1);
      ASSERT_EQ(bundles.size(), 1);
      auto fusedPlan =
          *built->module->getOps<mlir::shuttle::MaterializationPlanOp>()
               .begin();
      llvm::SmallVector<mlir::shuttle::MaterializationTaskOp> fusedTasks(
          fusedPlan.getBody()
              .front()
              .getOps<mlir::shuttle::MaterializationTaskOp>());
      auto ordinaryPlan =
          *materialized->module->getOps<mlir::shuttle::MaterializationPlanOp>()
               .begin();
      llvm::SmallVector<mlir::shuttle::MaterializationTaskOp> ordinaryTasks(
          ordinaryPlan.getBody()
              .front()
              .getOps<mlir::shuttle::MaterializationTaskOp>());
      auto ordinaryDevice =
          *materialized->module->getOps<mlir::shuttle::DeviceModuleOp>()
               .begin();
      auto fusedEntries =
          llvm::to_vector(devices.front()
                              .getBody()
                              .front()
                              .getOps<mlir::shuttle::DeviceEntryOp>());
      auto ordinaryEntries =
          llvm::to_vector(ordinaryDevice.getBody()
                              .front()
                              .getOps<mlir::shuttle::DeviceEntryOp>());
      for (auto [ordinal, task] : llvm::enumerate(fusedTasks)) {
        if (!task->getAttr("fold_epilogue_sources"))
          continue;
        auto ordinary = llvm::find_if(ordinaryTasks, [&](auto candidate) {
          auto left = candidate.getSource();
          auto right = task.getSource();
          return left.getFunctionOrdinal() == right.getFunctionOrdinal() &&
                 left.getBlockOrdinal() == right.getBlockOrdinal() &&
                 left.getOperationOrdinal() == right.getOperationOrdinal() &&
                 left.getResultOrdinal() == right.getResultOrdinal();
        });
        ASSERT_NE(ordinary, ordinaryTasks.end());
        const size_t ordinaryOrdinal = ordinary - ordinaryTasks.begin();
        mlir::shuttle::DeviceEntryOp fusedEntry = fusedEntries[ordinal];
        mlir::shuttle::DeviceEntryOp ordinaryEntry =
            ordinaryEntries[ordinaryOrdinal];
        EXPECT_EQ(fusedEntry.getGrid(), ordinaryEntry.getGrid());
        EXPECT_EQ(fusedEntry.getBlock(), ordinaryEntry.getBlock());
        EXPECT_EQ(fusedEntry.getDynamicSharedBytes(),
                  ordinaryEntry.getDynamicSharedBytes());
        auto code = [](mlir::shuttle::DeviceModuleOp device,
                       mlir::shuttle::DeviceEntryOp entry) {
          return llvm::StringRef(
              reinterpret_cast<const char *>(device.getCode().data() +
                                             entry.getCodeOffset()),
              entry.getCodeLength());
        };
        constexpr llvm::StringLiteral reductionStart = "  mov.u32 %r0";
        constexpr llvm::StringLiteral reductionEnd =
            "  add.rn.f32 %f37, %f35, %f36;\n";
        auto reductionBody = [&](llvm::StringRef ptx) {
          size_t begin = ptx.find(reductionStart);
          size_t end = ptx.find(reductionEnd);
          EXPECT_NE(begin, llvm::StringRef::npos);
          EXPECT_NE(end, llvm::StringRef::npos);
          return ptx.slice(begin, end + reductionEnd.size());
        };
        EXPECT_EQ(reductionBody(code(devices.front(), fusedEntry)),
                  reductionBody(code(ordinaryDevice, ordinaryEntry)));
      }
      EXPECT_FALSE(devices.front().getCode().empty());
      EXPECT_EQ(llvm::range_size(devices.front()
                                     .getBody()
                                     .front()
                                     .getOps<mlir::shuttle::DeviceEntryOp>()),
                plan->tasks);
      EXPECT_EQ(
          llvm::range_size(invocationAbis.front()
                               .getBody()
                               .front()
                               .getOps<mlir::shuttle::InvocationSlotOp>()),
          plan->buffers);
      for (mlir::shuttle::DeviceEntryOp entry :
           devices.front()
               .getBody()
               .front()
               .getOps<mlir::shuttle::DeviceEntryOp>()) {
        EXPECT_GT(entry.getCodeLength(), 0);
      }

      mlir::FailureOr<llvm::SmallVector<uint8_t>> transport =
          mlir::shuttle::serializeGpuExecutableBundle(*built->module);
      ASSERT_TRUE(mlir::succeeded(transport));
      auto decoded = mlir::shuttle::GpuExecutable::Load(*transport);
      ASSERT_TRUE(decoded.ok()) << decoded.status();
      EXPECT_EQ((*decoded)->codeBytes().size(),
                devices.front().getCode().size());
      EXPECT_EQ((*decoded)->launches().size(),
                plan->tasks);
      EXPECT_EQ((*decoded)->slots().size(),
                plan->buffers);

      auto entries =
          llvm::to_vector(devices.front()
                              .getBody()
                              .front()
                              .getOps<mlir::shuttle::DeviceEntryOp>());
      for (auto [ordinal, entry, launch] :
           llvm::enumerate(entries, (*decoded)->launches())) {
        EXPECT_EQ(launch.taskOrdinal, ordinal);
        EXPECT_EQ(launch.codeOffset, entry.getCodeOffset());
        EXPECT_EQ(launch.codeLength, entry.getCodeLength());
        EXPECT_EQ(launch.codeDigest, entry.getCodeDigest());
        EXPECT_EQ(launch.inputSlots,
                  std::vector<int64_t>(entry.getInputBuffers()));
        EXPECT_EQ(launch.outputSlots,
                  std::vector<int64_t>(entry.getOutputBuffers()));
        EXPECT_EQ(launch.dependencies,
                  std::vector<int64_t>(entry.getDependencies()));
        ASSERT_TRUE(entry.getGrid());
        ASSERT_TRUE(entry.getBlock());
        EXPECT_EQ(launch.grid,
                  (std::array<uint64_t, 3>{
                      static_cast<uint64_t>((*entry.getGrid())[0]),
                      static_cast<uint64_t>((*entry.getGrid())[1]),
                      static_cast<uint64_t>((*entry.getGrid())[2])}));
        EXPECT_EQ(launch.block,
                  (std::array<uint64_t, 3>{
                      static_cast<uint64_t>((*entry.getBlock())[0]),
                      static_cast<uint64_t>((*entry.getBlock())[1]),
                      static_cast<uint64_t>((*entry.getBlock())[2])}));
        EXPECT_EQ(launch.dynamicSharedMemoryBytes,
                  static_cast<uint64_t>(*entry.getDynamicSharedBytes()));
      }

      auto slots =
          llvm::to_vector(invocationAbis.front()
                              .getBody()
                              .front()
                              .getOps<mlir::shuttle::InvocationSlotOp>());
      for (auto [ordinal, slot, decodedSlot] :
           llvm::enumerate(slots, (*decoded)->slots())) {
        EXPECT_EQ(decodedSlot.ordinal, ordinal);
        EXPECT_EQ(printType(decodedSlot.tensorType),
                  printType(slot.getTensorType()));
        EXPECT_EQ(decodedSlot.requiredBytes, slot.getRequiredBytes());
        EXPECT_EQ(decodedSlot.byteStrides,
                  llvm::SmallVector<int64_t>(slot.getStrides()));
        EXPECT_EQ(decodedSlot.byteOffset, slot.getOffset());
        EXPECT_EQ(decodedSlot.alignment, slot.getAlignment());
        EXPECT_EQ(decodedSlot.addressSpace, slot.getAddressSpace());
        EXPECT_EQ(decodedSlot.access, slot.getAccess());
        EXPECT_EQ(decodedSlot.storage, slot.getStorage());
        EXPECT_EQ(decodedSlot.aliasGroup, slot.getAliasGroup());
        EXPECT_EQ(decodedSlot.reuseGroup, slot.getReuseGroup());
        EXPECT_EQ(decodedSlot.bindingKind, slot.getBinding());
        EXPECT_EQ(decodedSlot.bindingIndex,
                  slot.getBindingIndex().value_or(-1));
      }
    }
  }
}

constexpr llvm::StringLiteral kFusedMapAlgebra = R"mlir(
#matrix = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
#feature = affine_map<(d0, d1) -> (d1)>
#scalar = affine_map<() -> ()>
module attributes {
  shuttle.coverage_manifest = {
    canonical_tuning = "{\22cluster_shape\22:[],\22materialization\22:\22prefer_fusion\22,\22maximum_candidates\22:1,\22pipeline_stages\22:1,\22tile_sizes\22:[]}"
  }
} {
  func.func @anonymous(%input: tensor<2x4xf32>, %row_input: tensor<2xf32>, %feature_input: tensor<4xf32>, %fold_input: tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2xf32>) {
    %result:2 = "shuttle.region"(%input, %row_input, %feature_input, %fold_input) ({
    ^bb0(%region_input: tensor<2x4xf32>, %region_row: tensor<2xf32>, %region_feature: tensor<4xf32>, %region_fold_input: tensor<2x4xf32>):
      %added = "shuttle.map"(%region_input, %region_row) ({
      ^bb0(%element: f32, %row: f32):
        %value = arith.addf %element, %row : f32
        "shuttle.yield"(%value) : (f32) -> ()
      }) {indexing_maps = [#matrix, #row, #matrix], source = #shuttle.source_ref<0, 0, 0, 0>} : (tensor<2x4xf32>, tensor<2xf32>) -> tensor<2x4xf32>
      %multiplied = "shuttle.map"(%added, %region_feature) ({
      ^bb0(%element: f32, %feature: f32):
        %value = arith.mulf %element, %feature : f32
        "shuttle.yield"(%value) : (f32) -> ()
      }) {indexing_maps = [#matrix, #feature, #matrix], source = #shuttle.source_ref<0, 0, 0, 1>} : (tensor<2x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
      %init = "shuttle.map"() ({
      ^bb0:
        %zero = arith.constant 0.000000e+00 : f32
        "shuttle.yield"(%zero) : (f32) -> ()
      }) {indexing_maps = [#scalar], source = #shuttle.source_ref<0, 0, 0, 2>} : () -> tensor<f32>
      %folded = "shuttle.fold"(%region_fold_input, %init) ({
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %element, %accumulator : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {accumulator_types = [f32], operandSegmentSizes = array<i32: 1, 1>, order_free = true, reduction_dimensions = array<i64: 1>, source = #shuttle.source_ref<0, 0, 0, 3>} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
      "shuttle.yield"(%multiplied, %folded) : (tensor<2x4xf32>, tensor<2xf32>) -> ()
    }) {policy = #shuttle.policy<source_ordered>, source_refs = [#shuttle.source_ref<0, 0, 0, 0>, #shuttle.source_ref<0, 0, 0, 1>, #shuttle.source_ref<0, 0, 0, 2>, #shuttle.source_ref<0, 0, 0, 3>]} : (tensor<2x4xf32>, tensor<2xf32>, tensor<4xf32>, tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2xf32>)
    return %result#0, %result#1 : tensor<2x4xf32>, tensor<2xf32>
  }
}
)mlir";

constexpr llvm::StringLiteral kFoldEpilogueAlgebra = R"mlir(
#fold_input = affine_map<(d0, d1) -> (d0, d1)>
#fold_result = affine_map<(d0, d1) -> (d0)>
#boundary = affine_map<(d0, d1) -> (d1, d0)>
#output = affine_map<(d0, d1) -> (d0, d1)>
#scalar = affine_map<() -> ()>
module attributes {
  shuttle.coverage_manifest = {
    canonical_tuning = "{\22cluster_shape\22:[],\22materialization\22:\22prefer_fusion\22,\22maximum_candidates\22:1,\22pipeline_stages\22:1,\22tile_sizes\22:[]}"
  }
} {
  func.func @anonymous(%input: tensor<2x4xf32>, %boundary_input: tensor<1x2xf32>, %other_boundary_input: tensor<2x1xf32>) -> tensor<2x1xf32> {
    %result = "shuttle.region"(%input, %boundary_input, %other_boundary_input) ({
    ^bb0(%region_input: tensor<2x4xf32>, %region_boundary: tensor<1x2xf32>, %region_other_boundary: tensor<2x1xf32>):
      %init = "shuttle.map"() ({
      ^bb0:
        %zero = arith.constant 0.000000e+00 : f32
        "shuttle.yield"(%zero) : (f32) -> ()
      }) {indexing_maps = [#scalar], source = #shuttle.source_ref<0, 0, 0, 0>} : () -> tensor<f32>
      %folded = "shuttle.fold"(%region_input, %init) ({
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %element, %accumulator : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {accumulator_types = [f32], operandSegmentSizes = array<i32: 1, 1>, order_free = true, reduction_dimensions = array<i64: 1>, source = #shuttle.source_ref<0, 0, 0, 1>} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
      %subtracted = "shuttle.map"(%folded, %region_boundary) ({
      ^bb0(%lhs: f32, %rhs: f32):
        %difference = arith.subf %lhs, %rhs : f32
        "shuttle.yield"(%difference) : (f32) -> ()
      }) {indexing_maps = [#fold_result, #boundary, #output], source = #shuttle.source_ref<0, 0, 0, 2>} : (tensor<2xf32>, tensor<1x2xf32>) -> tensor<2x1xf32>
      %divided = "shuttle.map"(%subtracted, %region_other_boundary) ({
      ^bb0(%lhs: f32, %rhs: f32):
        %quotient = arith.divf %lhs, %rhs : f32
        "shuttle.yield"(%quotient) : (f32) -> ()
      }) {indexing_maps = [#output, #output, #output], source = #shuttle.source_ref<0, 0, 0, 3>} : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
      "shuttle.yield"(%divided) : (tensor<2x1xf32>) -> ()
    }) {policy = #shuttle.policy<source_ordered>, source_refs = [#shuttle.source_ref<0, 0, 0, 0>, #shuttle.source_ref<0, 0, 0, 1>, #shuttle.source_ref<0, 0, 0, 2>, #shuttle.source_ref<0, 0, 0, 3>]} : (tensor<2x4xf32>, tensor<1x2xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
    return %result : tensor<2x1xf32>
  }
}
)mlir";

TEST(Abi22MapFusionContract,
     RejectsCorruptOrderedProvenanceAndComposedInputMaps) {
  constexpr llvm::StringLiteral mutations[] = {
      "duplicate-source", "reordered-source", "stale-source", "missing-map",
      "reordered-map"};
  for (llvm::StringRef mutation : mutations) {
    for (bool foldEpilogue : {false, true}) {
      SCOPED_TRACE(
          (llvm::Twine(foldEpilogue ? "Fold epilogue " : "Map ") + mutation)
              .str());
      std::unique_ptr<BuiltModule> built =
          foldEpilogue
              ? buildAlgebra(kFoldEpilogueAlgebra,
                             mlir::shuttle::NumericalPolicy::SourceOrdered,
                             false)
              : buildAlgebra(kFusedMapAlgebra,
                             mlir::shuttle::NumericalPolicy::SourceOrdered,
                             false);
      ASSERT_TRUE(built);
      ASSERT_TRUE(foldEpilogue
                      ? mutateFoldEpilogueTask(*built->module, mutation)
                      : mutateFusedTask(*built->module, mutation));
      EXPECT_FALSE(verifyMaterializationPlan(*built->module));
    }
  }
}

TEST(Abi22MapFusionContract,
     FusedPtxPreservesDependentScalarOrderAndFingerprintsEveryOperation) {
  std::optional<std::string> source =
      fixtureSource("abi22-map-fusion-source-order.mlir");
  ASSERT_TRUE(source);
  for (mlir::shuttle::NumericalPolicy policy :
       {mlir::shuttle::NumericalPolicy::SourceOrdered,
        mlir::shuttle::NumericalPolicy::Fast}) {
    std::unique_ptr<BuiltModule> built = buildAlgebra(*source, policy);
    ASSERT_TRUE(built);
    std::optional<PlanSummary> originalSummary = summary(*built->module);
    ASSERT_TRUE(originalSummary);
    EXPECT_EQ(originalSummary->tasks, 4);
    EXPECT_EQ(originalSummary->buffers, 6);
    EXPECT_EQ(originalSummary->fusedMapGroupSizes, std::vector<int64_t>({2}));
    llvm::SmallVector<mlir::shuttle::DeviceModuleOp> devices(
        built->module->getOps<mlir::shuttle::DeviceModuleOp>());
    ASSERT_EQ(devices.size(), 1);
    llvm::StringRef ptx(
        reinterpret_cast<const char *>(devices.front().getCode().data()),
        devices.front().getCode().size());
    size_t subtract = ptx.find("sub.rn.f32");
    size_t divide = ptx.find("div.rn.f32");
    ASSERT_NE(subtract, llvm::StringRef::npos);
    ASSERT_NE(divide, llvm::StringRef::npos);
    EXPECT_LT(subtract, divide);

    std::string changed = *source;
    size_t operation = changed.find("arith.subf");
    ASSERT_NE(operation, std::string::npos);
    changed.replace(operation, std::string("arith.subf").size(), "arith.addf");
    std::unique_ptr<BuiltModule> mutated = buildAlgebra(changed, policy);
    ASSERT_TRUE(mutated);
    std::optional<PlanSummary> mutatedSummary = summary(*mutated->module);
    ASSERT_TRUE(mutatedSummary);
    EXPECT_NE(originalSummary->fingerprint, mutatedSummary->fingerprint);
  }
}

TEST(Abi22MapFusionContract, RejectsPermutedFoldResultEdge) {
  constexpr llvm::StringLiteral source = R"mlir(
#input = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#transpose = affine_map<(d0, d1) -> (d1, d0)>
#output = affine_map<(d0, d1) -> (d0, d1)>
#scalar = affine_map<() -> ()>
module attributes {
  shuttle.coverage_manifest = {
    canonical_tuning = "{\22cluster_shape\22:[],\22materialization\22:\22prefer_fusion\22,\22maximum_candidates\22:1,\22pipeline_stages\22:1,\22tile_sizes\22:[]}"
  }
} {
  func.func @anonymous(%input: tensor<2x3x4xf32>) -> tensor<3x2xf32> {
    %result = "shuttle.region"(%input) ({
    ^bb0(%region_input: tensor<2x3x4xf32>):
      %init = "shuttle.map"() ({
      ^bb0:
        %zero = arith.constant 0.000000e+00 : f32
        "shuttle.yield"(%zero) : (f32) -> ()
      }) {indexing_maps = [#scalar], source = #shuttle.source_ref<0, 0, 0, 0>} : () -> tensor<f32>
      %folded = "shuttle.fold"(%region_input, %init) ({
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %element, %accumulator : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {accumulator_types = [f32], operandSegmentSizes = array<i32: 1, 1>, order_free = true, reduction_dimensions = array<i64: 2>, source = #shuttle.source_ref<0, 0, 0, 1>} : (tensor<2x3x4xf32>, tensor<f32>) -> tensor<2x3xf32>
      %transposed = "shuttle.map"(%folded) ({
      ^bb0(%element: f32):
        "shuttle.yield"(%element) : (f32) -> ()
      }) {indexing_maps = [#transpose, #output], source = #shuttle.source_ref<0, 0, 0, 2>} : (tensor<2x3xf32>) -> tensor<3x2xf32>
      "shuttle.yield"(%transposed) : (tensor<3x2xf32>) -> ()
    }) {policy = #shuttle.policy<source_ordered>, source_refs = [#shuttle.source_ref<0, 0, 0, 0>, #shuttle.source_ref<0, 0, 0, 1>, #shuttle.source_ref<0, 0, 0, 2>]} : (tensor<2x3x4xf32>) -> tensor<3x2xf32>
    return %result : tensor<3x2xf32>
  }
}
)mlir";
  std::unique_ptr<BuiltModule> built =
      buildAlgebra(source, mlir::shuttle::NumericalPolicy::SourceOrdered);
  ASSERT_TRUE(built);
  std::optional<PlanSummary> plan = summary(*built->module);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->tasks, 3);
  EXPECT_EQ(plan->buffers, 4);
  EXPECT_TRUE(plan->fusedFoldMapGroupSizes.empty());
}

TEST(Abi22MapFusionContract, MultiFoldMapDiamondRemainsAdmissible) {
  constexpr llvm::StringLiteral source = R"mlir(
#input = affine_map<(d0, d1) -> (d0, d1)>
#element = affine_map<(d0) -> (d0)>
#scalar = affine_map<() -> ()>
module attributes {
  shuttle.coverage_manifest = {
    canonical_tuning = "{\22cluster_shape\22:[],\22materialization\22:\22prefer_fusion\22,\22maximum_candidates\22:1,\22pipeline_stages\22:1,\22tile_sizes\22:[]}"
  }
} {
  func.func @anonymous(%input: tensor<2x4xf32>) -> tensor<2xf32> {
    %result = "shuttle.region"(%input) ({
    ^bb0(%region_input: tensor<2x4xf32>):
      %init = "shuttle.map"() ({
      ^bb0:
        %zero = arith.constant 0.000000e+00 : f32
        "shuttle.yield"(%zero) : (f32) -> ()
      }) {indexing_maps = [#scalar], source = #shuttle.source_ref<0, 0, 0, 0>} : () -> tensor<f32>
      %left_fold = "shuttle.fold"(%region_input, %init) ({
      ^bb0(%element_value: f32, %accumulator: f32):
        %sum = arith.addf %element_value, %accumulator : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {accumulator_types = [f32], operandSegmentSizes = array<i32: 1, 1>, order_free = true, reduction_dimensions = array<i64: 1>, source = #shuttle.source_ref<0, 0, 0, 1>} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
      %right_fold = "shuttle.fold"(%region_input, %init) ({
      ^bb0(%element_value: f32, %accumulator: f32):
        %sum = arith.addf %element_value, %accumulator : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {accumulator_types = [f32], operandSegmentSizes = array<i32: 1, 1>, order_free = true, reduction_dimensions = array<i64: 1>, source = #shuttle.source_ref<0, 0, 0, 2>} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
      %left = "shuttle.map"(%left_fold) ({
      ^bb0(%value: f32):
        %scaled = arith.mulf %value, %value : f32
        "shuttle.yield"(%scaled) : (f32) -> ()
      }) {indexing_maps = [#element, #element], source = #shuttle.source_ref<0, 0, 0, 3>} : (tensor<2xf32>) -> tensor<2xf32>
      %right = "shuttle.map"(%right_fold) ({
      ^bb0(%value: f32):
        %shifted = arith.addf %value, %value : f32
        "shuttle.yield"(%shifted) : (f32) -> ()
      }) {indexing_maps = [#element, #element], source = #shuttle.source_ref<0, 0, 0, 4>} : (tensor<2xf32>) -> tensor<2xf32>
      %joined = "shuttle.map"(%left, %right) ({
      ^bb0(%lhs: f32, %rhs: f32):
        %sum = arith.addf %lhs, %rhs : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {indexing_maps = [#element, #element, #element], source = #shuttle.source_ref<0, 0, 0, 5>} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
      "shuttle.yield"(%joined) : (tensor<2xf32>) -> ()
    }) {policy = #shuttle.policy<source_ordered>, source_refs = [#shuttle.source_ref<0, 0, 0, 0>, #shuttle.source_ref<0, 0, 0, 1>, #shuttle.source_ref<0, 0, 0, 2>, #shuttle.source_ref<0, 0, 0, 3>, #shuttle.source_ref<0, 0, 0, 4>, #shuttle.source_ref<0, 0, 0, 5>]} : (tensor<2x4xf32>) -> tensor<2xf32>
    return %result : tensor<2xf32>
  }
}
)mlir";
  std::unique_ptr<BuiltModule> built =
      buildAlgebra(source, mlir::shuttle::NumericalPolicy::SourceOrdered);
  ASSERT_TRUE(built);
  std::optional<PlanSummary> plan = summary(*built->module);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->tasks, 4);
  EXPECT_EQ(plan->buffers, 5);
  EXPECT_EQ(plan->fusedMapGroupSizes, std::vector<int64_t>({3}));
  EXPECT_TRUE(plan->fusedFoldMapGroupSizes.empty());
}

TEST(Abi22MapFusionContract,
     FoldEpiloguePtxBindsAffineInputAndStoresFinalScalarResult) {
  constexpr llvm::StringLiteral source = R"mlir(
#fold_input = affine_map<(d0, d1) -> (d0, d1)>
#fold_result = affine_map<(d0, d1) -> (d0)>
#boundary = affine_map<(d0, d1) -> (d1, d0)>
#output = affine_map<(d0, d1) -> (d0, d1)>
#scalar = affine_map<() -> ()>
module attributes {
  shuttle.coverage_manifest = {
    canonical_tuning = "{\22cluster_shape\22:[],\22materialization\22:\22prefer_fusion\22,\22maximum_candidates\22:1,\22pipeline_stages\22:1,\22tile_sizes\22:[]}"
  }
} {
  func.func @anonymous(%input: tensor<2x4xf32>, %boundary_input: tensor<1x2xf32>) -> tensor<2x1xf32> {
    %result = "shuttle.region"(%input, %boundary_input) ({
    ^bb0(%region_input: tensor<2x4xf32>, %region_boundary: tensor<1x2xf32>):
      %init = "shuttle.map"() ({
      ^bb0:
        %zero = arith.constant 0.000000e+00 : f32
        "shuttle.yield"(%zero) : (f32) -> ()
      }) {indexing_maps = [#scalar], source = #shuttle.source_ref<0, 0, 0, 0>} : () -> tensor<f32>
      %folded = "shuttle.fold"(%region_input, %init) ({
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %element, %accumulator : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {accumulator_types = [f32], operandSegmentSizes = array<i32: 1, 1>, order_free = true, reduction_dimensions = array<i64: 1>, source = #shuttle.source_ref<0, 0, 0, 1>} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
      %subtracted = "shuttle.map"(%folded, %region_boundary) ({
      ^bb0(%lhs: f32, %rhs: f32):
        %difference = arith.subf %lhs, %rhs : f32
        "shuttle.yield"(%difference) : (f32) -> ()
      }) {indexing_maps = [#fold_result, #boundary, #output], source = #shuttle.source_ref<0, 0, 0, 2>} : (tensor<2xf32>, tensor<1x2xf32>) -> tensor<2x1xf32>
      %divided = "shuttle.map"(%subtracted, %region_boundary) ({
      ^bb0(%lhs: f32, %rhs: f32):
        %quotient = arith.divf %lhs, %rhs : f32
        "shuttle.yield"(%quotient) : (f32) -> ()
      }) {indexing_maps = [#output, #boundary, #output], source = #shuttle.source_ref<0, 0, 0, 3>} : (tensor<2x1xf32>, tensor<1x2xf32>) -> tensor<2x1xf32>
      "shuttle.yield"(%divided) : (tensor<2x1xf32>) -> ()
    }) {policy = #shuttle.policy<source_ordered>, source_refs = [#shuttle.source_ref<0, 0, 0, 0>, #shuttle.source_ref<0, 0, 0, 1>, #shuttle.source_ref<0, 0, 0, 2>, #shuttle.source_ref<0, 0, 0, 3>]} : (tensor<2x4xf32>, tensor<1x2xf32>) -> tensor<2x1xf32>
    return %result : tensor<2x1xf32>
  }
}
)mlir";
  for (mlir::shuttle::NumericalPolicy policy :
       {mlir::shuttle::NumericalPolicy::SourceOrdered,
        mlir::shuttle::NumericalPolicy::Fast}) {
    std::unique_ptr<BuiltModule> built = buildAlgebra(source, policy);
    ASSERT_TRUE(built);
    auto device =
        *built->module->getOps<mlir::shuttle::DeviceModuleOp>().begin();
    auto entry = llvm::find_if(
        device.getBody().front().getOps<mlir::shuttle::DeviceEntryOp>(),
        [](auto candidate) {
          return candidate.getDynamicSharedBytes().value_or(0) != 0;
        });
    ASSERT_NE(
        entry,
        device.getBody().front().getOps<mlir::shuttle::DeviceEntryOp>().end());
    mlir::shuttle::DeviceEntryOp foldEntry = *entry;
    llvm::StringRef ptx(
        reinterpret_cast<const char *>(device.getCode().data() +
                                       foldEntry.getCodeOffset()),
        foldEntry.getCodeLength());
    constexpr llvm::StringLiteral indexedLoad =
        "  mov.u64 %rd10, 0;\n"
        "  mad.lo.u64 %rd10, %rd10, 1, %rd9;\n"
        "  mad.lo.u64 %rd10, %rd10, 2, %rd8;\n"
        "  ld.param.u64 %rd11, [p2];\n"
        "  mad.lo.u64 %rd11, %rd10, 4, %rd11;\n"
        "  ld.global.f32 %f38, [%rd11];\n";
    constexpr llvm::StringLiteral subtract = "  sub.rn.f32 %f39, %f37, %f38;\n";
    constexpr llvm::StringLiteral divide = "  div.rn.f32 %f40, %f39, %f38;\n";
    constexpr llvm::StringLiteral finalStore =
        "  ld.param.u64 %rd12, [p3];\n"
        "  mad.lo.u64 %rd12, %rd0, 4, %rd12;\n"
        "  st.global.f32 [%rd12], %f40;\n";
    size_t loadPosition = ptx.find(indexedLoad);
    size_t subtractPosition = ptx.find(subtract);
    size_t dividePosition = ptx.find(divide);
    size_t storePosition = ptx.find(finalStore);
    ASSERT_NE(loadPosition, llvm::StringRef::npos);
    ASSERT_NE(subtractPosition, llvm::StringRef::npos);
    ASSERT_NE(dividePosition, llvm::StringRef::npos);
    ASSERT_NE(storePosition, llvm::StringRef::npos);
    EXPECT_LT(loadPosition, subtractPosition);
    EXPECT_LT(subtractPosition, dividePosition);
    EXPECT_LT(dividePosition, storePosition);
    EXPECT_EQ(ptx.count("st.global.f32"), 1u);
  }
}

TEST(Abi22MapFusionContract,
     PreHardwarePtxShapeRejectsNearbyInstructionMutations) {
  constexpr llvm::StringLiteral correct = R"ptx(
  ld.param.u64 %rd0, [p0];
  ld.global.u16 %h0, [%rd0];
  cvt.f32.bf16 %f0, %h0;
  mul.rn.f32 %f1, %f0, %f0;
  st.shared.f32 [%rd1], %f1;
  ld.shared.f32 %f2, [%rd1];
  ld.shared.f32 %f3, [%rd1+4];
  add.rn.f32 %f4, %f2, %f3;
  st.shared.f32 [%rd1], %f4;
  ld.shared.f32 %f5, [%rd1];
  mov.b32 %f6, 0f41980000;
  div.rn.f32 %f7, %f5, %f6;
  mov.b32 %f8, 0f358637BD;
  add.rn.f32 %f9, %f7, %f8;
  sqrt.rn.f32 %f10, %f9;
  mov.b32 %f11, 0f3F800000;
  div.rn.f32 %f12, %f11, %f10;
  mul.rn.f32 %f13, %f0, %f12;
  cvt.rn.bf16.f32 %h1, %f13;
  ld.param.u64 %rd2, [p1];
  st.global.u16 [%rd2], %h1;
)ptx";
  auto replace = [](llvm::StringRef source, llvm::StringRef oldText,
                    llvm::StringRef newText) {
    std::string changed = source.str();
    size_t position = changed.find(oldText.str());
    EXPECT_NE(position, std::string::npos);
    if (position != std::string::npos)
      changed.replace(position, oldText.size(), newText.str());
    return changed;
  };
  PtxDataflow valid = ptxDataflow(correct);
  ASSERT_EQ(valid.stores.size(), 1u);
  EXPECT_TRUE(containsRmsNormReciprocalSubgraph(
      valid, valid.stores.front().value, "0f41980000", "$shared"));
  EXPECT_TRUE(containsStagedSharedAdd(valid));

  for (std::string mutation : {
           replace(correct, "div.rn.f32 %f7, %f5, %f6",
                   "mul.rn.f32 %f7, %f5, %f6"),
           replace(correct, "div.rn.f32 %f12, %f11, %f10",
                   "div.rn.f32 %f12, %f10, %f11"),
           replace(correct, "div.rn.f32 %f12, %f11, %f10",
                   "mov.f32 %f12, %f10"),
           replace(correct, "mul.rn.f32 %f13, %f0, %f12", "neg.f32 %f13, %f12"),
       }) {
    PtxDataflow flow = ptxDataflow(mutation);
    ASSERT_EQ(flow.stores.size(), 1u);
    EXPECT_FALSE(containsRmsNormReciprocalSubgraph(
        flow, flow.stores.front().value, "0f41980000", "$shared"));
  }
  PtxDataflow reassociated = ptxDataflow(
      replace(correct, "add.rn.f32 %f4, %f2, %f3", "sub.rn.f32 %f4, %f2, %f3"));
  EXPECT_FALSE(containsStagedSharedAdd(reassociated));

  constexpr llvm::StringLiteral correctFoldLoop = R"ptx(
  mov.u32 %r0, %tid.x;
  mov.u32 %r1, %ctaid.x;
  setp.ge.u32 %p0, %r1, 3;
  @%p0 bra $done;
  mul.lo.u32 %r4, %r0, 2;
  add.u32 %r7, %r4, 2;
  min.u32 %r7, %r7, 330;
  mov.f32 %f0, 0f00000000;
$load:
  setp.ge.u32 %p1, %r4, %r7;
  @%p1 bra $loaded;
  cvt.u64.u32 %rd0, %r4;
  ld.global.f32 %f1, [%rd0];
  add.rn.f32 %f2, %f0, %f1;
  mov.f32 %f0, %f2;
  add.u32 %r4, %r4, 1;
  bra $load;
$loaded:
$done:
)ptx";
  constexpr llvm::StringLiteral renamedFoldLoop = R"ptx(
  mov.u32 %r31, %tid.x;
  mov.u32 %r29, %ctaid.x;
  setp.ge.u32 %p3, %r29, 3;
  @%p3 bra $all_done;
  mul.lo.u32 %r18, %r31, 2;
  add.u32 %r22, %r18, 2;
  min.u32 %r22, %r22, 330;
  mov.f32 %f9, 0f00000000;
$again:
  setp.ge.u32 %p7, %r18, %r22;
  @%p7 bra $finished;
  cvt.u64.u32 %rd6, %r18;
  ld.global.f32 %f12, [%rd6];
  add.rn.f32 %f13, %f9, %f12;
  mov.f32 %f9, %f13;
  add.u32 %r18, %r18, 1;
  bra $again;
$finished:
$all_done:
)ptx";
  EXPECT_TRUE(hasDisjointContiguousFoldLoop(correctFoldLoop, 3, 330, 2));
  EXPECT_TRUE(hasDisjointContiguousFoldLoop(renamedFoldLoop, 3, 330, 2));
  for (std::string mutation : {
           replace(correctFoldLoop, "@%p1 bra $loaded;", "nop;"),
           replace(correctFoldLoop, "add.u32 %r4, %r4, 1;",
                   "add.u32 %r4, %r4, 2;"),
           replace(correctFoldLoop, "bra $load;", "bra $loaded;"),
           replace(correctFoldLoop, "mov.u32 %r0, %tid.x;",
                   "mov.u32 %r0, %ctaid.x;"),
           replace(correctFoldLoop, "mov.u32 %r0, %tid.x;",
                   "mov.u32 %r0, %tid.y;"),
           replace(replace(replace(correctFoldLoop, "add.u32 %r7, %r4, 2;",
                                   "add.u32 %r4, %r4, 2;"),
                           "min.u32 %r7, %r7, 330;", "min.u32 %r4, %r4, 330;"),
                   "setp.ge.u32 %p1, %r4, %r7;", "setp.ge.u32 %p1, %r4, %r4;"),
           replace(correctFoldLoop, "ld.global.f32 %f1, [%rd0];",
                   "bra $loaded;\n  ld.global.f32 %f1, [%rd0];"),
           replace(correctFoldLoop, "$load:", "mov.u32 %r7, 0;\n$load:"),
           replace(correctFoldLoop, "ld.global.f32 %f1, [%rd0];",
                   "@%p2 mov.u32 %r7, 0;\n  ld.global.f32 %f1, [%rd0];"),
           replace(correctFoldLoop, "ld.global.f32 %f1, [%rd0];",
                   "ret;\n  ld.global.f32 %f1, [%rd0];"),
           replace(correctFoldLoop, "ld.global.f32 %f1, [%rd0];", "nop;"),
           replace(correctFoldLoop, "add.rn.f32 %f2, %f0, %f1;", "nop;"),
           replace(correctFoldLoop, "mov.f32 %f0, 0f00000000;", "ret;"),
           replace(correctFoldLoop, "mov.f32 %f0, 0f00000000;", "@%p2 ret;"),
           replace(correctFoldLoop, "mov.f32 %f0, 0f00000000;", "bra $loaded;"),
           replace(correctFoldLoop, "mov.f32 %f0, 0f00000000;", "exit;"),
           replace(correctFoldLoop, "cvt.u64.u32 %rd0, %r4;",
                   "cvt.u64.u32 %rd0, %r7;"),
           replace(correctFoldLoop, "cvt.u64.u32 %rd0, %r4;",
                   "@%p1 cvt.u64.u32 %rd0, %r4;"),
           replace(correctFoldLoop, "add.rn.f32 %f2, %f0, %f1;",
                   "@%p1 mov.f32 %f3, %f1;\n  add.rn.f32 %f2, %f0, %f3;"),
           replace(correctFoldLoop, "add.rn.f32 %f2, %f0, %f1;",
                   "mul.rn.f32 %f1, %f1, %f0;\n  add.rn.f32 %f2, %f0, %f1;"),
           replace(correctFoldLoop, "add.u32 %r4, %r4, 1;",
                   "add.rn.f32 %f3, %f0, %f1;\n  mov.f32 %f0, %f3;\n  "
                   "add.u32 %r4, %r4, 1;"),
           replace(correctFoldLoop, "mov.f32 %f0, 0f00000000;",
                   "mov.f32 %f0, 0f00000000;\n  mov.f32 %f0, 0f3F800000;"),
           replace(replace(correctFoldLoop, "cvt.u64.u32 %rd0, %r4;", "nop;"),
                   "mov.f32 %f0, 0f00000000;",
                   "cvt.u64.u32 %rd0, %r4;\n  mov.f32 %f0, "
                   "0f00000000;"),
           replace(correctFoldLoop, "@%p0 bra $done;", "bra $done;"),
           replace(correctFoldLoop, "setp.ge.u32 %p0, %r1, 3;",
                   "setp.lt.u32 %p0, %r1, 3;"),
           replace(correctFoldLoop, "setp.ge.u32 %p0, %r1, 3;",
                   "setp.ge.u32 %p0, %r1, 0;"),
           replace(correctFoldLoop, "@%p0 bra $done;", "@%p0 bra $loaded;"),
           replace(replace(correctFoldLoop, "mov.u32 %r1, %ctaid.x;",
                           "mov.u32 %r0, %ctaid.x;"),
                   "setp.ge.u32 %p0, %r1, 3;", "setp.ge.u32 %p0, %r0, 3;"),
           replace(correctFoldLoop, "mul.lo.u32 %r4, %r0, 2;",
                   "bra $loaded;\n  mul.lo.u32 %r4, %r0, 2;"),
           replace(correctFoldLoop, "mul.lo.u32 %r4, %r0, 2;",
                   "@!%p0 bra $loaded;\n  mul.lo.u32 %r4, %r0, 2;"),
           replace(correctFoldLoop, "mul.lo.u32 %r4, %r0, 2;",
                   "ret;\n  mul.lo.u32 %r4, %r0, 2;"),
           replace(correctFoldLoop, "mul.lo.u32 %r4, %r0, 2;",
                   "$skip:\n  mul.lo.u32 %r4, %r0, 2;"),
           replace(correctFoldLoop, "add.rn.f32 %f2, %f0, %f1;",
                   "mov.f32 %f1, 0f00000000;\n  add.rn.f32 %f2, %f0, %f1;"),
       })
    EXPECT_FALSE(hasDisjointContiguousFoldLoop(mutation, 3, 330, 2));
}

TEST(Abi22MapFusionContract,
     FusedNormFoldRegionsBindCompactBoundariesAndCtaLocalPhases) {
  struct Boundary {
    const char *filename;
    int64_t tasks;
    int64_t buffers;
    int64_t phases;
    int64_t temporaryBuffers;
  };
  constexpr Boundary boundaries[] = {
      {"jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-forward.mlir", 1,
       3, 1, 0},
      {"jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-backward.mlir", 3,
       6, 5, 1},
      {"jax-0.10.1-bf16-projected_parameter_broadcast-"
       "3x2x5x11x263-forward.mlir",
       1, 3, 1, 0},
      {"jax-0.10.1-bf16-projected_parameter_broadcast-"
       "3x2x5x11x263-backward.mlir",
       3, 6, 5, 1},
  };
  for (const Boundary &boundary : boundaries) {
    SCOPED_TRACE(boundary.filename);
    std::optional<std::string> source = fixtureSource(boundary.filename);
    ASSERT_TRUE(source);
    std::optional<std::string> sourceOrderedFingerprint;
    for (mlir::shuttle::NumericalPolicy policy :
         {mlir::shuttle::NumericalPolicy::SourceOrdered,
          mlir::shuttle::NumericalPolicy::Fast}) {
      std::unique_ptr<BuiltModule> built =
          build(*source, policy, kFusionTuning, false, true);
      ASSERT_TRUE(built);
      auto plan = *built->module->getOps<mlir::shuttle::MaterializationPlanOp>()
                       .begin();
      EXPECT_EQ(plan.getSchemaVersion(), 2);
      auto tasks = llvm::to_vector(
          plan.getBody()
              .front()
              .getOps<mlir::shuttle::MaterializationTaskOp>());
      auto buffers = llvm::to_vector(
          plan.getBody()
              .front()
              .getOps<mlir::shuttle::MaterializationBufferOp>());
      ASSERT_EQ(tasks.size(), boundary.tasks);
      ASSERT_EQ(buffers.size(), boundary.buffers);
      EXPECT_EQ(llvm::count_if(
                    buffers,
                    [](auto buffer) {
                      return buffer.getStorage() ==
                             mlir::shuttle::MaterializationStorage::Temporary;
                    }),
                boundary.temporaryBuffers);
      const bool backward =
          llvm::StringRef(boundary.filename).contains("backward");
      if (backward) {
        auto temporary = llvm::find_if(buffers, [](auto buffer) {
          return buffer.getStorage() ==
                 mlir::shuttle::MaterializationStorage::Temporary;
        });
        ASSERT_NE(temporary, buffers.end());
        auto type =
            mlir::cast<mlir::RankedTensorType>(temporary->getTensorType());
        llvm::SmallVector<int64_t> expectedShape =
            llvm::StringRef(boundary.filename).contains("3x2x5x11x263")
                ? llvm::SmallVector<int64_t>({3, 2, 5, 11})
                : llvm::SmallVector<int64_t>({2, 7});
        EXPECT_EQ(type.getShape(), llvm::ArrayRef<int64_t>(expectedShape));
        EXPECT_TRUE(type.getElementType().isF32());
        EXPECT_EQ(temporary->getProducer(), 0);
        EXPECT_EQ(temporary->getConsumers(), llvm::ArrayRef<int64_t>({1, 2}));
        EXPECT_EQ(temporary->getLifetimeStart(), 0);
        EXPECT_EQ(temporary->getLifetimeEnd(), 2);
      }

      llvm::SmallDenseMap<mlir::Attribute, mlir::Operation *> algebraOperations;
      built->module->walk([&](mlir::Operation *operation) {
        if (auto map = mlir::dyn_cast<mlir::shuttle::MapOp>(operation))
          algebraOperations.try_emplace(map.getSource(), operation);
        else if (auto fold = mlir::dyn_cast<mlir::shuttle::FoldOp>(operation))
          algebraOperations.try_emplace(fold.getSource(), operation);
      });
      llvm::SmallDenseMap<int64_t, mlir::shuttle::MaterializationBufferOp>
          bufferByOrdinal;
      for (auto buffer : buffers)
        bufferByOrdinal.try_emplace(buffer.getOrdinal(), buffer);

      llvm::SmallDenseMap<mlir::Attribute, int64_t> sourceOccurrences;
      llvm::SmallDenseMap<mlir::Attribute, int64_t> recomputationDeclarations;
      int64_t phaseCount = 0;
      for (auto [taskOrdinal, task] : llvm::enumerate(tasks)) {
        EXPECT_EQ(task.getKind(), mlir::shuttle::MaterializationTaskKind::Fold);
        auto sources =
            task->getAttrOfType<mlir::ArrayAttr>("fold_region_sources");
        auto phases =
            task->getAttrOfType<mlir::ArrayAttr>("fold_region_phases");
        auto boundaryUses =
            task->getAttrOfType<mlir::ArrayAttr>("fold_region_boundary_uses");
        ASSERT_TRUE(sources);
        ASSERT_TRUE(phases);
        ASSERT_TRUE(boundaryUses);
        ASSERT_FALSE(sources.empty());
        ASSERT_FALSE(phases.empty());
        ASSERT_FALSE(boundaryUses.empty());
        phaseCount += phases.size();
        llvm::SmallDenseSet<mlir::Attribute> taskSources;
        llvm::SmallVector<int64_t> taskSourceOrdinals;
        std::optional<mlir::shuttle::SourceRefAttr> previousSource;
        for (mlir::Attribute sourceRef : sources) {
          auto source = mlir::dyn_cast<mlir::shuttle::SourceRefAttr>(sourceRef);
          ASSERT_TRUE(source);
          EXPECT_TRUE(algebraOperations.contains(sourceRef));
          EXPECT_TRUE(taskSources.insert(sourceRef).second);
          if (previousSource)
            EXPECT_TRUE(sourcePrecedes(*previousSource, source));
          previousSource = source;
          taskSourceOrdinals.push_back(source.getOperationOrdinal());
          ++sourceOccurrences[sourceRef];
        }
        if (!backward) {
          EXPECT_EQ(taskSourceOrdinals,
                    llvm::to_vector(llvm::seq<int64_t>(0, 19)));
          EXPECT_EQ(task.getInputBuffers(), llvm::ArrayRef<int64_t>({0, 1}));
          EXPECT_EQ(task.getOutputBuffers(), llvm::ArrayRef<int64_t>({2}));
          EXPECT_TRUE(task.getDependencies().empty());
        } else {
          const std::array<llvm::SmallVector<int64_t>, 3> expectedSources = {
              llvm::SmallVector<int64_t>({0, 1, 2, 3}),
              llvm::SmallVector<int64_t>({0, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17,
                                          20, 21, 22, 23, 24, 27, 28, 29}),
              llvm::SmallVector<int64_t>({0,  4,  5,  6,  7,  8,  9,  10, 11,
                                          12, 13, 14, 15, 18, 19, 20, 25, 26,
                                          30, 31, 32, 33, 34, 35, 36, 37, 38,
                                          39, 40, 41, 42, 43, 44, 45, 46, 47})};
          const std::array<llvm::SmallVector<int64_t>, 3> expectedInputs = {
              llvm::SmallVector<int64_t>({0}),
              llvm::SmallVector<int64_t>({0, 3, 2}),
              llvm::SmallVector<int64_t>({0, 3, 1, 2})};
          const std::array<llvm::SmallVector<int64_t>, 3> expectedOutputs = {
              llvm::SmallVector<int64_t>({3}), llvm::SmallVector<int64_t>({4}),
              llvm::SmallVector<int64_t>({5})};
          const std::array<llvm::SmallVector<int64_t>, 3> expectedDependencies =
              {llvm::SmallVector<int64_t>(), llvm::SmallVector<int64_t>({0}),
               llvm::SmallVector<int64_t>({0})};
          ASSERT_LT(taskOrdinal, expectedSources.size());
          EXPECT_EQ(taskSourceOrdinals, expectedSources[taskOrdinal]);
          EXPECT_EQ(task.getInputBuffers(),
                    llvm::ArrayRef<int64_t>(expectedInputs[taskOrdinal]));
          EXPECT_EQ(task.getOutputBuffers(),
                    llvm::ArrayRef<int64_t>(expectedOutputs[taskOrdinal]));
          EXPECT_EQ(task.getDependencies(),
                    llvm::ArrayRef<int64_t>(expectedDependencies[taskOrdinal]));
        }
        llvm::SmallDenseSet<std::pair<mlir::Attribute, int64_t>>
            expectedBoundaryUses;
        for (mlir::Attribute sourceRef : taskSources) {
          auto map = mlir::dyn_cast_or_null<mlir::shuttle::MapOp>(
              algebraOperations.lookup(sourceRef));
          if (!map)
            continue;
          for (auto [operandOrdinal, operand] :
               llvm::enumerate(map.getInputs())) {
            mlir::Operation *producer = operand.getDefiningOp();
            mlir::Attribute producerSource;
            if (auto producerMap =
                    mlir::dyn_cast_or_null<mlir::shuttle::MapOp>(producer))
              producerSource = producerMap.getSource();
            else if (auto producerFold =
                         mlir::dyn_cast_or_null<mlir::shuttle::FoldOp>(
                             producer))
              producerSource = producerFold.getSource();
            if (!producerSource || !taskSources.contains(producerSource))
              expectedBoundaryUses.insert(
                  {sourceRef, static_cast<int64_t>(operandOrdinal)});
          }
        }
        std::optional<mlir::shuttle::FoldOp> previousFold;
        llvm::SmallDenseSet<mlir::Attribute> phaseSources;
        for (auto [phaseOrdinal, phaseAttribute] : llvm::enumerate(phases)) {
          auto phase = mlir::dyn_cast<mlir::DictionaryAttr>(phaseAttribute);
          ASSERT_TRUE(phase);
          auto phaseSource =
              mlir::dyn_cast_or_null<mlir::shuttle::SourceRefAttr>(
                  phase.get("source"));
          ASSERT_TRUE(phaseSource);
          auto fold = mlir::dyn_cast_or_null<mlir::shuttle::FoldOp>(
              algebraOperations.lookup(phaseSource));
          ASSERT_TRUE(fold);
          EXPECT_TRUE(taskSources.contains(phaseSource));
          EXPECT_TRUE(phaseSources.insert(phaseSource).second);
          auto domain = mlir::dyn_cast_or_null<mlir::DenseI64ArrayAttr>(
              phase.get("domain_shape"));
          auto reduction = mlir::dyn_cast_or_null<mlir::DenseI64ArrayAttr>(
              phase.get("reduction_dimensions"));
          auto orderFree =
              mlir::dyn_cast_or_null<mlir::BoolAttr>(phase.get("order_free"));
          ASSERT_TRUE(domain);
          ASSERT_TRUE(reduction);
          ASSERT_TRUE(orderFree);
          auto foldInput = mlir::cast<mlir::RankedTensorType>(
              fold.getInputs().front().getType());
          EXPECT_EQ(domain.asArrayRef(), foldInput.getShape());
          EXPECT_EQ(reduction.asArrayRef(), fold.getReductionDimensions());
          EXPECT_EQ(orderFree.getValue(), fold.getOrderFree());
          auto transition =
              mlir::dyn_cast_or_null<mlir::StringAttr>(phase.get("transition"));
          ASSERT_TRUE(transition);
          EXPECT_EQ(transition.getValue(),
                    phaseOrdinal == 0 ? "launch_boundary" : "cta_local");
          if (previousFold) {
            auto previousInput = mlir::cast<mlir::RankedTensorType>(
                previousFold->getInputs().front().getType());
            auto previousResult = mlir::cast<mlir::RankedTensorType>(
                previousFold->getResult(0).getType());
            auto result =
                mlir::cast<mlir::RankedTensorType>(fold.getResult(0).getType());
            EXPECT_EQ(previousResult, result);
            int64_t reductionCardinality = 1;
            for (int64_t axis : fold.getReductionDimensions())
              reductionCardinality *= foldInput.getDimSize(axis);
            EXPECT_EQ(reductionCardinality, 1);
            auto dependsOnPrevious = [&](mlir::Value value,
                                         auto &self) -> bool {
              if (value == previousFold->getResult(0))
                return true;
              auto map = value.getDefiningOp<mlir::shuttle::MapOp>();
              if (!map || !taskSources.contains(map.getSource()))
                return false;
              return llvm::any_of(map.getInputs(), [&](mlir::Value input) {
                return self(input, self);
              });
            };
            EXPECT_TRUE(
                dependsOnPrevious(fold.getInputs().front(), dependsOnPrevious));
            (void)previousInput;
          }
          previousFold = fold;
        }
        for (mlir::Attribute sourceRef : taskSources) {
          if (mlir::isa_and_nonnull<mlir::shuttle::FoldOp>(
                  algebraOperations.lookup(sourceRef)))
            EXPECT_TRUE(phaseSources.contains(sourceRef));
        }
        llvm::SmallDenseSet<std::pair<mlir::Attribute, int64_t>>
            actualBoundaryUses;
        for (mlir::Attribute useAttribute : boundaryUses) {
          auto use = mlir::dyn_cast<mlir::DictionaryAttr>(useAttribute);
          ASSERT_TRUE(use);
          auto inputBuffer = mlir::dyn_cast_or_null<mlir::IntegerAttr>(
              use.get("input_buffer"));
          auto useSource = mlir::dyn_cast_or_null<mlir::shuttle::SourceRefAttr>(
              use.get("source"));
          auto operand =
              mlir::dyn_cast_or_null<mlir::IntegerAttr>(use.get("operand"));
          auto indexing = mlir::dyn_cast_or_null<mlir::AffineMapAttr>(
              use.get("indexing_map"));
          auto phase =
              mlir::dyn_cast_or_null<mlir::IntegerAttr>(use.get("phase"));
          ASSERT_TRUE(inputBuffer);
          ASSERT_TRUE(useSource);
          ASSERT_TRUE(operand);
          ASSERT_TRUE(indexing);
          ASSERT_TRUE(phase);
          EXPECT_TRUE(taskSources.contains(useSource));
          EXPECT_TRUE(
              actualBoundaryUses.insert({useSource, operand.getInt()}).second);
          EXPECT_TRUE(
              llvm::is_contained(task.getInputBuffers(), inputBuffer.getInt()));
          EXPECT_GE(phase.getInt(), 0);
          EXPECT_LT(phase.getInt(), phases.size());
          auto map = mlir::dyn_cast_or_null<mlir::shuttle::MapOp>(
              algebraOperations.lookup(useSource));
          ASSERT_TRUE(map);
          ASSERT_GE(operand.getInt(), 0);
          ASSERT_LT(operand.getInt(), map.getInputs().size());
          int64_t expectedPhase = 0;
          for (mlir::Attribute phaseAttribute : phases) {
            auto phaseSource = mlir::cast<mlir::shuttle::SourceRefAttr>(
                mlir::cast<mlir::DictionaryAttr>(phaseAttribute).get("source"));
            if (sourcePrecedes(phaseSource, useSource))
              ++expectedPhase;
          }
          expectedPhase = std::min<int64_t>(expectedPhase, phases.size() - 1);
          EXPECT_EQ(phase.getInt(), expectedPhase);
          EXPECT_EQ(indexing, mlir::cast<mlir::AffineMapAttr>(
                                  map.getIndexingMaps()[operand.getInt()]));
          auto buffer = bufferByOrdinal.lookup(inputBuffer.getInt());
          ASSERT_TRUE(buffer);
          EXPECT_EQ(buffer.getTensorType(),
                    map.getInputs()[operand.getInt()].getType());
          const int64_t sourceOrdinal = useSource.getOperationOrdinal();
          std::optional<int64_t> expectedBuffer;
          if (!backward) {
            if (sourceOrdinal == 0)
              expectedBuffer = 0;
            else if (sourceOrdinal == 14)
              expectedBuffer = 1;
          } else if (taskOrdinal == 0 && sourceOrdinal == 0) {
            expectedBuffer = 0;
          } else if (taskOrdinal == 1) {
            if (sourceOrdinal == 0)
              expectedBuffer = 0;
            else if (sourceOrdinal == 4)
              expectedBuffer = 3;
            else if (sourceOrdinal == 20)
              expectedBuffer = 2;
          } else if (taskOrdinal == 2) {
            if (sourceOrdinal == 0)
              expectedBuffer = 0;
            else if (sourceOrdinal == 4)
              expectedBuffer = 3;
            else if (sourceOrdinal == 18)
              expectedBuffer = 1;
            else if (sourceOrdinal == 20)
              expectedBuffer = 2;
          }
          ASSERT_TRUE(expectedBuffer);
          EXPECT_EQ(inputBuffer.getInt(), *expectedBuffer);
        }
        EXPECT_EQ(actualBoundaryUses.size(), expectedBoundaryUses.size());
        EXPECT_TRUE(llvm::all_of(expectedBoundaryUses, [&](auto boundaryUse) {
          return actualBoundaryUses.contains(boundaryUse);
        }));
        if (auto recomputed = task->getAttrOfType<mlir::ArrayAttr>(
                "fold_region_recomputed_sources")) {
          llvm::SmallDenseSet<mlir::Attribute> taskRecomputations;
          for (mlir::Attribute sourceRef : recomputed) {
            EXPECT_TRUE(taskSources.contains(sourceRef));
            EXPECT_TRUE(mlir::isa_and_nonnull<mlir::shuttle::MapOp>(
                algebraOperations.lookup(sourceRef)));
            EXPECT_TRUE(taskRecomputations.insert(sourceRef).second);
            ++recomputationDeclarations[sourceRef];
          }
        }
      }
      EXPECT_EQ(phaseCount, boundary.phases);
      EXPECT_EQ(sourceOccurrences.size(), algebraOperations.size());

      for (auto [sourceRef, occurrences] : sourceOccurrences) {
        EXPECT_EQ(recomputationDeclarations.lookup(sourceRef),
                  occurrences > 1 ? occurrences : 0);
      }
      if (llvm::StringRef(boundary.filename).contains("backward"))
        EXPECT_FALSE(recomputationDeclarations.empty());
      else
        EXPECT_TRUE(recomputationDeclarations.empty());

      auto device =
          *built->module->getOps<mlir::shuttle::DeviceModuleOp>().begin();
      auto entries = llvm::to_vector(
          device.getBody().front().getOps<mlir::shuttle::DeviceEntryOp>());
      ASSERT_EQ(entries.size(), boundary.tasks);
      for (auto [entryOrdinal, entry] : llvm::enumerate(entries)) {
        llvm::StringRef ptx(
            reinterpret_cast<const char *>(device.getCode().data() +
                                           entry.getCodeOffset()),
            entry.getCodeLength());
        auto phases = tasks[entryOrdinal]->getAttrOfType<mlir::ArrayAttr>(
            "fold_region_phases");
        ASSERT_TRUE(phases);
        auto firstPhase = mlir::cast<mlir::DictionaryAttr>(phases[0]);
        auto domain = mlir::cast<mlir::DenseI64ArrayAttr>(
            firstPhase.get("domain_shape"));
        auto reduction = mlir::cast<mlir::DenseI64ArrayAttr>(
            firstPhase.get("reduction_dimensions"));
        auto primaryFold = mlir::cast<mlir::shuttle::FoldOp>(
            algebraOperations.lookup(firstPhase.get("source")));
        const int64_t outputExtent =
            staticElementCount(mlir::cast<mlir::RankedTensorType>(
                primaryFold.getResult(0).getType()));
        int64_t reductionExtent = 1;
        for (int64_t axis : reduction.asArrayRef())
          reductionExtent *= domain.asArrayRef()[axis];
        const int64_t reductionTile =
            std::min<int64_t>(reductionExtent, 256);
        const int64_t workgroupThreads =
            ((reductionTile + 31) / 32) * 32;
        ASSERT_TRUE(entry.getBlock());
        EXPECT_EQ(*entry.getBlock(),
                  llvm::ArrayRef<int64_t>({workgroupThreads, 1, 1}));
        const int64_t serialTiles =
            (reductionExtent + reductionTile - 1) / reductionTile;
        EXPECT_TRUE(hasDisjointContiguousFoldLoop(
            ptx, outputExtent, reductionExtent, serialTiles));
        EXPECT_NE(ptx.find("ld.global.u16"), llvm::StringRef::npos);
        EXPECT_NE(ptx.find("cvt.f32.bf16"), llvm::StringRef::npos);
        EXPECT_NE(ptx.find("add.rn.f32"), llvm::StringRef::npos);
        PtxDataflow flow = ptxDataflow(ptx);
        ASSERT_EQ(flow.stores.size(), 1u);
        const bool bf16Output = !backward || entryOrdinal != 0;
        EXPECT_EQ(flow.stores.front().bf16, bf16Output);
        std::set<std::string> valueDependencies =
            ptxDependencyClosure(flow, flow.stores.front().value);
        std::set<std::string> pointerDependencies =
            ptxDependencyClosure(flow, flow.stores.front().pointer);
        EXPECT_TRUE(valueDependencies.count("$shared"));
        EXPECT_TRUE(valueDependencies.count("$op:add.rn.f32"));
        EXPECT_TRUE(valueDependencies.count("$op:mul.rn.f32"));
        EXPECT_TRUE(valueDependencies.count("$ld.global.u16"));
        for (int64_t parameter = 0;
             parameter <
             static_cast<int64_t>(tasks[entryOrdinal].getInputBuffers().size());
             ++parameter)
          EXPECT_TRUE(valueDependencies.count(
              (llvm::Twine("p") + llvm::Twine(parameter)).str()));
        EXPECT_TRUE(pointerDependencies.count(
            (llvm::Twine("p") +
             llvm::Twine(tasks[entryOrdinal].getInputBuffers().size()))
                .str()));
        EXPECT_TRUE(containsStagedSharedAdd(flow));
        if (backward && entryOrdinal == 0) {
          EXPECT_NE(ptx.find("mul.rn.f32"), llvm::StringRef::npos);
          EXPECT_NE(ptx.find("st.global.f32"), llvm::StringRef::npos);
          EXPECT_EQ(ptx.find("st.global.u16"), llvm::StringRef::npos);
        } else {
          const llvm::StringRef extent =
              llvm::StringRef(boundary.filename).contains("263") ? "0f43838000"
                                                                 : "0f41980000";
          EXPECT_NE(ptx.find(extent), llvm::StringRef::npos);
          EXPECT_NE(ptx.find("0f358637BD"), llvm::StringRef::npos);
          EXPECT_NE(ptx.find("sqrt.rn.f32"), llvm::StringRef::npos);
          EXPECT_NE(ptx.find("st.global.u16"), llvm::StringRef::npos);
          EXPECT_TRUE(valueDependencies.count("$op:sqrt.rn.f32"));
          EXPECT_TRUE(valueDependencies.count("$op:div.rn.f32"));
          EXPECT_TRUE(valueDependencies.count(extent.str()));
          EXPECT_TRUE(valueDependencies.count("0f358637BD"));
          EXPECT_TRUE(valueDependencies.count("$op:cvt.rn.bf16.f32"));
          EXPECT_TRUE(containsRmsNormReciprocalSubgraph(
              flow, flow.stores.front().value, extent,
              backward ? "p1" : "$shared"));
          EXPECT_FALSE(valueDependencies.count("$op:neg.f32"));
        }
      }
      if (!backward) {
        llvm::StringRef ptx(
            reinterpret_cast<const char *>(device.getCode().data() +
                                           entries.front().getCodeOffset()),
            entries.front().getCodeLength());
        size_t load = ptx.find("ld.global.u16");
        size_t square = ptx.find("mul.rn.f32");
        size_t reduce = ptx.find("st.shared.f32");
        size_t rsqrt = ptx.find("sqrt.rn.f32");
        size_t store = ptx.find("st.global.u16");
        ASSERT_NE(load, llvm::StringRef::npos);
        ASSERT_NE(square, llvm::StringRef::npos);
        ASSERT_NE(reduce, llvm::StringRef::npos);
        ASSERT_NE(rsqrt, llvm::StringRef::npos);
        ASSERT_NE(store, llvm::StringRef::npos);
        EXPECT_LT(load, square);
        EXPECT_LT(square, reduce);
        EXPECT_LT(reduce, rsqrt);
        EXPECT_LT(rsqrt, store);
        EXPECT_GE(ptx.count("ld.global.u16"), 2u);
        EXPECT_EQ(ptx.count("st.global.u16"), 1u);
      }

      if (sourceOrderedFingerprint) {
        EXPECT_NE(*sourceOrderedFingerprint, plan.getFingerprint());
      } else {
        sourceOrderedFingerprint = plan.getFingerprint().str();
      }
      std::unique_ptr<BuiltModule> renamed =
          build(*source, policy, kFusionTuning, true, true);
      ASSERT_TRUE(renamed);
      auto renamedPlan =
          *renamed->module->getOps<mlir::shuttle::MaterializationPlanOp>()
               .begin();
      EXPECT_EQ(plan.getFingerprint(), renamedPlan.getFingerprint());
    }
  }
}

TEST(Abi22MapFusionContract,
     ComposedFoldRegionRequiresExactRepeatedFoldProvenance) {
  std::optional<std::string> source = fixtureSource(kFixtures[2].filename);
  ASSERT_TRUE(source);
  std::unique_ptr<BuiltModule> built =
      build(*source, mlir::shuttle::NumericalPolicy::SourceOrdered,
            kFusionTuning, false, false);
  ASSERT_TRUE(built);

  llvm::SmallDenseSet<mlir::Attribute> foldSources;
  built->module->walk([&](mlir::shuttle::FoldOp fold) {
    foldSources.insert(fold.getSource());
  });
  auto plan =
      *built->module->getOps<mlir::shuttle::MaterializationPlanOp>().begin();
  bool removedRepeatedFold = false;
  for (auto task : plan.getBody().front().getOps<
           mlir::shuttle::MaterializationTaskOp>()) {
    auto recomputed = task->getAttrOfType<mlir::ArrayAttr>(
        "fold_region_recomputed_sources");
    if (!recomputed)
      continue;
    auto repeatedFold = llvm::find_if(recomputed, [&](mlir::Attribute source) {
      return foldSources.contains(source);
    });
    if (repeatedFold == recomputed.end())
      continue;
    llvm::SmallVector<mlir::Attribute> changed(recomputed.getValue());
    changed.erase(changed.begin() + std::distance(recomputed.begin(),
                                                 repeatedFold));
    task->setAttr("fold_region_recomputed_sources",
                  mlir::ArrayAttr::get(built->context.get(), changed));
    removedRepeatedFold = true;
    break;
  }
  ASSERT_TRUE(removedRepeatedFold);
  plan.setFingerprint(mlir::shuttle::materializationPlanFingerprint(plan));
  EXPECT_FALSE(verifyMaterializationPlan(*built->module));
}

TEST(Abi22MapFusionContract,
     RepresentativeRmsNormBindsExactForwardBackwardAndComposedResources) {
  struct Boundary {
    llvm::StringLiteral name;
    llvm::StringLiteral fixture;
    int64_t launches;
    int64_t slots;
    int64_t operands;
    int64_t results;
    int64_t temporaryAllocations;
    int64_t temporaryBytes;
    int64_t rowLaunches;
    int64_t columnLaunches;
  };
  constexpr Boundary boundaries[] = {
      {"forward",
       "jax-0.10.1-bf16-projected_parameter_broadcast-2048x4096-forward.mlir",
       1, 3, 2, 1, 0, 0, 1, 0},
      {"backward_recompute",
       "jax-0.10.1-bf16-projected_parameter_broadcast-2048x4096-backward.mlir",
       3, 6, 3, 2, 1, 8192, 2, 1},
      {"composed",
       "jax-0.10.1-bf16-projected_parameter_broadcast-2048x4096-composed.mlir",
       4, 7, 3, 3, 1, 8192, 3, 1},
  };
  for (const Boundary &boundary : boundaries) {
    SCOPED_TRACE(boundary.name.str());
    std::optional<std::string> source = fixtureSource(boundary.fixture);
    ASSERT_TRUE(source);
    for (mlir::shuttle::NumericalPolicy policy :
         {mlir::shuttle::NumericalPolicy::SourceOrdered,
          mlir::shuttle::NumericalPolicy::Fast}) {
      SCOPED_TRACE(policy == mlir::shuttle::NumericalPolicy::SourceOrdered
                       ? "source_ordered"
                       : "fast");
      std::unique_ptr<BuiltModule> built =
          build(*source, policy, kFusionTuning, false, true);
      ASSERT_TRUE(built);
      auto plan = *built->module->getOps<mlir::shuttle::MaterializationPlanOp>()
                       .begin();
      auto tasks = llvm::to_vector(
          plan.getBody()
              .front()
              .getOps<mlir::shuttle::MaterializationTaskOp>());
      auto buffers = llvm::to_vector(
          plan.getBody()
              .front()
              .getOps<mlir::shuttle::MaterializationBufferOp>());
      EXPECT_EQ(tasks.size(), boundary.launches);
      EXPECT_EQ(buffers.size(), boundary.slots);
      EXPECT_EQ(llvm::count_if(buffers, [](auto buffer) {
                  return buffer.getStorage() ==
                         mlir::shuttle::MaterializationStorage::Temporary;
                }),
                boundary.temporaryAllocations);

      auto serialized =
          mlir::shuttle::serializeGpuExecutableBundle(*built->module);
      ASSERT_TRUE(mlir::succeeded(serialized));
      auto executable = mlir::shuttle::GpuExecutable::Load(*serialized);
      ASSERT_TRUE(executable.ok()) << executable.status();
      EXPECT_EQ((*executable)->launches().size(), boundary.launches);
      EXPECT_EQ((*executable)->slots().size(), boundary.slots);
      struct ExpectedBinding {
        mlir::shuttle::ExecutableBindingKind kind;
        int64_t index;
        int64_t slot;
        int64_t bytes;
        llvm::SmallVector<int64_t> shape;
      };
      constexpr int64_t fullBytes = 2048LL * 4096 * sizeof(uint16_t);
      constexpr int64_t featureBytes = 4096LL * sizeof(uint16_t);
      llvm::SmallVector<ExpectedBinding> expectedBindings = {
          {mlir::shuttle::ExecutableBindingKind::Operand, 0, 0, fullBytes,
           {2048, 4096}},
          {mlir::shuttle::ExecutableBindingKind::Operand, 1, 1, featureBytes,
           {4096}},
      };
      if (boundary.name != "forward")
        expectedBindings.push_back(
            {mlir::shuttle::ExecutableBindingKind::Operand, 2, 2, fullBytes,
             {2048, 4096}});
      if (boundary.name == "forward") {
        expectedBindings.push_back(
            {mlir::shuttle::ExecutableBindingKind::Result, 0, 2, fullBytes,
             {2048, 4096}});
      } else if (boundary.name == "backward_recompute") {
        expectedBindings.append({
            {mlir::shuttle::ExecutableBindingKind::Result, 0, 5, fullBytes,
             {2048, 4096}},
            {mlir::shuttle::ExecutableBindingKind::Result, 1, 4,
             featureBytes, {4096}},
        });
      } else {
        expectedBindings.append({
            {mlir::shuttle::ExecutableBindingKind::Result, 0, 4, fullBytes,
             {2048, 4096}},
            {mlir::shuttle::ExecutableBindingKind::Result, 1, 6, fullBytes,
             {2048, 4096}},
            {mlir::shuttle::ExecutableBindingKind::Result, 2, 5,
             featureBytes, {4096}},
        });
      }
      ASSERT_EQ((*executable)->externalBindings().size(),
                expectedBindings.size());
      for (auto [binding, expected] : llvm::zip_equal(
               (*executable)->externalBindings(), expectedBindings)) {
        EXPECT_EQ(binding.kind, expected.kind);
        EXPECT_EQ(binding.index, expected.index);
        EXPECT_EQ(binding.slotOrdinal, expected.slot);
        EXPECT_EQ(binding.requiredBytes, expected.bytes);
        EXPECT_EQ(binding.alignment, 2);
        auto type = mlir::dyn_cast<mlir::RankedTensorType>(binding.tensorType);
        ASSERT_TRUE(type);
        EXPECT_EQ(type.getShape(), llvm::ArrayRef<int64_t>(expected.shape));
        EXPECT_TRUE(type.getElementType().isBF16());
      }
      EXPECT_EQ(llvm::count_if((*executable)->externalBindings(), [](auto binding) {
                  return binding.kind ==
                         mlir::shuttle::ExecutableBindingKind::Operand;
                }),
                boundary.operands);
      EXPECT_EQ(llvm::count_if((*executable)->externalBindings(), [](auto binding) {
                  return binding.kind ==
                         mlir::shuttle::ExecutableBindingKind::Result;
                }),
                boundary.results);
      int64_t temporaryAllocations = 0;
      int64_t temporaryBytes = 0;
      for (const mlir::shuttle::GpuSlot &slot : (*executable)->slots()) {
        if (slot.storage != mlir::shuttle::MaterializationStorage::Temporary)
          continue;
        ++temporaryAllocations;
        temporaryBytes += slot.requiredBytes;
        auto type = mlir::cast<mlir::RankedTensorType>(slot.tensorType);
        EXPECT_EQ(type.getShape(), llvm::ArrayRef<int64_t>({2048}));
        EXPECT_TRUE(type.getElementType().isF32());
      }
      EXPECT_EQ(temporaryAllocations, boundary.temporaryAllocations);
      EXPECT_EQ(temporaryBytes, boundary.temporaryBytes);

      int64_t rowLaunches = 0;
      int64_t columnLaunches = 0;
      for (const mlir::shuttle::GpuLaunch &launch :
           (*executable)->launches()) {
        EXPECT_EQ(launch.block, (std::array<uint64_t, 3>{256, 1, 1}));
        EXPECT_EQ(launch.dynamicSharedMemoryBytes, 1024u);
        EXPECT_FALSE(launch.codeDigest.empty());
        if (launch.grid == std::array<uint64_t, 3>{2048, 1, 1})
          ++rowLaunches;
        else if (launch.grid == std::array<uint64_t, 3>{4096, 1, 1})
          ++columnLaunches;
        else
          ADD_FAILURE() << "unexpected representative RMSNorm grid";
      }
      EXPECT_EQ(rowLaunches, boundary.rowLaunches);
      EXPECT_EQ(columnLaunches, boundary.columnLaunches);
    }
  }
}

TEST(Abi22MapFusionContract,
     RejectsCorruptFoldRegionProvenanceAffineUsesAndPhaseLocality) {
  std::optional<std::string> source = fixtureSource(
      "jax-0.10.1-bf16-projected_parameter_broadcast-2x7x19-backward.mlir");
  ASSERT_TRUE(source);
  constexpr llvm::StringLiteral mutations[] = {"boundary-buffer",
                                               "missing-boundary",
                                               "duplicate-boundary",
                                               "boundary-indexing-map",
                                               "nonlocal-phase",
                                               "missing-phase",
                                               "duplicate-phase",
                                               "reordered-phase",
                                               "phase-domain-shape",
                                               "phase-reduction-dimensions",
                                               "phase-order-free",
                                               "wrong-valid-phase",
                                               "dangling-recomputation",
                                               "duplicate-recomputation",
                                               "missing-recomputation",
                                               "reordered-source"};
  for (llvm::StringRef mutation : mutations) {
    for (mlir::shuttle::NumericalPolicy policy :
         {mlir::shuttle::NumericalPolicy::SourceOrdered,
          mlir::shuttle::NumericalPolicy::Fast}) {
      SCOPED_TRACE(mutation.str());
      SCOPED_TRACE(policy == mlir::shuttle::NumericalPolicy::SourceOrdered
                       ? "source_ordered"
                       : "fast");
      std::unique_ptr<BuiltModule> built =
          build(*source, policy, kFusionTuning, false, false);
      ASSERT_TRUE(built);
      ASSERT_TRUE(mutateFoldRegionTask(*built->module, mutation));
      EXPECT_FALSE(verifyMaterializationPlan(*built->module));
    }
  }
}

TEST(Abi22MapFusionContract,
     RankTwoAndLargeNormsFuseButLiveOutAndSecondFoldDoNot) {
  constexpr llvm::StringLiteral source = R"mlir(
module @anonymous {
  func.func public @main(%arg0: tensor<2x19xbf16>, %arg1: tensor<19xbf16>) -> tensor<2x19xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<2x19xbf16>) -> tensor<2x19xf32>
    %1 = stablehlo.multiply %0, %0 : tensor<2x19xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<2x19xf32>, tensor<f32>) -> tensor<2xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
    %cst_0 = stablehlo.constant dense<1.900000e+01> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
    %5 = stablehlo.divide %3, %4 : tensor<2x1xf32>
    %cst_1 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<2x1xf32>
    %7 = stablehlo.add %5, %6 : tensor<2x1xf32>
    %8 = stablehlo.rsqrt %7 : tensor<2x1xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x19xf32>
    %10 = stablehlo.multiply %0, %9 : tensor<2x19xf32>
    %11 = stablehlo.convert %arg1 : (tensor<19xbf16>) -> tensor<19xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [1] : (tensor<19xf32>) -> tensor<1x19xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<1x19xf32>) -> tensor<2x19xf32>
    %14 = stablehlo.multiply %10, %13 : tensor<2x19xf32>
    %15 = stablehlo.convert %14 : (tensor<2x19xf32>) -> tensor<2x19xbf16>
    return %15 : tensor<2x19xbf16>
  }
}
)mlir";
  auto replaceAll = [](std::string value, llvm::StringRef oldText,
                       llvm::StringRef newText) {
    size_t position = 0;
    while ((position = value.find(oldText.str(), position)) !=
           std::string::npos) {
      value.replace(position, oldText.size(), newText.str());
      position += newText.size();
    }
    return value;
  };
  for (std::string candidate :
       {source.str(), replaceAll(replaceAll(source.str(), "2x19", "2048x4096"),
                                 "1.900000e+01", "4.096000e+03")}) {
    if (candidate.find("2048x4096") != std::string::npos) {
      candidate = replaceAll(candidate, "tensor<19x", "tensor<4096x");
      candidate = replaceAll(candidate, "1x19", "1x4096");
      candidate = replaceAll(candidate, "tensor<2xf32>", "tensor<2048xf32>");
      candidate = replaceAll(candidate, "2x1", "2048x1");
    }
    for (mlir::shuttle::NumericalPolicy policy :
         {mlir::shuttle::NumericalPolicy::SourceOrdered,
          mlir::shuttle::NumericalPolicy::Fast}) {
      std::unique_ptr<BuiltModule> built =
          build(candidate, policy, kFusionTuning, false, true);
      ASSERT_TRUE(built);
      auto plan = *built->module
                       ->getOps<mlir::shuttle::MaterializationPlanOp>()
                       .begin();
      EXPECT_EQ(llvm::range_size(
                    plan.getBody()
                        .front()
                        .getOps<mlir::shuttle::MaterializationTaskOp>()),
                1);
      auto task = *plan.getBody()
                       .front()
                       .getOps<mlir::shuttle::MaterializationTaskOp>()
                       .begin();
      EXPECT_TRUE(task->getAttr("fold_region_sources"));
      auto device =
          *built->module->getOps<mlir::shuttle::DeviceModuleOp>().begin();
      auto entries = llvm::to_vector(
          device.getBody().front().getOps<mlir::shuttle::DeviceEntryOp>());
      ASSERT_EQ(entries.size(), 1u);
      ASSERT_TRUE(entries.front().getGrid());
      ASSERT_TRUE(entries.front().getBlock());
      const bool large = candidate.find("2048x4096") != std::string::npos;
      EXPECT_EQ(*entries.front().getGrid(),
                llvm::ArrayRef<int64_t>({large ? 2048 : 2, 1, 1}));
      EXPECT_LE(entries.front().getBlock()->front(), 256);
      llvm::StringRef ptx(
          reinterpret_cast<const char *>(device.getCode().data() +
                                         entries.front().getCodeOffset()),
          entries.front().getCodeLength());
      EXPECT_NE(ptx.find("ld.global.u16"), llvm::StringRef::npos);
      EXPECT_NE(ptx.find("st.global.u16"), llvm::StringRef::npos);
      EXPECT_NE(ptx.find(large ? "0f45800000" : "0f41980000"),
                llvm::StringRef::npos);
      EXPECT_NE(ptx.find(large ? "4096" : "19"), llvm::StringRef::npos);
    }
  }

  auto remainsUnfused = [&](std::string mutation) {
    for (mlir::shuttle::NumericalPolicy policy :
         {mlir::shuttle::NumericalPolicy::SourceOrdered,
          mlir::shuttle::NumericalPolicy::Fast}) {
      std::unique_ptr<BuiltModule> built =
          build(mutation, policy, kFusionTuning, false, false);
      ASSERT_TRUE(built);
      auto plan = *built->module
                       ->getOps<mlir::shuttle::MaterializationPlanOp>()
                       .begin();
      EXPECT_GT(llvm::range_size(
                    plan.getBody()
                        .front()
                        .getOps<mlir::shuttle::MaterializationTaskOp>()),
                1);
      for (auto task : plan.getBody().front().getOps<
               mlir::shuttle::MaterializationTaskOp>())
        EXPECT_FALSE(task->getAttr("fold_region_sources"));
    }
  };
  std::string liveOut =
      replaceAll(source.str(),
                 "func.func public @main(%arg0: tensor<2x19xbf16>, %arg1: "
                 "tensor<19xbf16>) -> tensor<2x19xbf16>",
                 "func.func public @main(%arg0: tensor<2x19xbf16>, %arg1: "
                 "tensor<19xbf16>) -> (tensor<2x19xbf16>, tensor<2xf32>)");
  liveOut = replaceAll(liveOut, "return %15 : tensor<2x19xbf16>",
                       "return %15, %2 : tensor<2x19xbf16>, tensor<2xf32>");
  remainsUnfused(std::move(liveOut));

  std::string secondFold = source.str();
  secondFold = replaceAll(
      secondFold, "    %3 = stablehlo.broadcast_in_dim",
      "    %extra = stablehlo.reduce(%1 init: %cst) applies stablehlo.add "
      "across dimensions = [1] : (tensor<2x19xf32>, tensor<f32>) -> "
      "tensor<2xf32>\n    %3 = stablehlo.broadcast_in_dim");
  secondFold = replaceAll(
      secondFold,
      "    %15 = stablehlo.convert %14 : (tensor<2x19xf32>) -> "
      "tensor<2x19xbf16>",
      "    %extra_1 = stablehlo.broadcast_in_dim %extra, dims = [0] : "
      "(tensor<2xf32>) -> tensor<2x1xf32>\n"
      "    %extra_2 = stablehlo.broadcast_in_dim %extra_1, dims = [0, 1] : "
      "(tensor<2x1xf32>) -> tensor<2x19xf32>\n"
      "    %joined = stablehlo.add %14, %extra_2 : tensor<2x19xf32>\n"
      "    %15 = stablehlo.convert %joined : (tensor<2x19xf32>) -> "
      "tensor<2x19xbf16>");
  remainsUnfused(std::move(secondFold));
}

std::string replaceAllText(std::string value, llvm::StringRef oldText,
                           llvm::StringRef newText) {
  size_t position = 0;
  while ((position = value.find(oldText.str(), position)) !=
         std::string::npos) {
    value.replace(position, oldText.size(), newText.str());
    position += newText.size();
  }
  return value;
}

llvm::SmallVector<int64_t> sourceOperationOrdinals(mlir::ArrayAttr sources) {
  llvm::SmallVector<int64_t> ordinals;
  for (mlir::Attribute attribute : sources) {
    auto source = mlir::dyn_cast<mlir::shuttle::SourceRefAttr>(attribute);
    if (!source)
      return {};
    ordinals.push_back(source.getOperationOrdinal());
  }
  return ordinals;
}

int64_t materializationBytes(mlir::shuttle::MaterializationBufferOp buffer) {
  auto type = mlir::cast<mlir::RankedTensorType>(buffer.getTensorType());
  int64_t bytes = staticElementCount(type);
  if (type.getElementType().isBF16())
    return bytes * 2;
  if (type.getElementType().isF32())
    return bytes * 4;
  return -1;
}

llvm::SmallVector<int64_t>
foldRegionPhaseOrdinals(mlir::shuttle::MaterializationTaskOp task) {
  auto phases = task->getAttrOfType<mlir::ArrayAttr>("fold_region_phases");
  if (!phases)
    return {};
  llvm::SmallVector<int64_t> ordinals;
  for (mlir::Attribute attribute : phases) {
    auto phase = mlir::dyn_cast<mlir::DictionaryAttr>(attribute);
    auto source = phase ? phase.getAs<mlir::shuttle::SourceRefAttr>("source")
                        : mlir::shuttle::SourceRefAttr{};
    if (!source)
      return {};
    ordinals.push_back(source.getOperationOrdinal());
  }
  return ordinals;
}

struct ExpectedFoldRegionBoundaryUse {
  int64_t source;
  int64_t operand;
  int64_t inputBuffer;
  mlir::AffineMap indexingMap;
  int64_t phase;
};

bool hasExactFoldRegionBoundaryUses(
    mlir::shuttle::MaterializationTaskOp task,
    llvm::ArrayRef<ExpectedFoldRegionBoundaryUse> expected) {
  auto uses =
      task->getAttrOfType<mlir::ArrayAttr>("fold_region_boundary_uses");
  if (!uses || uses.size() != expected.size())
    return false;
  for (auto [attribute, boundaryUse] : llvm::zip(uses, expected)) {
    auto use = mlir::dyn_cast<mlir::DictionaryAttr>(attribute);
    auto source =
        use ? use.getAs<mlir::shuttle::SourceRefAttr>("source")
            : mlir::shuttle::SourceRefAttr{};
    auto operand = use ? use.getAs<mlir::IntegerAttr>("operand")
                       : mlir::IntegerAttr{};
    auto inputBuffer = use ? use.getAs<mlir::IntegerAttr>("input_buffer")
                           : mlir::IntegerAttr{};
    auto indexingMap = use ? use.getAs<mlir::AffineMapAttr>("indexing_map")
                           : mlir::AffineMapAttr{};
    auto phase = use ? use.getAs<mlir::IntegerAttr>("phase")
                     : mlir::IntegerAttr{};
    if (!use || use.size() != 5 || !source || !operand || !inputBuffer ||
        !indexingMap || !phase || source.getFunctionOrdinal() != 0 ||
        source.getBlockOrdinal() != 0 ||
        source.getOperationOrdinal() != boundaryUse.source ||
        source.getResultOrdinal() != 0 ||
        operand.getInt() != boundaryUse.operand ||
        inputBuffer.getInt() != boundaryUse.inputBuffer ||
        indexingMap.getValue() != boundaryUse.indexingMap ||
        phase.getInt() != boundaryUse.phase)
      return false;
  }
  return true;
}

bool hasOrdinaryJaxLayerNormBCPartition(mlir::ModuleOp module, int64_t rows,
                                        int64_t features, bool composed) {
  auto plans = llvm::to_vector(
      module.getOps<mlir::shuttle::MaterializationPlanOp>());
  if (plans.size() != 1)
    return false;
  auto tasks = llvm::to_vector(
      plans.front()
          .getBody()
          .front()
          .getOps<mlir::shuttle::MaterializationTaskOp>());
  auto buffers = llvm::to_vector(
      plans.front()
          .getBody()
          .front()
          .getOps<mlir::shuttle::MaterializationBufferOp>());
  const int64_t inputCount = composed ? 4 : 3;
  if (tasks.size() != 2 ||
      buffers.size() !=
          static_cast<size_t>(inputCount + (composed ? 6 : 5)))
    return false;

  llvm::SmallVector<int64_t> rowSources;
  llvm::SmallVector<int64_t> featureSources;
  auto appendSequence = [](llvm::SmallVectorImpl<int64_t> &values,
                           int64_t begin, int64_t end) {
    llvm::append_range(values, llvm::seq<int64_t>(begin, end));
  };
  if (composed) {
    appendSequence(rowSources, 0, 36);
    rowSources.push_back(46);
    rowSources.push_back(47);
    appendSequence(rowSources, 51, 80);
    featureSources = {0, 7, 8, 24, 25, 35, 36, 37, 38, 39,
                      40, 41, 42, 43, 44, 45, 48, 49, 50};
  } else {
    appendSequence(rowSources, 0, 24);
    appendSequence(rowSources, 26, 29);
    rowSources.push_back(39);
    rowSources.push_back(40);
    appendSequence(rowSources, 44, 73);
    featureSources = {0,  7,  8,  24, 25, 28, 29, 30, 31, 32,
                      33, 34, 35, 36, 37, 38, 41, 42, 43};
  }
  const llvm::SmallVector<int64_t> rowOutputs =
      composed ? llvm::SmallVector<int64_t>{6, 19, 34, 79}
               : llvm::SmallVector<int64_t>{6, 19, 72};
  const llvm::SmallVector<int64_t> featureOutputs =
      composed ? llvm::SmallVector<int64_t>{41, 50}
               : llvm::SmallVector<int64_t>{34, 43};
  const llvm::SmallVector<int64_t> rowPhases =
      composed ? llvm::SmallVector<int64_t>{2, 11, 53, 62, 70, 76}
               : llvm::SmallVector<int64_t>{2, 11, 46, 55, 63, 69};
  const llvm::SmallVector<int64_t> featurePhases =
      composed ? llvm::SmallVector<int64_t>{37, 40, 44, 49}
               : llvm::SmallVector<int64_t>{30, 33, 37, 42};
  const llvm::SmallVector<int64_t> recomputed =
      composed ? llvm::SmallVector<int64_t>{0, 7, 8, 24, 25, 35}
               : llvm::SmallVector<int64_t>{0, 7, 8, 28};
  auto sources = [&](int64_t task, llvm::StringRef name) {
    return tasks[task]->getAttrOfType<mlir::ArrayAttr>(name);
  };
  auto rowRegionSources = sources(0, "fold_region_sources");
  auto featureRegionSources = sources(1, "fold_region_sources");
  auto rowRegionOutputs = sources(0, "fold_region_output_sources");
  auto featureRegionOutputs = sources(1, "fold_region_output_sources");
  auto rowRecomputed = sources(0, "fold_region_recomputed_sources");
  auto featureRecomputed = sources(1, "fold_region_recomputed_sources");
  if (!rowRegionSources || !featureRegionSources || !rowRegionOutputs ||
      !featureRegionOutputs || !rowRecomputed || !featureRecomputed ||
      sourceOperationOrdinals(rowRegionSources) != rowSources ||
      sourceOperationOrdinals(featureRegionSources) != featureSources ||
      sourceOperationOrdinals(rowRegionOutputs) != rowOutputs ||
      sourceOperationOrdinals(featureRegionOutputs) != featureOutputs ||
      sourceOperationOrdinals(rowRecomputed) != recomputed ||
      sourceOperationOrdinals(featureRecomputed) != recomputed ||
      foldRegionPhaseOrdinals(tasks[0]) != rowPhases ||
      foldRegionPhaseOrdinals(tasks[1]) != featurePhases)
    return false;

  mlir::MLIRContext *context = module.getContext();
  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, context);
  mlir::AffineExpr d1 = mlir::getAffineDimExpr(1, context);
  mlir::AffineMap matrix =
      mlir::AffineMap::getMultiDimIdentityMap(2, context);
  mlir::AffineMap feature =
      mlir::AffineMap::get(1, 0, {d0}, context);
  mlir::AffineMap projectedRow =
      mlir::AffineMap::get(2, 0, {d0, d1.floorDiv(features)}, context);
  llvm::SmallVector<ExpectedFoldRegionBoundaryUse> rowBoundaryUses = {
      {0, 0, 0, matrix, 0},
      {26, 0, 1, feature, 2},
  };
  if (composed)
    rowBoundaryUses.push_back({30, 0, 2, feature, 2});
  rowBoundaryUses.push_back(
      {composed ? 35 : 28, 0, composed ? 3 : 2, matrix, 2});
  const int64_t meanBuffer = inputCount;
  const int64_t inverseBuffer = inputCount + 1;
  llvm::SmallVector<ExpectedFoldRegionBoundaryUse> featureBoundaryUses = {
      {0, 0, 0, matrix, 0},
      {7, 0, meanBuffer, projectedRow, 0},
      {24, 0, inverseBuffer, projectedRow, 0},
      {composed ? 35 : 28, 0, composed ? 3 : 2, matrix, 0},
  };
  const llvm::SmallVector<int64_t> expectedRowInputs =
      composed ? llvm::SmallVector<int64_t>{0, 1, 2, 3}
               : llvm::SmallVector<int64_t>{0, 1, 2};
  const llvm::SmallVector<int64_t> expectedFeatureInputs =
      composed ? llvm::SmallVector<int64_t>{0, 3, 4, 5}
               : llvm::SmallVector<int64_t>{0, 2, 3, 4};
  const llvm::SmallVector<int64_t> expectedRowBuffers =
      composed ? llvm::SmallVector<int64_t>{4, 5, 6, 9}
               : llvm::SmallVector<int64_t>{3, 4, 7};
  const llvm::SmallVector<int64_t> expectedFeatureBuffers =
      composed ? llvm::SmallVector<int64_t>{7, 8}
               : llvm::SmallVector<int64_t>{5, 6};
  const std::array<int64_t, 2> domain = {rows, features};
  if (tasks[0].getOrdinal() != 0 || tasks[1].getOrdinal() != 1 ||
      tasks[0].getKind() != mlir::shuttle::MaterializationTaskKind::Fold ||
      tasks[1].getKind() != mlir::shuttle::MaterializationTaskKind::Fold ||
      tasks[0].getDomainShape() != llvm::ArrayRef<int64_t>(domain) ||
      tasks[1].getDomainShape() != llvm::ArrayRef<int64_t>(domain) ||
      tasks[0].getReductionDimensions() != llvm::ArrayRef<int64_t>({1}) ||
      tasks[1].getReductionDimensions() != llvm::ArrayRef<int64_t>({0}) ||
      tasks[0].getInputBuffers() != llvm::ArrayRef(expectedRowInputs) ||
      tasks[1].getInputBuffers() != llvm::ArrayRef(expectedFeatureInputs) ||
      tasks[0].getOutputBuffers() != llvm::ArrayRef(expectedRowBuffers) ||
      tasks[1].getOutputBuffers() !=
          llvm::ArrayRef(expectedFeatureBuffers) ||
      !tasks[0].getDependencies().empty() ||
      tasks[1].getDependencies() != llvm::ArrayRef<int64_t>({0}) ||
      !hasExactFoldRegionBoundaryUses(tasks[0], rowBoundaryUses) ||
      !hasExactFoldRegionBoundaryUses(tasks[1], featureBoundaryUses))
    return false;

  auto hasTensor = [](mlir::shuttle::MaterializationBufferOp buffer,
                      llvm::ArrayRef<int64_t> shape, bool bf16) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(buffer.getTensorType());
    return type && type.getShape() == shape &&
           (bf16 ? type.getElementType().isBF16()
                 : type.getElementType().isF32());
  };
  const std::array<int64_t, 1> featureShape = {features};
  const std::array<int64_t, 2> compactRowShape = {rows, 1};
  for (auto [ordinal, buffer] : llvm::enumerate(buffers)) {
    if (buffer.getOrdinal() != static_cast<int64_t>(ordinal))
      return false;
  }
  for (int64_t ordinal = 0; ordinal < inputCount; ++ordinal) {
    bool vectorInput = ordinal == 1 || (composed && ordinal == 2);
    auto buffer = buffers[ordinal];
    if (!hasTensor(buffer, vectorInput ? llvm::ArrayRef<int64_t>(featureShape)
                                      : llvm::ArrayRef<int64_t>(domain),
                   true) ||
        !buffer.getLiveIn() || buffer.getLiveOut() ||
        buffer.getStorage() !=
            mlir::shuttle::MaterializationStorage::External)
      return false;
  }
  for (int64_t ordinal : {meanBuffer, inverseBuffer}) {
    auto buffer = buffers[ordinal];
    if (!hasTensor(buffer, compactRowShape, false) || buffer.getLiveIn() ||
        buffer.getLiveOut() ||
        buffer.getStorage() !=
            mlir::shuttle::MaterializationStorage::Temporary ||
        buffer.getProducer() != 0 ||
        buffer.getConsumers() != llvm::ArrayRef<int64_t>({1}) ||
        buffer.getLifetimeStart() != 0 || buffer.getLifetimeEnd() != 1)
      return false;
  }
  const int64_t publicStart = inputCount + 2;
  for (int64_t result = 0; result < (composed ? 4 : 3); ++result) {
    auto buffer = buffers[publicStart + result];
    const bool rowResult = composed ? result == 0 || result == 3 : result == 2;
    if (!hasTensor(buffer, rowResult ? llvm::ArrayRef<int64_t>(domain)
                                    : llvm::ArrayRef<int64_t>(featureShape),
                   true) ||
        buffer.getLiveIn() || !buffer.getLiveOut() ||
        buffer.getStorage() !=
            mlir::shuttle::MaterializationStorage::External ||
        buffer.getProducer() != (rowResult ? 0 : 1) ||
        !buffer.getConsumers().empty())
      return false;
  }
  return true;
}

bool hasGenericSharedStatPartition(mlir::ModuleOp module, int64_t rows,
                                   int64_t features) {
  llvm::SmallVector<mlir::shuttle::MaterializationPlanOp> plans(
      module.getOps<mlir::shuttle::MaterializationPlanOp>());
  if (plans.size() != 1)
    return false;
  auto tasks =
      llvm::to_vector(plans.front()
                          .getBody()
                          .front()
                          .getOps<mlir::shuttle::MaterializationTaskOp>());
  auto buffers =
      llvm::to_vector(plans.front()
                          .getBody()
                          .front()
                          .getOps<mlir::shuttle::MaterializationBufferOp>());
  if (tasks.size() != 2 || buffers.size() != 7)
    return false;

  const llvm::SmallVector<int64_t> expectedRowSources =
      llvm::to_vector(llvm::seq<int64_t>(0, 13));
  constexpr std::array<int64_t, 7> expectedFeatureSources = {3,  8,  9, 13,
                                                             14, 15, 16};
  constexpr std::array<int64_t, 3> expectedRecomputed = {3, 8, 9};
  auto rowSources =
      tasks[0]->getAttrOfType<mlir::ArrayAttr>("fold_region_sources");
  auto featureSources =
      tasks[1]->getAttrOfType<mlir::ArrayAttr>("fold_region_sources");
  auto rowOutputs =
      tasks[0]->getAttrOfType<mlir::ArrayAttr>("fold_region_output_sources");
  auto featureOutputs =
      tasks[1]->getAttrOfType<mlir::ArrayAttr>("fold_region_output_sources");
  auto rowRecomputed = tasks[0]->getAttrOfType<mlir::ArrayAttr>(
      "fold_region_recomputed_sources");
  auto featureRecomputed = tasks[1]->getAttrOfType<mlir::ArrayAttr>(
      "fold_region_recomputed_sources");
  if (!rowSources || !featureSources || !rowOutputs || !featureOutputs ||
      !rowRecomputed || !featureRecomputed ||
      sourceOperationOrdinals(rowSources) != expectedRowSources ||
      sourceOperationOrdinals(featureSources) !=
          llvm::ArrayRef<int64_t>(expectedFeatureSources) ||
      sourceOperationOrdinals(rowOutputs) !=
          llvm::ArrayRef<int64_t>({2, 7, 12}) ||
      sourceOperationOrdinals(featureOutputs) !=
          llvm::ArrayRef<int64_t>({14, 16}) ||
      sourceOperationOrdinals(rowRecomputed) !=
          llvm::ArrayRef<int64_t>(expectedRecomputed) ||
      sourceOperationOrdinals(featureRecomputed) !=
          llvm::ArrayRef<int64_t>(expectedRecomputed) ||
      foldRegionPhaseOrdinals(tasks[0]) !=
          llvm::ArrayRef<int64_t>({1, 6, 11}) ||
      foldRegionPhaseOrdinals(tasks[1]) != llvm::ArrayRef<int64_t>({14, 16}))
    return false;

  const std::array<int64_t, 2> domain = {rows, features};
  const std::array<int64_t, 1> rowShape = {rows};
  const std::array<int64_t, 1> featureShape = {features};
  mlir::MLIRContext *context = module.getContext();
  const mlir::AffineMap matrix =
      mlir::AffineMap::getMultiDimIdentityMap(2, context);
  const mlir::AffineMap row = mlir::AffineMap::get(
      2, 0, {mlir::getAffineDimExpr(0, context)}, context);
  const std::array<ExpectedFoldRegionBoundaryUse, 2> rowBoundaryUses = {
      ExpectedFoldRegionBoundaryUse{3, 0, 0, matrix, 1},
      ExpectedFoldRegionBoundaryUse{9, 1, 1, matrix, 2},
  };
  const std::array<ExpectedFoldRegionBoundaryUse, 4> featureBoundaryUses = {
      ExpectedFoldRegionBoundaryUse{3, 0, 0, matrix, 0},
      ExpectedFoldRegionBoundaryUse{3, 1, 2, row, 0},
      ExpectedFoldRegionBoundaryUse{8, 1, 3, row, 0},
      ExpectedFoldRegionBoundaryUse{9, 1, 1, matrix, 0},
  };
  if (tasks[0].getOrdinal() != 0 || tasks[1].getOrdinal() != 1 ||
      tasks[0].getKind() != mlir::shuttle::MaterializationTaskKind::Fold ||
      tasks[1].getKind() != mlir::shuttle::MaterializationTaskKind::Fold ||
      tasks[0].getDomainShape() != llvm::ArrayRef<int64_t>(domain) ||
      tasks[1].getDomainShape() != llvm::ArrayRef<int64_t>(domain) ||
      tasks[0].getReductionDimensions() != llvm::ArrayRef<int64_t>({1}) ||
      tasks[1].getReductionDimensions() != llvm::ArrayRef<int64_t>({0}) ||
      tasks[0].getInputBuffers() != llvm::ArrayRef<int64_t>({0, 1}) ||
      tasks[0].getOutputBuffers() != llvm::ArrayRef<int64_t>({2, 3, 4}) ||
      !tasks[0].getDependencies().empty() ||
      tasks[1].getInputBuffers() != llvm::ArrayRef<int64_t>({0, 1, 2, 3}) ||
      tasks[1].getOutputBuffers() != llvm::ArrayRef<int64_t>({5, 6}) ||
      tasks[1].getDependencies() != llvm::ArrayRef<int64_t>({0}) ||
      !hasExactFoldRegionBoundaryUses(tasks[0], rowBoundaryUses) ||
      !hasExactFoldRegionBoundaryUses(tasks[1], featureBoundaryUses))
    return false;

  auto tensor = [](mlir::shuttle::MaterializationBufferOp buffer,
                   llvm::ArrayRef<int64_t> shape) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(buffer.getTensorType());
    return type && type.getShape() == shape && type.getElementType().isF32();
  };
  for (auto [ordinal, buffer] : llvm::enumerate(buffers)) {
    if (buffer.getOrdinal() != static_cast<int64_t>(ordinal))
      return false;
  }
  if (!tensor(buffers[0], domain) || !tensor(buffers[1], domain) ||
      !buffers[0].getLiveIn() || !buffers[1].getLiveIn() ||
      buffers[0].getLiveOut() || buffers[1].getLiveOut() ||
      buffers[0].getStorage() !=
          mlir::shuttle::MaterializationStorage::External ||
      buffers[1].getStorage() !=
          mlir::shuttle::MaterializationStorage::External)
    return false;
  for (int64_t ordinal : {2, 3}) {
    auto buffer = buffers[ordinal];
    if (!tensor(buffer, rowShape) || buffer.getLiveIn() ||
        buffer.getLiveOut() ||
        buffer.getStorage() !=
            mlir::shuttle::MaterializationStorage::Temporary ||
        buffer.getProducer() != 0 ||
        buffer.getConsumers() != llvm::ArrayRef<int64_t>({1}) ||
        buffer.getLifetimeStart() != 0 || buffer.getLifetimeEnd() != 1)
      return false;
  }
  if (!tensor(buffers[4], domain) || !tensor(buffers[5], featureShape) ||
      !tensor(buffers[6], featureShape))
    return false;
  for (int64_t ordinal : {4, 5, 6}) {
    auto buffer = buffers[ordinal];
    const int64_t producer = ordinal == 4 ? 0 : 1;
    if (buffer.getLiveIn() || !buffer.getLiveOut() ||
        buffer.getStorage() !=
            mlir::shuttle::MaterializationStorage::External ||
        buffer.getProducer() != producer || !buffer.getConsumers().empty())
      return false;
  }
  return true;
}

bool mutateGenericSharedStatPartition(mlir::ModuleOp module,
                                      llvm::StringRef mutation) {
  auto plan = *module.getOps<mlir::shuttle::MaterializationPlanOp>().begin();
  auto tasks = llvm::to_vector(
      plan.getBody().front().getOps<mlir::shuttle::MaterializationTaskOp>());
  auto buffers = llvm::to_vector(
      plan.getBody().front().getOps<mlir::shuttle::MaterializationBufferOp>());
  if (tasks.empty() || buffers.size() < 3)
    return false;
  mlir::Builder builder(module.getContext());
  if (mutation == "source-order") {
    auto sources =
        tasks.front()->getAttrOfType<mlir::ArrayAttr>("fold_region_sources");
    llvm::SmallVector<mlir::Attribute> changed;
    if (sources && sources.size() > 1)
      llvm::append_range(changed, llvm::reverse(sources.getValue()));
    else
      changed.push_back(
          mlir::shuttle::SourceRefAttr::get(module.getContext(), 0, 0, 99, 0));
    tasks.front()->setAttr("fold_region_sources",
                           builder.getArrayAttr(changed));
  } else if (mutation == "drop-stat-input") {
    llvm::SmallVector<int64_t> inputs(tasks.back().getInputBuffers());
    auto stat = llvm::find(inputs, 2);
    if (stat == inputs.end())
      inputs.clear();
    else
      inputs.erase(stat);
    tasks.back()->setAttr("input_buffers",
                          builder.getDenseI64ArrayAttr(inputs));
  } else if (mutation == "swap-stat-bindings") {
    auto uses = tasks.back()->getAttrOfType<mlir::ArrayAttr>(
        "fold_region_boundary_uses");
    llvm::SmallVector<mlir::Attribute> changed;
    if (uses)
      llvm::append_range(changed, uses.getValue());
    bool changedMean = false;
    bool changedInverse = false;
    for (mlir::Attribute &attribute : changed) {
      auto use = mlir::dyn_cast<mlir::DictionaryAttr>(attribute);
      auto source =
          use ? use.getAs<mlir::shuttle::SourceRefAttr>("source")
              : mlir::shuttle::SourceRefAttr{};
      auto operand = use ? use.getAs<mlir::IntegerAttr>("operand")
                         : mlir::IntegerAttr{};
      if (!source || !operand || operand.getInt() != 1 ||
          (source.getOperationOrdinal() != 3 &&
           source.getOperationOrdinal() != 8))
        continue;
      llvm::SmallVector<mlir::NamedAttribute> fields(use.begin(), use.end());
      auto inputBuffer = llvm::find_if(fields, [](mlir::NamedAttribute field) {
        return field.getName() == "input_buffer";
      });
      if (inputBuffer == fields.end())
        return false;
      const bool mean = source.getOperationOrdinal() == 3;
      *inputBuffer = builder.getNamedAttr(
          "input_buffer", builder.getI64IntegerAttr(mean ? 3 : 2));
      attribute = builder.getDictionaryAttr(fields);
      changedMean |= mean;
      changedInverse |= !mean;
    }
    if (!changedMean || !changedInverse) {
      mlir::AffineMap row = mlir::AffineMap::get(
          2, 0, {mlir::getAffineDimExpr(0, module.getContext())},
          module.getContext());
      auto wrongUse = [&](int64_t source, int64_t inputBuffer) {
        return builder.getDictionaryAttr(
            {builder.getNamedAttr(
                 "input_buffer", builder.getI64IntegerAttr(inputBuffer)),
             builder.getNamedAttr(
                 "source",
                 mlir::shuttle::SourceRefAttr::get(module.getContext(), 0, 0,
                                                    source, 0)),
             builder.getNamedAttr("operand", builder.getI64IntegerAttr(1)),
             builder.getNamedAttr("indexing_map",
                                  mlir::AffineMapAttr::get(row)),
             builder.getNamedAttr("phase", builder.getI64IntegerAttr(0))});
      };
      changed = {wrongUse(3, 3), wrongUse(8, 2)};
    }
    tasks.back()->setAttr("fold_region_boundary_uses",
                          builder.getArrayAttr(changed));
  } else if (mutation == "swap-public-outputs") {
    llvm::SmallVector<int64_t> outputs(tasks.back().getOutputBuffers());
    if (outputs.size() < 2)
      outputs.push_back(outputs.empty() ? 0 : outputs.front());
    std::swap(outputs[0], outputs[1]);
    tasks.back()->setAttr("output_buffers",
                          builder.getDenseI64ArrayAttr(outputs));
  } else if (mutation == "stat-producer") {
    buffers[2]->setAttr("producer", builder.getI64IntegerAttr(1));
  } else if (mutation == "remove-output-source" ||
             mutation == "duplicate-output-source" ||
             mutation == "swap-output-sources") {
    auto outputs = tasks.back()->getAttrOfType<mlir::ArrayAttr>(
        "fold_region_output_sources");
    llvm::SmallVector<mlir::Attribute> changed;
    if (outputs)
      llvm::append_range(changed, outputs.getValue());
    if (changed.size() < 2)
      return false;
    if (mutation == "remove-output-source")
      changed.pop_back();
    else if (mutation == "duplicate-output-source")
      changed[1] = changed[0];
    else
      std::swap(changed[0], changed[1]);
    tasks.back()->setAttr("fold_region_output_sources",
                          builder.getArrayAttr(changed));
  } else if (mutation == "stat-storage") {
    buffers[2]->setAttr("storage",
                        mlir::shuttle::MaterializationStorageAttr::get(
                            module.getContext(),
                            mlir::shuttle::MaterializationStorage::External));
  } else {
    return false;
  }
  plan.setFingerprint(mlir::shuttle::materializationPlanFingerprint(plan));
  return true;
}

std::string ptxFingerprintDigest(const PtxDataflow &flow,
                                 llvm::StringRef value);

std::string ptxOutputControlDigest(llvm::StringRef ptx,
                                   int64_t outputOrdinal) {
  const std::string epilogueLabel =
      (llvm::Twine("$fold_region_output_") + llvm::Twine(outputOrdinal) +
       "_epilogue:")
          .str();
  const std::string doneLabel =
      (llvm::Twine("$fold_region_output_") + llvm::Twine(outputOrdinal) +
       "_done:")
          .str();
  size_t done = ptx.find(doneLabel);
  if (done == llvm::StringRef::npos)
    return {};
  size_t epilogue = ptx.take_front(done).rfind(epilogueLabel);
  size_t control = epilogue;
  if (control == llvm::StringRef::npos) {
    const std::string branch =
        (llvm::Twine("bra $fold_region_output_") +
         llvm::Twine(outputOrdinal) + "_done;")
            .str();
    control = ptx.take_front(done).rfind(branch);
  }
  if (control == llvm::StringRef::npos)
    return {};
  size_t start = ptx.take_front(control).rfind('\n');
  if (start == llvm::StringRef::npos)
    return {};
  start = ptx.take_front(start).rfind('\n');
  start = start == llvm::StringRef::npos ? 0 : start + 1;
  llvm::StringRef slice = ptx.slice(start, done + doneLabel.size());
  PtxDataflow prefixFlow = ptxDataflow(ptx.take_front(start));

  std::map<std::string, std::string> registers;
  int64_t nextPredicate = 0;
  int64_t nextU32 = 0;
  int64_t nextU64 = 0;
  int64_t nextF32 = 0;
  int64_t nextBf16 = 0;
  auto canonicalRegister = [&](llvm::StringRef token) -> std::string {
    auto found = registers.find(token.str());
    if (found != registers.end())
      return found->second;
    if (token == "%r0" || token == "%r1") {
      auto reaching = prefixFlow.currentValues.find(token.str());
      if (reaching == prefixFlow.currentValues.end())
        return {};
      std::string canonical =
          "$live:" + ptxFingerprintDigest(prefixFlow, reaching->second);
      registers.try_emplace(token.str(), canonical);
      return canonical;
    }
    llvm::StringRef prefix;
    int64_t *next = nullptr;
    if (token.starts_with("%p")) {
      prefix = "$p";
      next = &nextPredicate;
    } else if (token.starts_with("%rd")) {
      prefix = "$rd";
      next = &nextU64;
    } else if (token.starts_with("%f")) {
      prefix = "$f";
      next = &nextF32;
    } else if (token.starts_with("%h")) {
      prefix = "$h";
      next = &nextBf16;
    } else {
      prefix = "$r";
      next = &nextU32;
    }
    std::string canonical =
        (llvm::Twine(prefix) + llvm::Twine((*next)++)).str();
    registers.try_emplace(token.str(), canonical);
    return canonical;
  };
  std::string skeleton;
  size_t cursor = 0;
  for (llvm::StringRef token : ptxTokens(slice)) {
    size_t position = slice.find(token, cursor);
    if (position == llvm::StringRef::npos)
      return {};
    skeleton += slice.slice(cursor, position).str();
    std::string replacement =
        token.starts_with("%") ? canonicalRegister(token) : token.str();
    if (replacement.empty())
      return {};
    skeleton += replacement;
    cursor = position + token.size();
  }
  skeleton += slice.drop_front(cursor).str();
  llvm::SHA256 digest;
  digest.update(skeleton);
  return llvm::toHex(digest.final(), true);
}

struct ExpectedPtxOutput {
  bool bf16;
  llvm::StringRef pointerDigest;
  llvm::StringRef valueDigest;
  llvm::StringRef controlDigest;
  llvm::StringRef inputParameters;
};

bool hasExactMultiOutputStoreDataflow(
    llvm::StringRef ptx, int64_t inputCount,
    llvm::ArrayRef<ExpectedPtxOutput> expectedOutputs) {
  PtxDataflow flow = ptxDataflow(ptx);
  if (flow.stores.size() != expectedOutputs.size())
    return false;
  for (auto [outputOrdinal, store, expected] :
       llvm::enumerate(flow.stores, expectedOutputs)) {
    std::set<std::string> pointer = ptxDependencyClosure(flow, store.pointer);
    const std::string parameter =
        (llvm::Twine("p") + llvm::Twine(inputCount + outputOrdinal)).str();
    if (store.bf16 != expected.bf16 || !pointer.count(parameter) ||
        !pointer.count("$op:mad.lo.u64") ||
        ptxFingerprintDigest(flow, store.pointer) != expected.pointerDigest ||
        ptxFingerprintDigest(flow, store.value) != expected.valueDigest ||
        ptxOutputControlDigest(ptx, outputOrdinal) != expected.controlDigest)
      return false;
    for (int64_t other = 0;
         other < static_cast<int64_t>(expectedOutputs.size()); ++other) {
      const std::string otherParameter =
          (llvm::Twine("p") + llvm::Twine(inputCount + other)).str();
      if (other != static_cast<int64_t>(outputOrdinal) &&
          pointer.count(otherParameter))
        return false;
    }
    std::set<std::string> value = ptxDependencyClosure(flow, store.value);
    if (!value.count("$shared") && !value.count("$ld.global.f32") &&
        !value.count("$ld.global.u16"))
      return false;
    std::string parameters;
    for (const std::string &dependency : value) {
      if (!llvm::StringRef(dependency).starts_with("p"))
        continue;
      if (!parameters.empty())
        parameters += ",";
      parameters += dependency;
    }
    if (parameters != expected.inputParameters)
      return false;
  }
  return true;
}

std::string ptxFingerprintDigest(const PtxDataflow &flow,
                                 llvm::StringRef value) {
  std::map<std::string, std::string> memo;
  std::set<std::string> active;
  auto digestValue = [&](auto &self, llvm::StringRef current) -> std::string {
    auto memoized = memo.find(current.str());
    if (memoized != memo.end())
      return memoized->second;
    if (!active.insert(current.str()).second)
      return "$cycle";
    llvm::SHA256 digest;
    const PtxDataflow::Definition *definition = ptxDefinition(flow, current);
    if (!definition) {
      digest.update(current);
    } else {
      digest.update(definition->predicated ? "@" : "_");
      digest.update(ptxInstructionSkeleton(definition->instruction));
      for (llvm::StringRef operand : definition->operands)
        digest.update(self(self, operand));
    }
    active.erase(current.str());
    std::string result = llvm::toHex(digest.final(), true);
    memo.try_emplace(current.str(), result);
    return result;
  };
  return digestValue(digestValue, value);
}

std::string misrouteFirstPtxOutput(llvm::StringRef ptx, int64_t inputCount) {
  std::string result = ptx.str();
  const std::string correct =
      (llvm::Twine("[p") + llvm::Twine(inputCount) + "]").str();
  const std::string wrong =
      (llvm::Twine("[p") + llvm::Twine(inputCount + 1) + "]").str();
  size_t position = result.find(correct);
  if (position != std::string::npos)
    result.replace(position, correct.size(), wrong);
  return result;
}

std::string zeroFirstPtxOutputValue(llvm::StringRef ptx) {
  std::string result = ptx.str();
  size_t store = result.find("st.global.");
  size_t comma = store == std::string::npos ? std::string::npos
                                            : result.find(", ", store);
  size_t end = comma == std::string::npos ? std::string::npos
                                          : result.find(';', comma);
  if (end != std::string::npos)
    result.replace(comma + 2, end - comma - 2, "0f00000000");
  return result;
}

std::string swapFirstTwoPtxOutputValues(llvm::StringRef ptx) {
  std::string result = ptx.str();
  llvm::SmallVector<size_t> stores;
  for (size_t position = 0;
       (position = result.find("st.global.", position)) != std::string::npos;
       ++position)
    stores.push_back(position);
  if (stores.size() < 2)
    return result;
  auto valueRange = [&](size_t store) {
    size_t begin = result.find(", ", store);
    size_t end = begin == std::string::npos ? std::string::npos
                                            : result.find(';', begin);
    return std::pair(begin == std::string::npos ? begin : begin + 2, end);
  };
  auto [firstBegin, firstEnd] = valueRange(stores[0]);
  auto [secondBegin, secondEnd] = valueRange(stores[1]);
  if (firstEnd == std::string::npos || secondEnd == std::string::npos)
    return result;
  std::string first = result.substr(firstBegin, firstEnd - firstBegin);
  std::string second = result.substr(secondBegin, secondEnd - secondBegin);
  result.replace(secondBegin, secondEnd - secondBegin, first);
  result.replace(firstBegin, firstEnd - firstBegin, second);
  return result;
}

std::string rerouteFirstPtxInputLeaf(llvm::StringRef ptx) {
  std::string result = ptx.str();
  size_t position = result.find("[p0]");
  if (position != std::string::npos)
    result.replace(position, 4, "[p1]");
  return result;
}

std::string changeFirstPtxOutputByteScale(llvm::StringRef ptx) {
  std::string result = ptx.str();
  size_t store = result.find("st.global.");
  if (store == std::string::npos)
    return result;
  size_t lineStart = result.rfind('\n', store);
  llvm::StringRef storeLine(result.data() + (lineStart + 1),
                            result.find('\n', store) - lineStart - 1);
  std::vector<std::string> storeTokens = ptxTokens(storeLine);
  if (storeTokens.empty())
    return result;
  std::string definition = "mad.lo.u64 " + storeTokens[0] + ",";
  size_t address = result.rfind(definition, store);
  size_t addressEnd = address == std::string::npos ? std::string::npos
                                                   : result.find('\n', address);
  if (addressEnd == std::string::npos)
    return result;
  size_t scale = result.find(", 4,", address);
  if (scale != std::string::npos && scale < addressEnd)
    result.replace(scale, 4, ", 8,");
  else {
    scale = result.find(", 2,", address);
    if (scale != std::string::npos && scale < addressEnd)
      result.replace(scale, 4, ", 4,");
  }
  return result;
}

std::string flipFirstPtxOutputGuardPolarity(llvm::StringRef ptx) {
  std::string result = ptx.str();
  size_t branch = result.find("@%p3 bra $fold_region_output_0_");
  if (branch != std::string::npos)
    result.insert(branch + 1, "!");
  return result;
}

std::string redirectFirstPtxOutputGuard(llvm::StringRef ptx) {
  std::string result = ptx.str();
  const std::string branch = "bra $fold_region_output_0_done;";
  size_t position = result.find(branch);
  if (position != std::string::npos)
    result.replace(position, branch.size(), "bra $done;");
  return result;
}

std::string rerouteFirstFoldOutputGuardToBlockId(llvm::StringRef ptx) {
  std::string result = ptx.str();
  const std::string correct = "setp.ne.u32 %p3, %r0, 0;";
  size_t position = result.find(correct);
  if (position != std::string::npos)
    result.replace(position, correct.size(), "setp.ne.u32 %p3, %r1, 0;");
  return result;
}

constexpr llvm::StringLiteral kGenericSharedStatAlgebra = R"mlir(
#matrix = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
#vector = affine_map<(d0) -> (d0)>
#scalar = affine_map<() -> ()>
module attributes {
  shuttle.coverage_manifest = {
    canonical_tuning = "{\22cluster_shape\22:[],\22materialization\22:\22prefer_fusion\22,\22maximum_candidates\22:1,\22pipeline_stages\22:1,\22tile_sizes\22:[]}"
  }
} {
  func.func @anonymous(%x: tensor<2x4xf32>, %dy: tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<4xf32>, tensor<4xf32>) {
    %row_output, %dependent_column, %independent_column = "shuttle.region"(%x, %dy) ({
    ^bb0(%region_x: tensor<2x4xf32>, %region_dy: tensor<2x4xf32>):
      %mean_init = "shuttle.map"() ({
        %zero = arith.constant 0.000000e+00 : f32
        "shuttle.yield"(%zero) : (f32) -> ()
      }) {indexing_maps = [#scalar], source = #shuttle.source_ref<0, 0, 0, 0>} : () -> tensor<f32>
      %mean_sum = "shuttle.fold"(%region_x, %mean_init) ({
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %element, %accumulator : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {accumulator_types = [f32], operandSegmentSizes = array<i32: 1, 1>, order_free = true, reduction_dimensions = array<i64: 1>, source = #shuttle.source_ref<0, 0, 1, 0>} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
      %mean = "shuttle.map"(%mean_sum) ({
      ^bb0(%sum: f32):
        %count = arith.constant 4.000000e+00 : f32
        %value = arith.divf %sum, %count : f32
        "shuttle.yield"(%value) : (f32) -> ()
      }) {indexing_maps = [#vector, #vector], source = #shuttle.source_ref<0, 0, 2, 0>} : (tensor<2xf32>) -> tensor<2xf32>
      %centered = "shuttle.map"(%region_x, %mean) ({
      ^bb0(%element: f32, %row_mean: f32):
        %value = arith.subf %element, %row_mean : f32
        "shuttle.yield"(%value) : (f32) -> ()
      }) {indexing_maps = [#matrix, #row, #matrix], source = #shuttle.source_ref<0, 0, 3, 0>} : (tensor<2x4xf32>, tensor<2xf32>) -> tensor<2x4xf32>
      %square = "shuttle.map"(%centered) ({
      ^bb0(%element: f32):
        %value = arith.mulf %element, %element : f32
        "shuttle.yield"(%value) : (f32) -> ()
      }) {indexing_maps = [#matrix, #matrix], source = #shuttle.source_ref<0, 0, 4, 0>} : (tensor<2x4xf32>) -> tensor<2x4xf32>
      %variance_init = "shuttle.map"() ({
        %zero = arith.constant 0.000000e+00 : f32
        "shuttle.yield"(%zero) : (f32) -> ()
      }) {indexing_maps = [#scalar], source = #shuttle.source_ref<0, 0, 5, 0>} : () -> tensor<f32>
      %variance_sum = "shuttle.fold"(%square, %variance_init) ({
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %element, %accumulator : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {accumulator_types = [f32], operandSegmentSizes = array<i32: 1, 1>, order_free = true, reduction_dimensions = array<i64: 1>, source = #shuttle.source_ref<0, 0, 6, 0>} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
      %inverse = "shuttle.map"(%variance_sum) ({
      ^bb0(%sum: f32):
        %count = arith.constant 4.000000e+00 : f32
        %epsilon = arith.constant 1.000000e-05 : f32
        %variance = arith.divf %sum, %count : f32
        %shifted = arith.addf %variance, %epsilon : f32
        %value = math.rsqrt %shifted : f32
        "shuttle.yield"(%value) : (f32) -> ()
      }) {indexing_maps = [#vector, #vector], source = #shuttle.source_ref<0, 0, 7, 0>} : (tensor<2xf32>) -> tensor<2xf32>
      %normalized = "shuttle.map"(%centered, %inverse) ({
      ^bb0(%element: f32, %row_inverse: f32):
        %value = arith.mulf %element, %row_inverse : f32
        "shuttle.yield"(%value) : (f32) -> ()
      }) {indexing_maps = [#matrix, #row, #matrix], source = #shuttle.source_ref<0, 0, 8, 0>} : (tensor<2x4xf32>, tensor<2xf32>) -> tensor<2x4xf32>
      %weighted = "shuttle.map"(%normalized, %region_dy) ({
      ^bb0(%element: f32, %cotangent: f32):
        %value = arith.mulf %element, %cotangent : f32
        "shuttle.yield"(%value) : (f32) -> ()
      }) {indexing_maps = [#matrix, #matrix, #matrix], source = #shuttle.source_ref<0, 0, 9, 0>} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
      %correction_init = "shuttle.map"() ({
        %zero = arith.constant 0.000000e+00 : f32
        "shuttle.yield"(%zero) : (f32) -> ()
      }) {indexing_maps = [#scalar], source = #shuttle.source_ref<0, 0, 10, 0>} : () -> tensor<f32>
      %correction = "shuttle.fold"(%weighted, %correction_init) ({
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %element, %accumulator : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {accumulator_types = [f32], operandSegmentSizes = array<i32: 1, 1>, order_free = true, reduction_dimensions = array<i64: 1>, source = #shuttle.source_ref<0, 0, 11, 0>} : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
      %row_result = "shuttle.map"(%normalized, %correction) ({
      ^bb0(%element: f32, %row_correction: f32):
        %count = arith.constant 4.000000e+00 : f32
        %average = arith.divf %row_correction, %count : f32
        %value = arith.subf %element, %average : f32
        "shuttle.yield"(%value) : (f32) -> ()
      }) {indexing_maps = [#matrix, #row, #matrix], source = #shuttle.source_ref<0, 0, 12, 0>} : (tensor<2x4xf32>, tensor<2xf32>) -> tensor<2x4xf32>
      %dependent_init = "shuttle.map"() ({
        %zero = arith.constant 0.000000e+00 : f32
        "shuttle.yield"(%zero) : (f32) -> ()
      }) {indexing_maps = [#scalar], source = #shuttle.source_ref<0, 0, 13, 0>} : () -> tensor<f32>
      %dependent = "shuttle.fold"(%weighted, %dependent_init) ({
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %element, %accumulator : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {accumulator_types = [f32], operandSegmentSizes = array<i32: 1, 1>, order_free = true, reduction_dimensions = array<i64: 0>, source = #shuttle.source_ref<0, 0, 14, 0>} : (tensor<2x4xf32>, tensor<f32>) -> tensor<4xf32>
      %independent_init = "shuttle.map"() ({
        %zero = arith.constant 0.000000e+00 : f32
        "shuttle.yield"(%zero) : (f32) -> ()
      }) {indexing_maps = [#scalar], source = #shuttle.source_ref<0, 0, 15, 0>} : () -> tensor<f32>
      %independent = "shuttle.fold"(%region_dy, %independent_init) ({
      ^bb0(%element: f32, %accumulator: f32):
        %sum = arith.addf %element, %accumulator : f32
        "shuttle.yield"(%sum) : (f32) -> ()
      }) {accumulator_types = [f32], operandSegmentSizes = array<i32: 1, 1>, order_free = true, reduction_dimensions = array<i64: 0>, source = #shuttle.source_ref<0, 0, 16, 0>} : (tensor<2x4xf32>, tensor<f32>) -> tensor<4xf32>
      "shuttle.yield"(%row_result, %dependent, %independent) : (tensor<2x4xf32>, tensor<4xf32>, tensor<4xf32>) -> ()
    }) {policy = #shuttle.policy<source_ordered>, source_refs = [#shuttle.source_ref<0, 0, 0, 0>, #shuttle.source_ref<0, 0, 1, 0>, #shuttle.source_ref<0, 0, 2, 0>, #shuttle.source_ref<0, 0, 3, 0>, #shuttle.source_ref<0, 0, 4, 0>, #shuttle.source_ref<0, 0, 5, 0>, #shuttle.source_ref<0, 0, 6, 0>, #shuttle.source_ref<0, 0, 7, 0>, #shuttle.source_ref<0, 0, 8, 0>, #shuttle.source_ref<0, 0, 9, 0>, #shuttle.source_ref<0, 0, 10, 0>, #shuttle.source_ref<0, 0, 11, 0>, #shuttle.source_ref<0, 0, 12, 0>, #shuttle.source_ref<0, 0, 13, 0>, #shuttle.source_ref<0, 0, 14, 0>, #shuttle.source_ref<0, 0, 15, 0>, #shuttle.source_ref<0, 0, 16, 0>]} : (tensor<2x4xf32>, tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<4xf32>, tensor<4xf32>)
    return %row_output, %dependent_column, %independent_column : tensor<2x4xf32>, tensor<4xf32>, tensor<4xf32>
  }
}
)mlir";

TEST(Abi22MapFusionContract,
     SharedRowStatsPartitionRowAndSiblingColumnOutputsGenerically) {
  llvm::StringRef source = kGenericSharedStatAlgebra;
  constexpr std::array<llvm::StringLiteral, 2> directFoldControlDigests = {
      "a574d2e3a5d2e2e2c56a7b19f5f35f0c5011f4e61c30a57446227a91c88ad512",
      "59f8da67863782cda325603bdec07ee371ddb657e55678e00120771efd6c7e60",
  };
  struct ShapeCase {
    int64_t rows;
    int64_t features;
  };
  for (ShapeCase shape : {ShapeCase{2, 4}, ShapeCase{3, 7}}) {
    std::string shapeSource = source.str();
    if (shape.rows != 2 || shape.features != 4) {
      shapeSource = replaceAllText(shapeSource, "2x4", "3x7");
      shapeSource =
          replaceAllText(shapeSource, "tensor<2xf32>", "tensor<3xf32>");
      shapeSource =
          replaceAllText(shapeSource, "tensor<4xf32>", "tensor<7xf32>");
      shapeSource =
          replaceAllText(shapeSource, "4.000000e+00", "7.000000e+00");
    }
    for (mlir::shuttle::NumericalPolicy policy :
         {mlir::shuttle::NumericalPolicy::SourceOrdered,
          mlir::shuttle::NumericalPolicy::Fast}) {
      SCOPED_TRACE(std::to_string(shape.rows) + "x" +
                   std::to_string(shape.features));
      std::unique_ptr<BuiltModule> built = buildAlgebra(shapeSource, policy);
      ASSERT_TRUE(built);
      EXPECT_TRUE(hasGenericSharedStatPartition(*built->module, shape.rows,
                                                shape.features));

      auto device =
          *built->module->getOps<mlir::shuttle::DeviceModuleOp>().begin();
      auto entries = llvm::to_vector(
          device.getBody().front().getOps<mlir::shuttle::DeviceEntryOp>());
      ASSERT_EQ(entries.size(), 2u);
      auto featureEntry = entries[1];
      llvm::StringRef ptx(
          reinterpret_cast<const char *>(device.getCode().data() +
                                         featureEntry.getCodeOffset()),
          featureEntry.getCodeLength());
      for (auto [outputOrdinal, expected] :
           llvm::enumerate(directFoldControlDigests))
        EXPECT_EQ(ptxOutputControlDigest(ptx, outputOrdinal), expected);
      auto expectFirstFoldControlRejected = [&](std::string mutated) {
        ASSERT_NE(mutated, ptx);
        EXPECT_NE(ptxOutputControlDigest(mutated, 0),
                  directFoldControlDigests[0]);
      };
      expectFirstFoldControlRejected(
          rerouteFirstFoldOutputGuardToBlockId(ptx));
      expectFirstFoldControlRejected(flipFirstPtxOutputGuardPolarity(ptx));
      expectFirstFoldControlRejected(redirectFirstPtxOutputGuard(ptx));

      for (llvm::StringRef mutation :
           {"source-order", "drop-stat-input", "swap-stat-bindings",
            "swap-public-outputs", "stat-producer", "stat-storage",
            "remove-output-source", "duplicate-output-source",
            "swap-output-sources"}) {
        SCOPED_TRACE(mutation.str());
        std::unique_ptr<BuiltModule> changed =
            buildAlgebra(shapeSource, policy);
        ASSERT_TRUE(changed);
        ASSERT_TRUE(
            mutateGenericSharedStatPartition(*changed->module, mutation));
        EXPECT_FALSE(hasGenericSharedStatPartition(
            *changed->module, shape.rows, shape.features));
        EXPECT_FALSE(verifyMaterializationPlan(*changed->module));
      }
    }
  }
}

bool hasSquareSharedStatFoldControl(mlir::ModuleOp module, bool sharedFold) {
  auto plans =
      llvm::to_vector(module.getOps<mlir::shuttle::MaterializationPlanOp>());
  if (plans.size() != 1)
    return false;
  auto tasks =
      llvm::to_vector(plans.front()
                          .getBody()
                          .front()
                          .getOps<mlir::shuttle::MaterializationTaskOp>());
  if (tasks.size() != 2)
    return false;
  auto sources = [](mlir::shuttle::MaterializationTaskOp task) {
    return sourceOperationOrdinals(
        task->getAttrOfType<mlir::ArrayAttr>("fold_region_sources"));
  };
  const llvm::SmallVector<int64_t> expectedRow =
      llvm::to_vector(llvm::seq<int64_t>(0, 13));
  const llvm::SmallVector<int64_t> expectedFeature =
      sharedFold ? llvm::SmallVector<int64_t>{0, 1, 3, 8, 9, 13, 14, 15, 16, 17}
                 : llvm::SmallVector<int64_t>{3, 8, 9, 13, 14, 15, 16};
  const llvm::SmallVector<int64_t> expectedFeaturePhases =
      sharedFold ? llvm::SmallVector<int64_t>{1, 14, 17}
                 : llvm::SmallVector<int64_t>{14, 16};
  return sources(tasks[0]) == expectedRow &&
         sources(tasks[1]) == expectedFeature &&
         foldRegionPhaseOrdinals(tasks[0]) ==
             llvm::ArrayRef<int64_t>({1, 6, 11}) &&
         foldRegionPhaseOrdinals(tasks[1]) == expectedFeaturePhases &&
         tasks[0].getReductionDimensions() == llvm::ArrayRef<int64_t>({1}) &&
         tasks[1].getReductionDimensions() == llvm::ArrayRef<int64_t>({1});
}

TEST(Abi22MapFusionContract, RejectsCrossTaskFoldRecomputationSpecifically) {
  std::string controlSource = kGenericSharedStatAlgebra.str();
  controlSource = replaceAllText(controlSource, "2x4", "4x4");
  controlSource =
      replaceAllText(controlSource, "tensor<2xf32>", "tensor<4xf32>");
  controlSource =
      replaceAllText(controlSource, "reduction_dimensions = array<i64: 0>",
                     "reduction_dimensions = array<i64: 1>");
  std::unique_ptr<BuiltModule> control = buildAlgebra(
      controlSource, mlir::shuttle::NumericalPolicy::SourceOrdered, false);
  ASSERT_TRUE(control);
  ASSERT_TRUE(hasSquareSharedStatFoldControl(*control->module, false));

  std::string shared = replaceAllText(
      controlSource,
      "      %independent = \"shuttle.fold\"(%region_dy, %independent_init)",
      "      %shared_fold_input = \"shuttle.map\"(%region_dy, %mean_sum) "
      "({\n"
      "      ^bb0(%element: f32, %row_sum: f32):\n"
      "        %value = arith.addf %element, %row_sum : f32\n"
      "        \"shuttle.yield\"(%value) : (f32) -> ()\n"
      "      }) {indexing_maps = [#matrix, #row, #matrix], source = "
      "#shuttle.source_ref<0, 0, 16, 0>} : (tensor<4x4xf32>, "
      "tensor<4xf32>) -> tensor<4x4xf32>\n"
      "      %independent = \"shuttle.fold\"(%shared_fold_input, "
      "%independent_init)");
  shared =
      replaceAllText(shared,
                     "reduction_dimensions = array<i64: 1>, source = "
                     "#shuttle.source_ref<0, 0, 16, 0>} : (tensor<4x4xf32>, "
                     "tensor<f32>) -> tensor<4xf32>\n      \"shuttle.yield\"",
                     "reduction_dimensions = array<i64: 1>, source = "
                     "#shuttle.source_ref<0, 0, 17, 0>} : (tensor<4x4xf32>, "
                     "tensor<f32>) -> tensor<4xf32>\n      \"shuttle.yield\"");
  shared = replaceAllText(
      shared,
      "#shuttle.source_ref<0, 0, 15, 0>, #shuttle.source_ref<0, 0, 16, "
      "0>]}",
      "#shuttle.source_ref<0, 0, 15, 0>, #shuttle.source_ref<0, 0, 16, "
      "0>, #shuttle.source_ref<0, 0, 17, 0>]}");
  std::unique_ptr<BuiltModule> rejected = buildAlgebra(
      shared, mlir::shuttle::NumericalPolicy::SourceOrdered, false);
  ASSERT_TRUE(rejected);
  EXPECT_FALSE(hasSquareSharedStatFoldControl(*rejected->module, true));
}

TEST(Abi22MapFusionContract,
     OrdinaryJaxLayerNormBackwardAndComposedBindGenericTwoTaskTopology) {
  struct Fixture {
    const char *filename;
    int64_t rows;
    int64_t features;
    bool composed;
    int64_t rowOutputCount;
    std::array<const char *, 6> pointerDigests;
    std::array<const char *, 6> valueDigests;
    std::array<const char *, 6> controlDigests;
    std::array<const char *, 6> inputParameters;
  };
  // These digests seal register-insensitive reaching-definition trees. They
  // retain predicates, immediates, input parameters, and address byte scales.
  constexpr std::array<Fixture, 4> fixtures = {{
      {"jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-backward.mlir",
       7,
       13,
       false,
       3,
       {"bbddc0570461712a03c47ad9005bdffe0b6a4cb2bb9b3867906f5052a44e42da",
        "a7f6c3dedf98acd3098ee27ae350b186062d8641f1743190da0c17c7232ca88d",
        "b0796817f5f958da69d54927a5c2b0c3e081591ce6d1645b604f389bb4ea7a95",
        "a9fcb27d2c0ce92b9f2f4855aae48676e5915b8cbd96471c7ae482cf5009f398",
        "2264b4a85cf88e0ca855ec47de72346a2154c40441160baa4110620d75f5ddbd", ""},
       {"7b71842ccee7fdfe2106329aadddf9a809b88dd7a49fe4a30335a5e359c87db5",
        "e7cc4974afdecfe0069fc7d099ab9da3d6606aca3e468929d8d29b6afc5f4a7c",
        "948ec3995104fbff504ef2af2bdef2d5c5b84d72c26c36a1b4f321a8763ffcbd",
        "4ea4548e103b339bd5d55854a32d9ed632ce44a3fac32d39385cfafd84677835",
        "ba762e537b95bdc44ff9ea14e83475661a6ade0179251a656a85b8dd7204a783", ""},
       {"45d5a22cff40debc4aa9f94931bc48963b2a40d18a13aaa802caa085a2511125",
        "541f42872150196df4f51c5f02a4ddaa7d7bf868ba9e0b6aa50f24b03c3194e4",
        "8243c7964b200ca043eec5c49337518c02c43d96a55260f4b189885e04938288",
        "9a4162f457f270197017d0ec305c6c76f4eaab47127d28cd501586455f59fef1",
        "b0eca7cf355f42f5a90aeb0175feb7395fddf91e4d02c2274b9211bdf09f1025",
        ""},
       {"p0", "p0", "p0,p1,p2", "p1", "p0,p1,p2,p3", ""}},
      {"jax-0.10.1-bf16-layer_norm_81928ab3539c0f03-composed.mlir",
       7,
       13,
       true,
       4,
       {"a7f6c3dedf98acd3098ee27ae350b186062d8641f1743190da0c17c7232ca88d",
        "5baade21272a35250fab8e64012a6f3bf4d617a77f14eca81eb0d11a2e348f69",
        "ce54909013fa6797c7617c231162a7dd902bd83b69e77bac9c9314f3514fb2ba",
        "a7df71b6574756f25888934eb405878e32de265579d767533eaff963a74ef067",
        "a9fcb27d2c0ce92b9f2f4855aae48676e5915b8cbd96471c7ae482cf5009f398",
        "2264b4a85cf88e0ca855ec47de72346a2154c40441160baa4110620d75f5ddbd"},
       {"7b71842ccee7fdfe2106329aadddf9a809b88dd7a49fe4a30335a5e359c87db5",
        "e7cc4974afdecfe0069fc7d099ab9da3d6606aca3e468929d8d29b6afc5f4a7c",
        "5a37df67cbfb820e044f7e08cbcc68de1b9008d695a5f07a74ac49018bc6743c",
        "e982e5187665143b7060ee99f8df8c6ecba60aa816514bccd8a0e631c4489dca",
        "4ea4548e103b339bd5d55854a32d9ed632ce44a3fac32d39385cfafd84677835",
        "ba762e537b95bdc44ff9ea14e83475661a6ade0179251a656a85b8dd7204a783"},
       {"c6703f2336e2ba3f0463c60c230ab74e3c9d9590b0c9f69f9072970e474e9a4c",
        "629665b0c4ad9d08745198fed97aa22b191a2aa61d2cefb99de7594b8d3aceec",
        "cc34100e4aa0eaf70dc91e0b77cbd4843cd0d65233ee85e3d7f4e85d318a25e1",
        "402b8a3732d87b33e6acf6ea7ac357d8192740c477bbb5eac290357f955aff26",
        "9a4162f457f270197017d0ec305c6c76f4eaab47127d28cd501586455f59fef1",
        "b0eca7cf355f42f5a90aeb0175feb7395fddf91e4d02c2274b9211bdf09f1025"},
       {"p0", "p0", "p0,p1,p2", "p0,p1,p3", "p1", "p0,p1,p2,p3"}},
      {"jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-backward.mlir",
       2048,
       4096,
       false,
       3,
       {"bbddc0570461712a03c47ad9005bdffe0b6a4cb2bb9b3867906f5052a44e42da",
        "a7f6c3dedf98acd3098ee27ae350b186062d8641f1743190da0c17c7232ca88d",
        "f4065574218a062887a692c9bdbaae945c5d1818424c540d45247706cc367f49",
        "e3434569ffbd6288dc2cae99d2f38122baddb12589013307fe6f297aceea8fdd",
        "9a7596dd60d79efd5338d8a79ff4b08a6c7f71d39b8b7a12139c2ad171d4ef60", ""},
       {"1d92bac043f65dcc9bd0ce68b749e3f12c4b55227d0bfa9d637d0f449ec3d977",
        "3f4d6542eb0f8415523428c80e91eddb7ea852cb69a0b041aa298ac70e54c985",
        "f287dceb40d1523dd06b1bce7d4d7502c62ca0c7782dd6b1ca4688704d611de8",
        "4764abf32a66ed2208781e798b792197df47b5ba9af9d7d759ad9af5f3dac018",
        "cd2b45f73b977d0258420899923c80e038a52a1792f4a076a40367529b6711ec", ""},
       {"3290c72b40857b474228e51426ee17d5fb2eec392abc6eb76084f8883cba97c4",
        "05665ce8138b8112a59c5807e0f59a5e73dc9452a6eb34df9826284272e033af",
        "b3e8d81fb1694b8a474ef874c4642c9356fc8d3b5925f36b300fe0b808fa5f37",
        "031ffa1049ca2a3c95d222efc10a485470c56143b7c91a5ed2eaf830923961d9",
        "e3e4422fb331199ede3a78a72fc402538fba46f2b8251fb9d0679d71d8c75917",
        ""},
       {"p0", "p0", "p0,p1,p2", "p1", "p0,p1,p2,p3", ""}},
      {"jax-0.10.1-bf16-layer_norm_44d152ecc3e9ff18-composed.mlir",
       2048,
       4096,
       true,
       4,
       {"a7f6c3dedf98acd3098ee27ae350b186062d8641f1743190da0c17c7232ca88d",
        "5baade21272a35250fab8e64012a6f3bf4d617a77f14eca81eb0d11a2e348f69",
        "c709eb3af0a93256d5b76b94925c1e801582b4a26cbe0a9d29139c16685b23db",
        "a4dfd0f1e306a074f90fd811c30ebbafff10629118f48f3d606aaf96838ec0fa",
        "e3434569ffbd6288dc2cae99d2f38122baddb12589013307fe6f297aceea8fdd",
        "9a7596dd60d79efd5338d8a79ff4b08a6c7f71d39b8b7a12139c2ad171d4ef60"},
       {"1d92bac043f65dcc9bd0ce68b749e3f12c4b55227d0bfa9d637d0f449ec3d977",
        "3f4d6542eb0f8415523428c80e91eddb7ea852cb69a0b041aa298ac70e54c985",
        "f8e450703ba2b413689d264a5540531cb6becaf61ad710f1c8b2f37a151a5c7e",
        "b43694d152c34ab031548b757d6c6a48cb72390c768ab10b0bfbb1f493f9003d",
        "4764abf32a66ed2208781e798b792197df47b5ba9af9d7d759ad9af5f3dac018",
        "cd2b45f73b977d0258420899923c80e038a52a1792f4a076a40367529b6711ec"},
       {"afe90d94f6e03940b678df98a010ddb786badb660870b3ddd9c0f374b8e8fddf",
        "73e71124680600c64327470c8aca8fe3956b325490e54470284c3910ec424b75",
        "39d35d22b5285784d4f69bac0a965052b5b111298b7b4489a18a1c4403cd8902",
        "a5d458aaebcdf13f686dd6918abc4bf1a7e9b88432c3899e1daf33f07dc15861",
        "031ffa1049ca2a3c95d222efc10a485470c56143b7c91a5ed2eaf830923961d9",
        "e3e4422fb331199ede3a78a72fc402538fba46f2b8251fb9d0679d71d8c75917"},
       {"p0", "p0", "p0,p1,p2", "p0,p1,p3", "p1", "p0,p1,p2,p3"}},
  }};
  for (const Fixture &fixture : fixtures) {
    SCOPED_TRACE(fixture.filename);
    std::optional<std::string> source = fixtureSource(fixture.filename);
    ASSERT_TRUE(source);
    for (mlir::shuttle::NumericalPolicy policy :
         {mlir::shuttle::NumericalPolicy::SourceOrdered,
          mlir::shuttle::NumericalPolicy::Fast}) {
      SCOPED_TRACE(policy == mlir::shuttle::NumericalPolicy::SourceOrdered
                       ? "source_ordered"
                       : "fast");
      std::unique_ptr<BuiltModule> built =
          build(*source, policy, kFusionTuning, false, true);
      ASSERT_TRUE(built);
      ASSERT_TRUE(hasOrdinaryJaxLayerNormBCPartition(
          *built->module, fixture.rows, fixture.features, fixture.composed));

      mlir::FailureOr<llvm::SmallVector<uint8_t>> transport =
          mlir::shuttle::serializeGpuExecutableBundle(*built->module);
      ASSERT_TRUE(mlir::succeeded(transport));
      auto executable = mlir::shuttle::GpuExecutable::Load(*transport);
      ASSERT_TRUE(executable.ok()) << executable.status();
      EXPECT_EQ((*executable)->launches().size(), 2u);
      EXPECT_EQ((*executable)->slots().size(), fixture.composed ? 10u : 8u);
      const int64_t inputCount = fixture.composed ? 4 : 3;
      const int64_t matrixBytes = fixture.rows * fixture.features * 2;
      const int64_t featureBytes = fixture.features * 2;
      // External result indices are the public function result order, not the
      // materialization-buffer or task-output order.
      const llvm::SmallVector<int64_t> expectedBindingSlots =
          fixture.composed
              ? llvm::SmallVector<int64_t>{0, 1, 2, 3, 6, 9, 8, 7}
              : llvm::SmallVector<int64_t>{0, 1, 2, 7, 6, 5};
      const llvm::SmallVector<int64_t> expectedBindingBytes =
          fixture.composed
              ? llvm::SmallVector<int64_t>{matrixBytes, featureBytes,
                                           featureBytes, matrixBytes,
                                           matrixBytes, matrixBytes,
                                           featureBytes, featureBytes}
              : llvm::SmallVector<int64_t>{matrixBytes, featureBytes,
                                           matrixBytes, matrixBytes,
                                           featureBytes, featureBytes};
      ASSERT_EQ((*executable)->externalBindings().size(),
                expectedBindingSlots.size());
      for (auto [ordinal, slot, bytes] : llvm::enumerate(
               expectedBindingSlots, expectedBindingBytes)) {
        const bool operand = static_cast<int64_t>(ordinal) < inputCount;
        const int64_t index = operand ? static_cast<int64_t>(ordinal)
                                      : static_cast<int64_t>(ordinal) -
                                            inputCount;
        auto binding = llvm::find_if(
            (*executable)->externalBindings(), [&](const auto &candidate) {
              return candidate.kind ==
                         (operand
                              ? mlir::shuttle::ExecutableBindingKind::Operand
                              : mlir::shuttle::ExecutableBindingKind::Result) &&
                     candidate.index == index;
            });
        ASSERT_NE(binding, (*executable)->externalBindings().end());
        EXPECT_EQ(binding->kind,
                  operand ? mlir::shuttle::ExecutableBindingKind::Operand
                          : mlir::shuttle::ExecutableBindingKind::Result);
        EXPECT_EQ(binding->index, index);
        EXPECT_EQ(binding->slotOrdinal, slot);
        EXPECT_EQ(binding->requiredBytes, bytes);
        EXPECT_EQ(binding->alignment, 2);
        auto type =
            mlir::dyn_cast<mlir::RankedTensorType>(binding->tensorType);
        ASSERT_TRUE(type);
        EXPECT_TRUE(type.getElementType().isBF16());
      }
      const llvm::SmallVector<int64_t> expectedRowOutputs =
          fixture.composed ? llvm::SmallVector<int64_t>{4, 5, 6, 9}
                           : llvm::SmallVector<int64_t>{3, 4, 7};
      const llvm::SmallVector<int64_t> expectedFeatureOutputs =
          fixture.composed ? llvm::SmallVector<int64_t>{7, 8}
                           : llvm::SmallVector<int64_t>{5, 6};
      const auto &launches = (*executable)->launches();
      ASSERT_EQ(launches.size(), 2u);
      EXPECT_EQ(launches[0].outputSlots,
                std::vector<int64_t>(expectedRowOutputs.begin(),
                                     expectedRowOutputs.end()));
      EXPECT_EQ(launches[1].outputSlots,
                std::vector<int64_t>(expectedFeatureOutputs.begin(),
                                     expectedFeatureOutputs.end()));
      for (auto [launchOrdinal, launch] : llvm::enumerate(launches)) {
        llvm::SmallVector<ExpectedPtxOutput> expectedPtxOutputs;
        const int64_t outputOffset =
            launchOrdinal == 0 ? 0 : fixture.rowOutputCount;
        for (auto [outputOrdinal, slot] : llvm::enumerate(launch.outputSlots)) {
          auto type = mlir::cast<mlir::RankedTensorType>(
              (*executable)->slots()[slot].tensorType);
          const int64_t expectedOrdinal = outputOffset + outputOrdinal;
          expectedPtxOutputs.push_back(
              {type.getElementType().isBF16(),
               fixture.pointerDigests[expectedOrdinal],
               fixture.valueDigests[expectedOrdinal],
               fixture.controlDigests[expectedOrdinal],
               fixture.inputParameters[expectedOrdinal]});
        }
        llvm::StringRef ptx(
            reinterpret_cast<const char *>((*executable)->codeBytes().data() +
                                           launch.codeOffset),
            launch.codeLength);
        ASSERT_TRUE(hasExactMultiOutputStoreDataflow(
            ptx, launch.inputSlots.size(), expectedPtxOutputs));
        EXPECT_FALSE(hasExactMultiOutputStoreDataflow(
            misrouteFirstPtxOutput(ptx, launch.inputSlots.size()),
            launch.inputSlots.size(), expectedPtxOutputs));
        EXPECT_FALSE(hasExactMultiOutputStoreDataflow(
            zeroFirstPtxOutputValue(ptx), launch.inputSlots.size(),
            expectedPtxOutputs));
        EXPECT_FALSE(hasExactMultiOutputStoreDataflow(
            swapFirstTwoPtxOutputValues(ptx), launch.inputSlots.size(),
            expectedPtxOutputs));
        EXPECT_FALSE(hasExactMultiOutputStoreDataflow(
            rerouteFirstPtxInputLeaf(ptx), launch.inputSlots.size(),
            expectedPtxOutputs));
        EXPECT_FALSE(hasExactMultiOutputStoreDataflow(
            changeFirstPtxOutputByteScale(ptx), launch.inputSlots.size(),
            expectedPtxOutputs));
        EXPECT_FALSE(hasExactMultiOutputStoreDataflow(
            flipFirstPtxOutputGuardPolarity(ptx), launch.inputSlots.size(),
            expectedPtxOutputs));
        EXPECT_FALSE(hasExactMultiOutputStoreDataflow(
            redirectFirstPtxOutputGuard(ptx), launch.inputSlots.size(),
            expectedPtxOutputs));
      }
      int64_t temporarySlots = 0;
      int64_t temporaryBytes = 0;
      for (const mlir::shuttle::GpuSlot &slot : (*executable)->slots()) {
        if (slot.storage != mlir::shuttle::MaterializationStorage::Temporary)
          continue;
        ++temporarySlots;
        temporaryBytes += slot.requiredBytes;
      }
      EXPECT_EQ(temporarySlots, 2);
      EXPECT_EQ(temporaryBytes, fixture.rows * 2 * 4);

      mlir::PassManager replace(built->context.get());
      replace.addPass(
          mlir::shuttle::createReplaceWithGpuExecutableBundlePass());
      ASSERT_TRUE(mlir::succeeded(replace.run(*built->module)));
      llvm::SmallVector<mlir::Operation *> calls;
      built->module->walk([&](mlir::Operation *operation) {
        if (operation->getName().getStringRef() == "stablehlo.custom_call")
          calls.push_back(operation);
      });
      ASSERT_EQ(calls.size(), 1u);
      mlir::func::FuncOp function =
          *built->module->getOps<mlir::func::FuncOp>().begin();
      auto returnOp =
          mlir::cast<mlir::func::ReturnOp>(function.getBody().front().back());
      ASSERT_EQ(calls.front()->getNumResults(), returnOp.getNumOperands());
      for (auto [publicResult, returned] :
           llvm::enumerate(returnOp.getOperands())) {
        auto result = mlir::dyn_cast<mlir::OpResult>(returned);
        ASSERT_TRUE(result);
        EXPECT_EQ(result.getOwner(), calls.front());
        EXPECT_EQ(result.getResultNumber(), publicResult);
      }
    }
  }
}

TEST(Abi22MapFusionContract,
     PublicResultProjectionReplacesPartialMultiRegionGraph) {
  auto fixture = fixtureSource(
      "jax-0.10.1-bf16-matmul-vjp-17x33x19-composed.mlir");
  ASSERT_TRUE(fixture);
  constexpr llvm::StringLiteral originalConversion =
      "%1 = stablehlo.convert %0 : (tensor<17x19xf32>) -> "
      "tensor<17x19xbf16>";
  constexpr llvm::StringLiteral duplicatedConversion =
      "%1 = stablehlo.convert %0 : (tensor<17x19xf32>) -> "
      "tensor<17x19xbf16>\n"
      "    %12 = stablehlo.convert %0 : (tensor<17x19xf32>) -> "
      "tensor<17x19xbf16>";
  constexpr llvm::StringLiteral originalReturn =
      "return %1, %11, %7 : tensor<17x19xbf16>, tensor<17x33xbf16>, "
      "tensor<33x19xbf16>";
  constexpr llvm::StringLiteral expandedReturn =
      "return %12, %1, %11, %7 : tensor<17x19xbf16>, "
      "tensor<17x19xbf16>, tensor<17x33xbf16>, tensor<33x19xbf16>";
  constexpr llvm::StringLiteral originalResults =
      "-> (tensor<17x19xbf16> {jax.result_info = \"result[0]\"}, "
      "tensor<17x33xbf16> {jax.result_info = \"result[1]\"}, "
      "tensor<33x19xbf16> {jax.result_info = \"result[2]\"})";
  constexpr llvm::StringLiteral expandedResults =
      "-> (tensor<17x19xbf16> {jax.result_info = \"result[0]\"}, "
      "tensor<17x19xbf16> {jax.result_info = \"internal\"}, "
      "tensor<17x33xbf16> {jax.result_info = \"result[1]\"}, "
      "tensor<33x19xbf16> {jax.result_info = \"result[2]\"})";
  std::string source = *fixture;
  ASSERT_EQ(llvm::StringRef(source).count(originalConversion), 1u);
  ASSERT_EQ(llvm::StringRef(source).count(originalReturn), 1u);
  ASSERT_EQ(llvm::StringRef(source).count(originalResults), 1u);
  size_t position = source.find(originalResults.str());
  source.replace(position, originalResults.size(), expandedResults.str());
  position = source.find(originalConversion.str());
  source.replace(position, originalConversion.size(),
                 duplicatedConversion.str());
  position = source.find(originalReturn.str());
  source.replace(position, originalReturn.size(), expandedReturn.str());
  std::unique_ptr<BuiltModule> built =
      build(source, mlir::shuttle::NumericalPolicy::SourceOrdered,
            kFusionTuning, false, true);
  ASSERT_TRUE(built);
  EXPECT_EQ(llvm::range_size(
                built->module->getOps<mlir::shuttle::ExecutableBundleOp>()),
            2u);

  mlir::func::FuncOp function =
      *built->module->getOps<mlir::func::FuncOp>().begin();
  llvm::SmallVector<mlir::shuttle::RegionOp> regions;
  function.walk(
      [&](mlir::shuttle::RegionOp region) { regions.push_back(region); });
  ASSERT_EQ(regions.size(), 2u);
  ASSERT_EQ(regions[0].getNumResults(), 2u);
  ASSERT_EQ(function.getNumArguments(), 3u);
  int64_t cotangentOperand = -1;
  for (auto [index, input] : llvm::enumerate(regions[1].getInputs())) {
    if (input == function.getArgument(2))
      cotangentOperand = index;
  }
  ASSERT_GE(cotangentOperand, 0);
  regions[1]->setOperand(cotangentOperand, regions[0].getResult(0));
  auto originalReturnOp =
      mlir::cast<mlir::func::ReturnOp>(function.getBody().front().back());
  ASSERT_EQ(originalReturnOp.getNumOperands(), 4u);
  ASSERT_EQ(originalReturnOp.getOperand(0), regions[0].getResult(1));
  ASSERT_EQ(originalReturnOp.getOperand(1), regions[0].getResult(0));
  originalReturnOp->eraseOperands(1);
  llvm::SmallVector<mlir::Type> publicResultTypes;
  for (mlir::Value result : originalReturnOp.getOperands())
    publicResultTypes.push_back(result.getType());
  function.setType(mlir::FunctionType::get(
      built->context.get(), function.getArgumentTypes(), publicResultTypes));

  mlir::PassManager replace(built->context.get());
  replace.addPass(mlir::shuttle::createReplaceWithGpuExecutableBundlePass());
  ASSERT_TRUE(mlir::succeeded(replace.run(*built->module)));
  llvm::SmallVector<mlir::Operation *> shuttleCalls;
  built->module->walk([&](mlir::Operation *operation) {
    if (operation->getName().getStringRef() != "stablehlo.custom_call")
      return;
    auto target = operation->getAttrOfType<mlir::StringAttr>(
        "call_target_name");
    ASSERT_TRUE(target);
    if (target.getValue() == "shuttle.gpu.executable_bundle.v2")
      shuttleCalls.push_back(operation);
  });
  ASSERT_EQ(shuttleCalls.size(), 2u);
  ASSERT_EQ(shuttleCalls[0]->getNumResults(), 2u);
  ASSERT_EQ(shuttleCalls[1]->getNumResults(), 2u);
  int64_t interCallOperands = 0;
  for (mlir::Value operand : shuttleCalls[1]->getOperands()) {
    auto result = mlir::dyn_cast<mlir::OpResult>(operand);
    if (!result || result.getOwner() != shuttleCalls[0])
      continue;
    ++interCallOperands;
    EXPECT_EQ(result.getResultNumber(), 1u);
  }
  EXPECT_EQ(interCallOperands, 1);

  auto returnOp =
      mlir::cast<mlir::func::ReturnOp>(function.getBody().front().back());
  ASSERT_EQ(returnOp.getNumOperands(), 3u);
  auto first = mlir::cast<mlir::OpResult>(returnOp.getOperand(0));
  auto second = mlir::cast<mlir::OpResult>(returnOp.getOperand(1));
  auto third = mlir::cast<mlir::OpResult>(returnOp.getOperand(2));
  EXPECT_EQ(first.getOwner(), shuttleCalls[0]);
  EXPECT_EQ(first.getResultNumber(), 0u);
  EXPECT_EQ(second.getOwner(), shuttleCalls[1]);
  EXPECT_EQ(second.getResultNumber(), 0u);
  EXPECT_EQ(third.getOwner(), shuttleCalls[1]);
  EXPECT_EQ(third.getResultNumber(), 1u);
}

bool hasSourceOrderedContractRecurrence(llvm::StringRef ptx,
                                        int64_t reduction) {
  llvm::SmallVector<llvm::StringRef> lines;
  ptx.split(lines, '\n');
  for (llvm::StringRef &line : lines)
    line = line.trim();
  auto tokens = [](llvm::StringRef line) {
    std::vector<std::string> result;
    result.push_back(line.split(' ').first.str());
    llvm::append_range(result, ptxTokens(line));
    return result;
  };
  for (size_t loopLine = 0; loopLine + 3 < lines.size(); ++loopLine) {
    if (!lines[loopLine].ends_with(":"))
      continue;
    std::vector<std::string> comparison = tokens(lines[loopLine + 1]);
    const std::string boundSuffix =
        (llvm::Twine(", ") + llvm::Twine(reduction) + ";").str();
    if (comparison.size() != 3 || comparison[0] != "setp.ge.u64" ||
        !lines[loopLine + 1].ends_with(boundSuffix))
      continue;
    llvm::StringRef predicate = comparison[1];
    llvm::StringRef coordinate = comparison[2];
    const std::string branchPrefix =
        (llvm::Twine("@") + predicate + " bra ").str();
    if (!lines[loopLine + 2].starts_with(branchPrefix) ||
        !lines[loopLine + 2].ends_with(";"))
      continue;
    llvm::StringRef exit = lines[loopLine + 2]
                               .drop_front(branchPrefix.size())
                               .drop_back();
    llvm::StringRef loop = lines[loopLine].drop_back();
    const std::string increment =
        (llvm::Twine("add.u64 ") + coordinate + ", " + coordinate + ", 1;")
            .str();
    const std::string backedge = (llvm::Twine("bra ") + loop + ";").str();
    const std::string exitLabel = (llvm::Twine(exit) + ":").str();
    int64_t recurrences = 0;
    for (size_t line = loopLine + 3; line + 2 < lines.size(); ++line) {
      if (lines[line] == exitLabel)
        break;
      if (lines[line].starts_with("bra ") || lines[line].contains(" bra ") ||
          lines[line] == "ret;" || lines[line].contains(" ret;") ||
          lines[line] == "exit;" || lines[line].contains(" exit;") ||
          lines[line].ends_with(":"))
        return false;
      std::vector<std::string> instruction = tokens(lines[line]);
      if (instruction.size() != 5 || instruction[0] != "fma.rn.f32" ||
          instruction[1] != instruction[4])
        continue;
      PtxDataflow prefix = ptxDataflow(llvm::join(
          llvm::ArrayRef(lines).slice(loopLine + 3, line - loopLine - 3),
          "\n"));
      bool indexedInputs = true;
      for (llvm::StringRef operand : {instruction[2], instruction[3]}) {
        auto reaching = prefix.currentValues.find(operand.str());
        if (reaching == prefix.currentValues.end()) {
          indexedInputs = false;
          break;
        }
        std::set<std::string> closure =
            ptxDependencyClosure(prefix, reaching->second);
        indexedInputs &= closure.count(coordinate.str()) &&
                         closure.count("$ld.global.u16");
      }
      if (!indexedInputs)
        return false;
      ++recurrences;
      if (lines[line + 1] != increment || lines[line + 2] != backedge ||
          line + 3 >= lines.size() || lines[line + 3] != exitLabel)
        return false;
      return recurrences == 1;
    }
  }
  return false;
}

TEST(Abi22MapFusionContract,
     SourceOrderedContractRecurrenceBindsControlAndIndexedLoads) {
  constexpr llvm::StringLiteral recurrence = R"ptx(
  mov.u64 %rd4, 0;
  mov.f32 %f0, 0f00000000;
$loop_renamable:
  setp.ge.u64 %p0, %rd4, 65;
  @%p0 bra $finished_renamable;
  cvt.u64.u32 %rd5, %rd4;
  ld.global.u16 %h0, [%rd5];
  cvt.f32.bf16 %f1, %h0;
  mul.lo.u64 %rd6, %rd4, 2;
  ld.global.u16 %h1, [%rd6];
  cvt.f32.bf16 %f2, %h1;
  fma.rn.f32 %f0, %f1, %f2, %f0;
  add.u64 %rd4, %rd4, 1;
  bra $loop_renamable;
$finished_renamable:
)ptx";
  auto replaceAll = [](llvm::StringRef source, llvm::StringRef oldText,
                       llvm::StringRef newText) {
    std::string changed = source.str();
    size_t position = changed.find(oldText.str());
    EXPECT_NE(position, std::string::npos);
    while (position != std::string::npos) {
      changed.replace(position, oldText.size(), newText.str());
      position = changed.find(oldText.str(), position + newText.size());
    }
    return changed;
  };
  EXPECT_TRUE(hasSourceOrderedContractRecurrence(recurrence, 65));
  EXPECT_TRUE(hasSourceOrderedContractRecurrence(
      replaceAll(replaceAll(recurrence, "$loop_renamable", "$renamed_loop"),
                 "$finished_renamable", "$renamed_exit"),
      65));
  for (std::string mutation : {
           replaceAll(recurrence, "@%p0 bra $finished_renamable;",
                      "@!%p0 bra $finished_renamable;"),
           replaceAll(recurrence, "cvt.u64.u32 %rd5, %rd4;",
                      "cvt.u64.u32 %rd5, %rd9;"),
           replaceAll(recurrence, "add.u64 %rd4, %rd4, 1;", "nop;"),
           replaceAll(recurrence, "fma.rn.f32 %f0, %f1, %f2, %f0;",
                      "bra $finished_renamable;"),
       })
    EXPECT_FALSE(hasSourceOrderedContractRecurrence(mutation, 65));
}

TEST(Abi22MapFusionContract,
     NormContractCandidatesBindCostTopologyTransportAndPolicy) {
  struct Boundary {
    const char *filename;
    int64_t rows;
    int64_t reduction;
    int64_t columns;
  };
  constexpr Boundary boundaries[] = {
      {"jax-0.10.1-bf16-rmsnorm_contract-5x19x7-forward.mlir", 5, 19, 7},
      {"jax-0.10.1-bf16-rmsnorm_contract-5x65x7-forward.mlir", 5, 65, 7},
      {"jax-0.10.1-bf16-rmsnorm_contract-5x19x9-forward.mlir", 5, 19, 9},
      {"jax-0.10.1-bf16-rmsnorm_contract-5x19x257-forward.mlir", 5, 19,
       257},
      {"jax-0.10.1-bf16-rmsnorm_contract-3x2x5x11x263x521-forward.mlir",
       330, 263, 521},
  };
  for (const Boundary &boundary : boundaries) {
    SCOPED_TRACE(boundary.filename);
    std::optional<std::string> source = fixtureSource(boundary.filename);
    ASSERT_TRUE(source);
    std::optional<std::string> sourceOrderedFingerprint;
    for (mlir::shuttle::NumericalPolicy policy :
         {mlir::shuttle::NumericalPolicy::SourceOrdered,
          mlir::shuttle::NumericalPolicy::Fast}) {
      const int64_t outputCapacity =
          policy == mlir::shuttle::NumericalPolicy::Fast ? 8 : 256;
      const int64_t outputTile = std::min(boundary.columns, outputCapacity);
      const int64_t outputTiles =
          (boundary.columns + outputTile - 1) / outputTile;
      const bool inlineFold = outputTiles == 1;
      const int64_t expectedTasks = inlineFold ? 1 : 2;
      const int64_t expectedBuffers = inlineFold ? 4 : 5;
      const int64_t expectedTemporaryBytes = inlineFold ? 0 : boundary.rows * 4;

      std::unique_ptr<BuiltModule> built =
          build(*source, policy, kFusionTuning, false, true);
      ASSERT_TRUE(built);
      std::optional<PlanSummary> planSummary = summary(*built->module);
      ASSERT_TRUE(planSummary);
      EXPECT_TRUE(planSummary->hasContractRegion);
      EXPECT_TRUE(planSummary->exactSourceCoverage);
      EXPECT_EQ(planSummary->maps, 19);
      EXPECT_EQ(planSummary->folds, 1);
      EXPECT_EQ(planSummary->contracts, 1);
      EXPECT_EQ(planSummary->tasks, expectedTasks);
      EXPECT_EQ(planSummary->buffers, expectedBuffers);

      auto plan = *built->module->getOps<mlir::shuttle::MaterializationPlanOp>()
                       .begin();
      EXPECT_EQ(plan.getSchemaVersion(), 2);
      auto tasks =
          llvm::to_vector(plan.getBody()
                              .front()
                              .getOps<mlir::shuttle::MaterializationTaskOp>());
      auto buffers = llvm::to_vector(
          plan.getBody()
              .front()
              .getOps<mlir::shuttle::MaterializationBufferOp>());
      ASSERT_EQ(tasks.size(), expectedTasks);
      ASSERT_EQ(buffers.size(), expectedBuffers);
      auto contractTask = llvm::find_if(tasks, [](auto task) {
        return task.getKind() ==
               mlir::shuttle::MaterializationTaskKind::Contract;
      });
      ASSERT_NE(contractTask, tasks.end());
      mlir::shuttle::MaterializationTaskOp contract = *contractTask;
      EXPECT_EQ(contract.getInputBuffers(),
                inlineFold ? llvm::ArrayRef<int64_t>({0, 1, 2})
                           : llvm::ArrayRef<int64_t>({0, 3, 1, 2}));
      EXPECT_EQ(contract.getOutputBuffers(),
                llvm::ArrayRef<int64_t>({inlineFold ? 3 : 4}));
      EXPECT_EQ(contract.getDependencies(),
                inlineFold ? llvm::ArrayRef<int64_t>()
                           : llvm::ArrayRef<int64_t>({0}));

      auto strategy = contract->getAttrOfType<mlir::StringAttr>(
          "contract_region_strategy");
      auto sources = contract->getAttrOfType<mlir::ArrayAttr>(
          "contract_region_sources");
      auto foldSource =
          contract->getAttrOfType<mlir::shuttle::SourceRefAttr>(
              "contract_region_fold_source");
      auto contractSource =
          contract->getAttrOfType<mlir::shuttle::SourceRefAttr>(
              "contract_region_contract_source");
      auto boundarySource =
          contract->getAttrOfType<mlir::shuttle::SourceRefAttr>(
              "contract_region_bf16_boundary_source");
      auto epilogueSources = contract->getAttrOfType<mlir::ArrayAttr>(
          "contract_region_epilogue_sources");
      auto normalizedOperand = contract->getAttrOfType<mlir::IntegerAttr>(
          "contract_region_normalized_operand");
      auto cost = contract->getAttrOfType<mlir::DictionaryAttr>(
          "contract_region_cost");
      ASSERT_TRUE(strategy);
      ASSERT_TRUE(sources);
      ASSERT_TRUE(foldSource);
      ASSERT_TRUE(contractSource);
      ASSERT_TRUE(boundarySource);
      ASSERT_TRUE(epilogueSources);
      ASSERT_TRUE(normalizedOperand);
      ASSERT_TRUE(cost);
      EXPECT_EQ(strategy.getValue(), inlineFold ? "inline_fold" : "compact_stat");
      EXPECT_EQ(foldSource.getOperationOrdinal(), 3);
      EXPECT_EQ(contractSource.getOperationOrdinal(), 19);
      EXPECT_EQ(boundarySource.getOperationOrdinal(), 18);
      EXPECT_EQ(normalizedOperand.getInt(), 0);
      EXPECT_EQ(sourceOperationOrdinals(epilogueSources),
                llvm::ArrayRef<int64_t>({20}));
      EXPECT_EQ(sourceOperationOrdinals(sources),
                inlineFold
                    ? llvm::to_vector(llvm::seq<int64_t>(0, 21))
                    : llvm::SmallVector<int64_t>({0, 12, 13, 14, 15, 16, 17,
                                                  18, 19, 20}));
      auto expectCost = [&](llvm::StringRef field, int64_t expected) {
        auto value = cost.getAs<mlir::IntegerAttr>(field);
        ASSERT_TRUE(value) << field.str();
        EXPECT_EQ(value.getInt(), expected) << field.str();
      };
      expectCost("rows", boundary.rows);
      expectCost("reduction", boundary.reduction);
      expectCost("columns", boundary.columns);
      expectCost("output_tile", outputTile);
      expectCost("output_tiles", outputTiles);
      expectCost("duplicated_fold_elements",
                 boundary.rows * boundary.reduction * (outputTiles - 1));
      expectCost("normalized_temporary_bytes",
                 boundary.rows * boundary.reduction * 2);
      expectCost("contract_temporary_bytes",
                 boundary.rows * boundary.columns * 4);
      expectCost("compact_stat_bytes", expectedTemporaryBytes);

      int64_t actualTemporaryBytes = 0;
      for (auto buffer : buffers) {
        if (buffer.getStorage() ==
            mlir::shuttle::MaterializationStorage::Temporary)
          actualTemporaryBytes += materializationBytes(buffer);
      }
      EXPECT_EQ(actualTemporaryBytes, expectedTemporaryBytes);
      if (!inlineFold) {
        auto foldTask = llvm::find_if(tasks, [](auto task) {
          return task.getKind() ==
                 mlir::shuttle::MaterializationTaskKind::Fold;
        });
        ASSERT_NE(foldTask, tasks.end());
        mlir::shuttle::MaterializationTaskOp compactFold = *foldTask;
        EXPECT_EQ(compactFold.getInputBuffers(), llvm::ArrayRef<int64_t>({0}));
        EXPECT_EQ(compactFold.getOutputBuffers(), llvm::ArrayRef<int64_t>({3}));
        EXPECT_TRUE(compactFold.getDependencies().empty());
        auto foldSources = compactFold->getAttrOfType<mlir::ArrayAttr>(
            "fold_region_sources");
        ASSERT_TRUE(foldSources);
        EXPECT_EQ(sourceOperationOrdinals(foldSources),
                  llvm::to_vector(llvm::seq<int64_t>(0, 12)));
        auto foldRecomputed = compactFold->getAttrOfType<mlir::ArrayAttr>(
            "fold_region_recomputed_sources");
        auto contractRecomputed = contract->getAttrOfType<mlir::ArrayAttr>(
            "contract_region_recomputed_sources");
        ASSERT_TRUE(foldRecomputed);
        ASSERT_TRUE(contractRecomputed);
        EXPECT_EQ(sourceOperationOrdinals(foldRecomputed),
                  llvm::ArrayRef<int64_t>({0}));
        EXPECT_EQ(sourceOperationOrdinals(contractRecomputed),
                  llvm::ArrayRef<int64_t>({0}));
      }

      auto device =
          *built->module->getOps<mlir::shuttle::DeviceModuleOp>().begin();
      auto entries = llvm::to_vector(
          device.getBody().front().getOps<mlir::shuttle::DeviceEntryOp>());
      ASSERT_EQ(entries.size(), expectedTasks);
      mlir::shuttle::DeviceEntryOp contractEntry = entries.back();
      ASSERT_TRUE(contractEntry.getGrid());
      ASSERT_TRUE(contractEntry.getBlock());
      EXPECT_EQ(*contractEntry.getGrid(),
                llvm::ArrayRef<int64_t>({boundary.rows, outputTiles, 1}));
      const int64_t expectedContractThreads =
          policy == mlir::shuttle::NumericalPolicy::Fast
              ? outputTile * 32
              : ((outputTile + 31) / 32) * 32;
      EXPECT_EQ(*contractEntry.getBlock(),
                llvm::ArrayRef<int64_t>({expectedContractThreads, 1, 1}));
      EXPECT_EQ(*contractEntry.getDynamicSharedBytes(),
                inlineFold ? ((boundary.reduction + 31) / 32) * 32 * 4
                           : std::min<int64_t>(boundary.reduction, 256) * 2);
      if (!inlineFold) {
        ASSERT_TRUE(entries.front().getGrid());
        ASSERT_TRUE(entries.front().getBlock());
        EXPECT_EQ(*entries.front().getGrid(),
                  llvm::ArrayRef<int64_t>({boundary.rows, 1, 1}));
        const int64_t foldThreads =
            ((std::min<int64_t>(boundary.reduction, 256) + 31) / 32) * 32;
        EXPECT_EQ(*entries.front().getBlock(),
                  llvm::ArrayRef<int64_t>({foldThreads, 1, 1}));
        EXPECT_EQ(*entries.front().getDynamicSharedBytes(), foldThreads * 4);
      }

      llvm::StringRef contractPtx(
          reinterpret_cast<const char *>(device.getCode().data() +
                                         contractEntry.getCodeOffset()),
          contractEntry.getCodeLength());
      PtxDataflow contractFlow = ptxDataflow(contractPtx);
      ASSERT_EQ(contractFlow.stores.size(), 1u);
      std::set<std::string> contractValueDependencies = ptxDependencyClosure(
          contractFlow, contractFlow.stores.front().value);
      EXPECT_NE(contractPtx.find("cvt.rn.bf16.f32"), llvm::StringRef::npos);
      EXPECT_NE(contractPtx.find("cvt.f32.bf16"), llvm::StringRef::npos);
      EXPECT_NE(contractPtx.find("fma.rn.f32"), llvm::StringRef::npos);
      EXPECT_TRUE(contractValueDependencies.count("$op:cvt.rn.bf16.f32"));
      EXPECT_TRUE(contractValueDependencies.count("$op:cvt.f32.bf16"));
      EXPECT_TRUE(contractValueDependencies.count("$op:fma.rn.f32"));
      if (policy == mlir::shuttle::NumericalPolicy::SourceOrdered) {
        EXPECT_TRUE(
            hasSourceOrderedContractRecurrence(contractPtx, boundary.reduction));
        EXPECT_EQ(contractPtx.find("shfl.sync.down.b32"), llvm::StringRef::npos);
      } else {
        EXPECT_NE(contractPtx.find("shfl.sync.down.b32"),
                  llvm::StringRef::npos);
        EXPECT_FALSE(
            hasSourceOrderedContractRecurrence(contractPtx, boundary.reduction));
      }
      EXPECT_EQ(contractPtx.count("st.global.u16"), 1u);
      if (!inlineFold) {
        EXPECT_TRUE(contractValueDependencies.count("p1"));
        EXPECT_TRUE(contractValueDependencies.count("$op:ld.global.f32"));

        llvm::StringRef foldPtx(
            reinterpret_cast<const char *>(device.getCode().data() +
                                           entries.front().getCodeOffset()),
            entries.front().getCodeLength());
        PtxDataflow foldFlow = ptxDataflow(foldPtx);
        ASSERT_EQ(foldFlow.stores.size(), 1u);
        EXPECT_FALSE(foldFlow.stores.front().bf16);
        std::set<std::string> foldPointerDependencies = ptxDependencyClosure(
            foldFlow, foldFlow.stores.front().pointer);
        EXPECT_TRUE(foldPointerDependencies.count("p1"));
        llvm::StringRef reductionLiteral =
            boundary.reduction == 263 ? "0f43838000" : "0f41980000";
        EXPECT_TRUE(containsRmsNormReciprocalSubgraph(
            foldFlow, foldFlow.stores.front().value, reductionLiteral,
            "$shared"));
      }

      mlir::FailureOr<llvm::SmallVector<uint8_t>> transport =
          mlir::shuttle::serializeGpuExecutableBundle(*built->module);
      ASSERT_TRUE(mlir::succeeded(transport));
      auto executable = mlir::shuttle::GpuExecutable::Load(*transport);
      ASSERT_TRUE(executable.ok()) << executable.status();
      EXPECT_EQ((*executable)->launches().size(), expectedTasks);
      EXPECT_EQ((*executable)->slots().size(), expectedBuffers);
      int64_t transportedTemporarySlots = 0;
      int64_t transportedTemporaryBytes = 0;
      for (const mlir::shuttle::GpuSlot &slot : (*executable)->slots()) {
        if (slot.storage != mlir::shuttle::MaterializationStorage::Temporary)
          continue;
        ++transportedTemporarySlots;
        transportedTemporaryBytes += slot.requiredBytes;
      }
      EXPECT_EQ(transportedTemporarySlots, inlineFold ? 0 : 1);
      EXPECT_EQ(transportedTemporaryBytes, expectedTemporaryBytes);

      std::unique_ptr<BuiltModule> renamed =
          build(*source, policy, kFusionTuning, true, true);
      ASSERT_TRUE(renamed);
      auto renamedPlan =
          *renamed->module->getOps<mlir::shuttle::MaterializationPlanOp>()
               .begin();
      EXPECT_EQ(plan.getFingerprint(), renamedPlan.getFingerprint());
      if (sourceOrderedFingerprint)
        EXPECT_NE(*sourceOrderedFingerprint, plan.getFingerprint());
      else
        sourceOrderedFingerprint = plan.getFingerprint().str();
    }
  }
}

TEST(Abi22MapFusionContract,
     ComposedVjpReusesOneSpecializedPrimalBoundary) {
  struct Boundary {
    const char *filename;
    int64_t columns;
    int64_t expectedLaunches;
    int64_t expectedTemporaries;
    int64_t expectedTemporaryBytes;
  };
  constexpr Boundary boundaries[] = {
      {"jax-0.10.1-bf16-rmsnorm_contract-5x19x7-composed.mlir", 7,
       39, 36, 6996},
      {"jax-0.10.1-bf16-rmsnorm_contract-5x19x257-composed.mlir", 257,
       40, 37, 79016},
  };
  for (const Boundary &boundary : boundaries) {
    SCOPED_TRACE(boundary.filename);
    std::optional<std::string> source = fixtureSource(boundary.filename);
    ASSERT_TRUE(source);
    for (mlir::shuttle::NumericalPolicy policy :
         {mlir::shuttle::NumericalPolicy::SourceOrdered,
          mlir::shuttle::NumericalPolicy::Fast}) {
      SCOPED_TRACE(mlir::shuttle::stringifyNumericalPolicy(policy).str());
      std::unique_ptr<BuiltModule> built =
          build(*source, policy, kFusionTuning, false, true);
      ASSERT_TRUE(built);
      std::optional<PlanSummary> planSummary = summary(*built->module);
      ASSERT_TRUE(planSummary);
      EXPECT_TRUE(planSummary->hasContractRegion);
      EXPECT_EQ(planSummary->hasFoldRegion, boundary.columns == 257);
      EXPECT_TRUE(planSummary->exactSourceCoverage);

      auto plan = *built->module->getOps<mlir::shuttle::MaterializationPlanOp>()
                       .begin();
      auto tasks = llvm::to_vector(
          plan.getBody()
              .front()
              .getOps<mlir::shuttle::MaterializationTaskOp>());
      auto buffers = llvm::to_vector(
          plan.getBody()
              .front()
              .getOps<mlir::shuttle::MaterializationBufferOp>());
      EXPECT_LT(tasks.size(), 42u);
      auto specialized = llvm::find_if(tasks, [](auto task) {
        return static_cast<bool>(task->template getAttrOfType<mlir::ArrayAttr>(
            "contract_region_sources"));
      });
      ASSERT_NE(specialized, tasks.end());
      EXPECT_EQ(llvm::count_if(tasks, [](auto task) {
                  return static_cast<bool>(
                      task->template getAttrOfType<mlir::ArrayAttr>(
                          "contract_region_sources"));
                }),
                1);
      mlir::shuttle::MaterializationTaskOp specializedTask = *specialized;
      EXPECT_EQ(specializedTask.getKind(),
                mlir::shuttle::MaterializationTaskKind::Contract);
      auto sources = specializedTask->getAttrOfType<mlir::ArrayAttr>(
          "contract_region_sources");
      ASSERT_TRUE(sources);
      llvm::SmallVector<int64_t> expectedPrimalSources;
      if (boundary.columns == 7) {
        for (int64_t ordinal = 0; ordinal <= 11; ++ordinal)
          expectedPrimalSources.push_back(ordinal);
        for (int64_t ordinal = 16; ordinal <= 24; ++ordinal)
          expectedPrimalSources.push_back(ordinal);
      } else {
        expectedPrimalSources.push_back(0);
        for (int64_t ordinal = 16; ordinal <= 24; ++ordinal)
          expectedPrimalSources.push_back(ordinal);
      }
      EXPECT_EQ(sourceOperationOrdinals(sources), expectedPrimalSources);
      auto foldSource =
          specializedTask->getAttrOfType<mlir::shuttle::SourceRefAttr>(
              "contract_region_fold_source");
      auto contractSource =
          specializedTask->getAttrOfType<mlir::shuttle::SourceRefAttr>(
              "contract_region_contract_source");
      auto boundarySource =
          specializedTask->getAttrOfType<mlir::shuttle::SourceRefAttr>(
              "contract_region_bf16_boundary_source");
      auto epilogueSources = specializedTask->getAttrOfType<mlir::ArrayAttr>(
          "contract_region_epilogue_sources");
      ASSERT_TRUE(foldSource);
      ASSERT_TRUE(contractSource);
      ASSERT_TRUE(boundarySource);
      ASSERT_TRUE(epilogueSources);
      EXPECT_EQ(foldSource.getOperationOrdinal(), 3);
      EXPECT_EQ(boundarySource.getOperationOrdinal(), 22);
      EXPECT_EQ(contractSource.getOperationOrdinal(), 23);
      EXPECT_EQ(sourceOperationOrdinals(epilogueSources),
                llvm::ArrayRef<int64_t>({24}));

      std::optional<mlir::shuttle::MaterializationBufferOp> normalized;
      std::optional<mlir::shuttle::MaterializationBufferOp> primalResult;
      for (int64_t output : specializedTask.getOutputBuffers()) {
        auto found = llvm::find_if(buffers, [&](auto buffer) {
          return buffer.getOrdinal() == output;
        });
        ASSERT_NE(found, buffers.end());
        auto type =
            mlir::dyn_cast<mlir::RankedTensorType>(found->getTensorType());
        ASSERT_TRUE(type);
        if (type.getShape() == llvm::ArrayRef<int64_t>({5, 19}) &&
            type.getElementType().isBF16() && !found->getLiveOut()) {
          ASSERT_FALSE(normalized);
          normalized = *found;
        }
        if (type.getShape() ==
                llvm::ArrayRef<int64_t>({5, boundary.columns}) &&
            type.getElementType().isBF16() && found->getLiveOut()) {
          ASSERT_FALSE(primalResult);
          primalResult = *found;
        }
      }
      ASSERT_TRUE(normalized);
      ASSERT_TRUE(primalResult);
      EXPECT_EQ(specializedTask.getOutputBuffers().size(), 2);
      EXPECT_EQ(normalized->getStorage(),
                mlir::shuttle::MaterializationStorage::Temporary);
      ASSERT_TRUE(normalized->getProducer());
      EXPECT_EQ(*normalized->getProducer(), specializedTask.getOrdinal());
      ASSERT_EQ(normalized->getConsumers().size(), 1u);
      const int64_t conversionOrdinal = normalized->getConsumers().front();
      ASSERT_GE(conversionOrdinal, 0);
      ASSERT_LT(conversionOrdinal, static_cast<int64_t>(tasks.size()));
      auto conversion = tasks[conversionOrdinal];
      EXPECT_EQ(conversion.getKind(),
                mlir::shuttle::MaterializationTaskKind::Map);
      ASSERT_EQ(conversion.getOutputBuffers().size(), 1u);
      auto converted = llvm::find_if(buffers, [&](auto buffer) {
        return buffer.getOrdinal() == conversion.getOutputBuffers().front();
      });
      ASSERT_NE(converted, buffers.end());
      EXPECT_EQ(llvm::count_if(tasks, [&](auto task) {
                  return task.getKind() ==
                             mlir::shuttle::MaterializationTaskKind::Contract &&
                         llvm::is_contained(task.getInputBuffers(),
                                            converted->getOrdinal());
                }),
                1);

      mlir::FailureOr<llvm::SmallVector<uint8_t>> transport =
          mlir::shuttle::serializeGpuExecutableBundle(*built->module);
      ASSERT_TRUE(mlir::succeeded(transport));
      auto executable = mlir::shuttle::GpuExecutable::Load(*transport);
      ASSERT_TRUE(executable.ok()) << executable.status();
      ASSERT_EQ((*executable)->launches().size(),
                static_cast<size_t>(boundary.expectedLaunches));
      EXPECT_EQ(tasks.size(), static_cast<size_t>(boundary.expectedLaunches));
      int64_t temporaryCount = 0;
      int64_t temporaryBytes = 0;
      for (const mlir::shuttle::GpuSlot &slot : (*executable)->slots()) {
        if (slot.storage != mlir::shuttle::MaterializationStorage::Temporary)
          continue;
        ++temporaryCount;
        temporaryBytes += slot.requiredBytes;
      }
      EXPECT_EQ(temporaryCount, boundary.expectedTemporaries);
      EXPECT_EQ(temporaryBytes, boundary.expectedTemporaryBytes);
      auto device =
          *built->module->getOps<mlir::shuttle::DeviceModuleOp>().begin();
      auto entries = llvm::to_vector(
          device.getBody().front().getOps<mlir::shuttle::DeviceEntryOp>());
      ASSERT_EQ(entries.size(), tasks.size());
      mlir::shuttle::DeviceEntryOp specializedEntry =
          entries[specializedTask.getOrdinal()];
      llvm::StringRef specializedPtx(
          reinterpret_cast<const char *>(device.getCode().data() +
                                         specializedEntry.getCodeOffset()),
          specializedEntry.getCodeLength());
      PtxDataflow specializedFlow = ptxDataflow(specializedPtx);
      ASSERT_EQ(specializedFlow.stores.size(), 2u);
      EXPECT_TRUE(specializedFlow.stores.front().bf16);
      EXPECT_TRUE(specializedFlow.stores.back().bf16);
      EXPECT_EQ(specializedPtx.count("st.global.u16"), 2u);
      const auto &specializedLaunch =
          (*executable)->launches()[specializedTask.getOrdinal()];
      EXPECT_EQ(specializedLaunch.outputSlots.size(), 2u);
      EXPECT_EQ(specializedLaunch.outputSlots.front(),
                normalized->getOrdinal());
      EXPECT_EQ(specializedLaunch.outputSlots.back(),
                primalResult->getOrdinal());
    }
  }
}

TEST(Abi22MapFusionContract,
     AnonymousContractFoldCompositionOwnsTheFullAndCompactSsaCut) {
  struct Fixture {
    const char *filename;
    llvm::SmallVector<int64_t> firstSources;
    llvm::SmallVector<int64_t> secondSources;
    llvm::SmallVector<int64_t> compactShape;
    llvm::SmallVector<int64_t> fullContractShape;
    llvm::SmallVector<int64_t> publicShape;
    std::array<llvm::SmallVector<int64_t>, 4> publicInputShapes;
  };
  const Fixture fixtures[] = {
      {"abi22-generic-contract-fold-composition-rank2.mlir",
       {0, 1},
       {2, 3},
       {2},
       {2, 5},
       {2, 7},
       {{{2, 3}, {3, 5}, {5, 7}, {}}}},
      {"abi22-generic-contract-fold-composition-rank3.mlir",
      {4, 6},
      {9, 13},
      {2, 3},
       {2, 3, 269},
      {2, 3, 13},
       {{{2, 3, 4}, {4, 269}, {269, 13}, {}}}},
  };
  for (const Fixture &fixture : fixtures) {
    SCOPED_TRACE(fixture.filename);
    std::optional<std::string> source = fixtureSource(fixture.filename);
    ASSERT_TRUE(source);
    for (mlir::shuttle::NumericalPolicy policy :
         {mlir::shuttle::NumericalPolicy::SourceOrdered,
          mlir::shuttle::NumericalPolicy::Fast}) {
      SCOPED_TRACE(mlir::shuttle::stringifyNumericalPolicy(policy).str());
      for (bool renamed : {false, true}) {
        SCOPED_TRACE(renamed ? "renamed" : "original");
        std::unique_ptr<BuiltModule> built =
            buildAlgebra(*source, policy, false, renamed);
        ASSERT_TRUE(built);

        auto plans = llvm::to_vector(
            built->module->getOps<mlir::shuttle::MaterializationPlanOp>());
        ASSERT_EQ(plans.size(), 1u);
        auto tasks = llvm::to_vector(
            plans.front()
                .getBody()
                .front()
                .getOps<mlir::shuttle::MaterializationTaskOp>());
        auto buffers = llvm::to_vector(
            plans.front()
                .getBody()
                .front()
                .getOps<mlir::shuttle::MaterializationBufferOp>());
        auto first = llvm::find_if(tasks, [](auto task) {
          auto kind = task->template getAttrOfType<mlir::StringAttr>(
              "composition_kind");
          return kind && kind.getValue() == "contract_fold";
        });
        auto second = llvm::find_if(tasks, [](auto task) {
          auto kind = task->template getAttrOfType<mlir::StringAttr>(
              "composition_kind");
          return kind && kind.getValue() == "map_contract";
        });
        ASSERT_NE(first, tasks.end());
        ASSERT_NE(second, tasks.end());
        EXPECT_EQ(
            llvm::count_if(tasks,
                           [](auto task) {
                             return static_cast<bool>(
                                 task->template getAttrOfType<mlir::StringAttr>(
                                     "composition_kind"));
                           }),
            2);
        EXPECT_EQ(first->getKind(),
                  mlir::shuttle::MaterializationTaskKind::Fold);
        EXPECT_EQ(second->getKind(),
                  mlir::shuttle::MaterializationTaskKind::Contract);

        auto requireSources = [&](auto task, llvm::StringRef name,
                                  llvm::ArrayRef<int64_t> expected) {
          auto sources = task->template getAttrOfType<mlir::ArrayAttr>(name);
          ASSERT_TRUE(sources) << name.str();
          EXPECT_EQ(sourceOperationOrdinals(sources), expected) << name.str();
        };
        requireSources(*first, "composition_sources", fixture.firstSources);
        requireSources(*second, "composition_sources", fixture.secondSources);

        auto hasShape = [](mlir::shuttle::MaterializationBufferOp buffer,
                           llvm::ArrayRef<int64_t> shape) {
          auto type =
              mlir::dyn_cast<mlir::RankedTensorType>(buffer.getTensorType());
          return type && type.getShape() == shape &&
                 type.getElementType().isF32();
        };
        auto compact = llvm::find_if(buffers, [&](auto buffer) {
          return hasShape(buffer, fixture.compactShape) &&
                 buffer.getStorage() ==
                     mlir::shuttle::MaterializationStorage::Temporary;
        });
        ASSERT_NE(compact, buffers.end());
        auto fullContract = llvm::find_if(buffers, [&](auto buffer) {
          return hasShape(buffer, fixture.fullContractShape) &&
                 buffer.getStorage() ==
                     mlir::shuttle::MaterializationStorage::Temporary;
        });
        ASSERT_NE(fullContract, buffers.end());
        EXPECT_EQ(llvm::count_if(
                      buffers,
                      [&](auto buffer) {
                        return buffer.getStorage() ==
                               mlir::shuttle::MaterializationStorage::Temporary;
                      }),
                  2);
        for (auto boundary : {*fullContract, *compact}) {
          EXPECT_FALSE(boundary.getLiveIn());
          EXPECT_FALSE(boundary.getLiveOut());
          EXPECT_EQ(boundary.getProducer(), first->getOrdinal());
          EXPECT_EQ(boundary.getConsumers(),
                    llvm::ArrayRef<int64_t>(
                        {static_cast<int64_t>(second->getOrdinal())}));
          EXPECT_EQ(boundary.getLifetimeStart(), first->getOrdinal());
          EXPECT_EQ(boundary.getLifetimeEnd(), second->getOrdinal());
          EXPECT_TRUE(llvm::is_contained(first->getOutputBuffers(),
                                         boundary.getOrdinal()));
          EXPECT_TRUE(llvm::is_contained(second->getInputBuffers(),
                                         boundary.getOrdinal()));
        }
        ASSERT_EQ(first->getOutputBuffers().size(), 2u);
        EXPECT_EQ(first->getOutputBuffers().front(),
                  fullContract->getOrdinal());
        EXPECT_EQ(first->getOutputBuffers().back(), compact->getOrdinal());

        auto publicResult = llvm::find_if(buffers, [&](auto buffer) {
          return hasShape(buffer, fixture.publicShape) && buffer.getLiveOut();
        });
        ASSERT_NE(publicResult, buffers.end());
        EXPECT_FALSE(publicResult->getLiveIn());
        EXPECT_EQ(publicResult->getStorage(),
                  mlir::shuttle::MaterializationStorage::External);
        EXPECT_EQ(publicResult->getProducer(), second->getOrdinal());
        EXPECT_TRUE(llvm::is_contained(second->getOutputBuffers(),
                                       publicResult->getOrdinal()));
        EXPECT_EQ(llvm::count_if(
                      buffers, [](auto buffer) { return buffer.getLiveIn(); }),
                  4);
        EXPECT_EQ(llvm::count_if(
                      buffers, [](auto buffer) { return buffer.getLiveOut(); }),
                  1);

        llvm::SmallVector<int64_t> publicInputBuffers;
        for (llvm::ArrayRef<int64_t> inputShape : fixture.publicInputShapes) {
          auto input = llvm::find_if(buffers, [&](auto buffer) {
            return buffer.getLiveIn() && !buffer.getLiveOut() &&
                   buffer.getStorage() ==
                       mlir::shuttle::MaterializationStorage::External &&
                   hasShape(buffer, inputShape);
          });
          ASSERT_NE(input, buffers.end());
          publicInputBuffers.push_back(input->getOrdinal());
        }
        for (int64_t index : {0, 1, 3})
          EXPECT_TRUE(llvm::is_contained(first->getInputBuffers(),
                                         publicInputBuffers[index]));
        EXPECT_FALSE(llvm::is_contained(first->getInputBuffers(),
                                        publicInputBuffers[2]));
        EXPECT_TRUE(llvm::is_contained(second->getInputBuffers(),
                                       publicInputBuffers[2]));
        for (int64_t index : {0, 1, 3})
          EXPECT_FALSE(llvm::is_contained(second->getInputBuffers(),
                                          publicInputBuffers[index]));

        auto cost =
            (*second)->getAttrOfType<mlir::DictionaryAttr>("composition_cost");
        ASSERT_TRUE(cost);
        auto integer = [&](llvm::StringRef name) {
          auto value = cost.getAs<mlir::IntegerAttr>(name);
          EXPECT_TRUE(value) << name.str();
          return value ? value.getInt() : int64_t{-1};
        };
        const int64_t baselineLaunches = integer("baseline_launches");
        const int64_t candidateLaunches = integer("candidate_launches");
        const int64_t baselineTemporaryBytes =
            integer("baseline_temporary_bytes");
        const int64_t candidateTemporaryBytes =
            integer("candidate_temporary_bytes");
        const int64_t baselineTemporaryTraffic =
            integer("baseline_temporary_traffic_bytes");
        const int64_t candidateTemporaryTraffic =
            integer("candidate_temporary_traffic_bytes");
        EXPECT_LE(candidateLaunches, baselineLaunches);
        EXPECT_LE(candidateTemporaryBytes, baselineTemporaryBytes);
        EXPECT_LE(candidateTemporaryTraffic, baselineTemporaryTraffic);
        EXPECT_TRUE(candidateLaunches < baselineLaunches ||
                    candidateTemporaryBytes < baselineTemporaryBytes ||
                    candidateTemporaryTraffic < baselineTemporaryTraffic);
        EXPECT_EQ(integer("contract_evaluations"), 1);

        std::unique_ptr<BuiltModule> executableBuilt =
            buildAlgebra(*source, policy, true, renamed);
        ASSERT_TRUE(executableBuilt);
        auto device = *executableBuilt->module
                           ->getOps<mlir::shuttle::DeviceModuleOp>()
                           .begin();
        auto entries = llvm::to_vector(
            device.getBody().front().getOps<mlir::shuttle::DeviceEntryOp>());
        ASSERT_GT(entries.size(),
                  static_cast<size_t>(second->getOrdinal()));
        auto entryPtx = [&](int64_t ordinal) {
          mlir::shuttle::DeviceEntryOp entry = entries[ordinal];
          return llvm::StringRef(
              reinterpret_cast<const char *>(device.getCode().data() +
                                               entry.getCodeOffset()),
              entry.getCodeLength());
        };
        llvm::StringRef contractFoldPtx = entryPtx(first->getOrdinal());
        llvm::StringRef mapContractPtx = entryPtx(second->getOrdinal());
        EXPECT_EQ(contractFoldPtx.count(".param .u64 p"), 5u);
        EXPECT_EQ(contractFoldPtx.count("st.global.f32"), 2u);
        EXPECT_TRUE(contractFoldPtx.contains("[p0]"));
        EXPECT_TRUE(contractFoldPtx.contains("[p1]"));
        EXPECT_TRUE(contractFoldPtx.contains("[p2]"));
        EXPECT_TRUE(contractFoldPtx.contains("[p3]"));
        EXPECT_TRUE(contractFoldPtx.contains("[p4]"));
        EXPECT_TRUE(contractFoldPtx.contains("fma.rn.f32"));
        EXPECT_TRUE(contractFoldPtx.contains("st.shared.f32"));
        EXPECT_TRUE(contractFoldPtx.contains("ld.shared.f32"));
        EXPECT_FALSE(contractFoldPtx.contains("add.u32 %r4, %r4, %r2"));
        EXPECT_TRUE(contractFoldPtx.contains("add.u64"));
        EXPECT_EQ(mapContractPtx.count(".param .u64 p"), 4u);
        EXPECT_EQ(mapContractPtx.count("st.global.f32"), 1u);
        EXPECT_TRUE(mapContractPtx.contains("[p0]"));
        EXPECT_TRUE(mapContractPtx.contains("[p1]"));
        EXPECT_TRUE(mapContractPtx.contains("[p2]"));
        EXPECT_TRUE(mapContractPtx.contains("[p3]"));
        EXPECT_TRUE(mapContractPtx.contains("mul.rn.f32"));
        EXPECT_TRUE(mapContractPtx.contains("fma.rn.f32"));
        PtxDataflow contractFoldFlow = ptxDataflow(contractFoldPtx);
        PtxDataflow mapContractFlow = ptxDataflow(mapContractPtx);
        ASSERT_EQ(contractFoldFlow.stores.size(), 2u);
        ASSERT_EQ(mapContractFlow.stores.size(), 1u);
        std::set<std::string> compactClosure = ptxDependencyClosure(
            contractFoldFlow, contractFoldFlow.stores.back().value);
        EXPECT_TRUE(compactClosure.count("p0"));
        EXPECT_TRUE(compactClosure.count("p1"));
        EXPECT_TRUE(compactClosure.count("p2"));
        EXPECT_TRUE(compactClosure.count("$shared"));
        EXPECT_TRUE(compactClosure.count("$op:fma.rn.f32"));
        std::set<std::string> resultClosure = ptxDependencyClosure(
            mapContractFlow, mapContractFlow.stores.front().value);
        EXPECT_TRUE(resultClosure.count("p0"));
        EXPECT_TRUE(resultClosure.count("p1"));
        EXPECT_TRUE(resultClosure.count("p2"));
        EXPECT_TRUE(resultClosure.count("$op:mul.rn.f32"));
        EXPECT_TRUE(resultClosure.count("$op:fma.rn.f32"));
        mlir::FailureOr<llvm::SmallVector<uint8_t>> transport =
            mlir::shuttle::serializeGpuExecutableBundle(
                *executableBuilt->module);
        ASSERT_TRUE(mlir::succeeded(transport));
        auto executable = mlir::shuttle::GpuExecutable::Load(*transport);
        ASSERT_TRUE(executable.ok()) << executable.status();
        ASSERT_GT((*executable)->launches().size(),
                  static_cast<size_t>(second->getOrdinal()));
        auto transportedCompact = llvm::find_if(
            (*executable)->slots(), [&](const mlir::shuttle::GpuSlot &slot) {
              auto type =
                  mlir::dyn_cast<mlir::RankedTensorType>(slot.tensorType);
              return slot.storage ==
                         mlir::shuttle::MaterializationStorage::Temporary &&
                     type &&
                     type.getShape() ==
                         llvm::ArrayRef<int64_t>(fixture.compactShape) &&
                     type.getElementType().isF32();
            });
        ASSERT_NE(transportedCompact, (*executable)->slots().end());
        auto transportedFull = llvm::find_if(
            (*executable)->slots(), [&](const mlir::shuttle::GpuSlot &slot) {
              auto type =
                  mlir::dyn_cast<mlir::RankedTensorType>(slot.tensorType);
              return slot.storage ==
                         mlir::shuttle::MaterializationStorage::Temporary &&
                     type &&
                     type.getShape() ==
                         llvm::ArrayRef<int64_t>(fixture.fullContractShape) &&
                     type.getElementType().isF32();
            });
        ASSERT_NE(transportedFull, (*executable)->slots().end());
        llvm::SmallVector<int64_t> transportedInputSlots;
        for (auto [index, inputShape] :
             llvm::enumerate(fixture.publicInputShapes)) {
          auto binding = llvm::find_if(
              (*executable)->externalBindings(), [&](const auto &candidate) {
                if (candidate.kind !=
                        mlir::shuttle::ExecutableBindingKind::Operand ||
                    candidate.index != static_cast<int64_t>(index))
                  return false;
                auto type = mlir::dyn_cast<mlir::RankedTensorType>(
                    candidate.tensorType);
                return type &&
                       type.getShape() == llvm::ArrayRef<int64_t>(inputShape) &&
                       type.getElementType().isF32();
              });
          ASSERT_NE(binding, (*executable)->externalBindings().end());
          transportedInputSlots.push_back(binding->slotOrdinal);
        }
        auto resultBinding = llvm::find_if(
            (*executable)->externalBindings(), [&](const auto &candidate) {
              if (candidate.kind !=
                      mlir::shuttle::ExecutableBindingKind::Result ||
                  candidate.index != 0)
                return false;
              auto type =
                  mlir::dyn_cast<mlir::RankedTensorType>(candidate.tensorType);
              return type &&
                     type.getShape() ==
                         llvm::ArrayRef<int64_t>(fixture.publicShape) &&
                     type.getElementType().isF32();
            });
        ASSERT_NE(resultBinding, (*executable)->externalBindings().end());
        EXPECT_EQ((*executable)->externalBindings().size(), 5u);
        for (int64_t index : {0, 1, 3})
          EXPECT_TRUE(llvm::is_contained(
              (*executable)->launches()[first->getOrdinal()].inputSlots,
              transportedInputSlots[index]));
        EXPECT_TRUE(llvm::is_contained(
            (*executable)->launches()[second->getOrdinal()].inputSlots,
            transportedInputSlots[2]));
        EXPECT_TRUE(llvm::is_contained(
            (*executable)->launches()[second->getOrdinal()].outputSlots,
            resultBinding->slotOrdinal));
        for (int64_t ordinal :
             {transportedFull->ordinal, transportedCompact->ordinal}) {
          EXPECT_TRUE(llvm::is_contained(
              (*executable)->launches()[first->getOrdinal()].outputSlots,
              ordinal));
          EXPECT_TRUE(llvm::is_contained(
              (*executable)->launches()[second->getOrdinal()].inputSlots,
              ordinal));
        }
        EXPECT_TRUE(llvm::is_contained(
            (*executable)->launches()[second->getOrdinal()].dependencies,
            first->getOrdinal()));
      }
    }
  }
}

TEST(Abi22MapFusionContract,
     GenericContractFoldCompositionRejectsBoundaryAndOwnershipCorruption) {
  std::optional<std::string> source =
      fixtureSource("abi22-generic-contract-fold-composition-rank2.mlir");
  ASSERT_TRUE(source);
  for (llvm::StringRef mutation : {
           "output-order",
           "missing-output",
           "compact-type",
           "compact-element-type",
           "extra-public-result",
           "full-live-out",
           "boundary-storage",
           "boundary-producer",
           "lifetime",
           "missing-consumer",
           "missing-second-input",
           "missing-public-input",
           "rebound-public-input",
           "missing-second-public-input",
           "rebound-second-public-input",
           "second-input-order",
           "missing-public-result",
           "rebound-public-result",
           "missing-kind",
           "unknown-kind",
           "source-ownership",
           "duplicate-source-ownership",
           "cost",
           "underreported-cost",
           "missing-cost-field",
           "contract-evaluations",
       }) {
    SCOPED_TRACE(mutation.str());
    std::unique_ptr<BuiltModule> built =
        buildAlgebra(*source, mlir::shuttle::NumericalPolicy::SourceOrdered,
                     /*executable=*/false);
    ASSERT_TRUE(built);
    ASSERT_TRUE(mutateGenericContractFoldComposition(*built->module, mutation));
    EXPECT_FALSE(verifyMaterializationPlan(*built->module));
  }
}

TEST(Abi22MapFusionContract,
     HigherRankHybridPreservesTheNormalizedBoundaryShape) {
  std::optional<std::string> source = fixtureSource(
      "jax-0.10.1-bf16-rmsnorm_contract-2x5x19x7-boundary-consumer.mlir");
  ASSERT_TRUE(source);
  for (mlir::shuttle::NumericalPolicy policy :
       {mlir::shuttle::NumericalPolicy::SourceOrdered,
        mlir::shuttle::NumericalPolicy::Fast}) {
    SCOPED_TRACE(mlir::shuttle::stringifyNumericalPolicy(policy).str());
    std::unique_ptr<BuiltModule> built =
        build(*source, policy, kFusionTuning, false, true);
    ASSERT_TRUE(built);
    std::optional<PlanSummary> planSummary = summary(*built->module);
    ASSERT_TRUE(planSummary);
    EXPECT_TRUE(planSummary->hasContractRegion);
    EXPECT_FALSE(planSummary->hasFoldRegion);
    EXPECT_TRUE(planSummary->exactSourceCoverage);

    auto plan =
        *built->module->getOps<mlir::shuttle::MaterializationPlanOp>().begin();
    auto tasks = llvm::to_vector(
        plan.getBody().front().getOps<mlir::shuttle::MaterializationTaskOp>());
    auto buffers =
        llvm::to_vector(plan.getBody()
                            .front()
                            .getOps<mlir::shuttle::MaterializationBufferOp>());
    auto specialized = llvm::find_if(tasks, [](auto task) {
      return static_cast<bool>(task->template getAttrOfType<mlir::ArrayAttr>(
          "contract_region_sources"));
    });
    ASSERT_NE(specialized, tasks.end());
    ASSERT_EQ(specialized->getOutputBuffers().size(), 2u);
    auto normalized = llvm::find_if(buffers, [&](auto buffer) {
      return buffer.getOrdinal() == specialized->getOutputBuffers().front();
    });
    auto primal = llvm::find_if(buffers, [&](auto buffer) {
      return buffer.getOrdinal() == specialized->getOutputBuffers().back();
    });
    ASSERT_NE(normalized, buffers.end());
    ASSERT_NE(primal, buffers.end());
    auto normalizedType =
        mlir::cast<mlir::RankedTensorType>(normalized->getTensorType());
    auto primalType =
        mlir::cast<mlir::RankedTensorType>(primal->getTensorType());
    EXPECT_EQ(normalizedType.getShape(), llvm::ArrayRef<int64_t>({2, 5, 19}));
    EXPECT_EQ(primalType.getShape(), llvm::ArrayRef<int64_t>({2, 5, 7}));
    EXPECT_FALSE(normalized->getLiveOut());
    EXPECT_EQ(normalized->getStorage(),
              mlir::shuttle::MaterializationStorage::Temporary);
    ASSERT_EQ(normalized->getConsumers().size(), 1u);
    EXPECT_EQ(tasks[normalized->getConsumers().front()].getKind(),
              mlir::shuttle::MaterializationTaskKind::Map);
    EXPECT_TRUE(primal->getLiveOut());

    mlir::FailureOr<llvm::SmallVector<uint8_t>> transport =
        mlir::shuttle::serializeGpuExecutableBundle(*built->module);
    ASSERT_TRUE(mlir::succeeded(transport));
    auto executable = mlir::shuttle::GpuExecutable::Load(*transport);
    ASSERT_TRUE(executable.ok()) << executable.status();
    const auto &specializedLaunch =
        (*executable)->launches()[specialized->getOrdinal()];
    std::vector<int64_t> expectedOutputSlots{
        static_cast<int64_t>(normalized->getOrdinal()),
        static_cast<int64_t>(primal->getOrdinal())};
    EXPECT_EQ(specializedLaunch.outputSlots, expectedOutputSlots);
  }
}

TEST(Abi22MapFusionContract,
     CompactHybridDeclaresEverySharedRegionRecomputation) {
  std::optional<std::string> source = fixtureSource(
      "jax-0.10.1-bf16-rmsnorm_contract-5x19x257-projection-vjp.mlir");
  ASSERT_TRUE(source);
  for (mlir::shuttle::NumericalPolicy policy :
       {mlir::shuttle::NumericalPolicy::SourceOrdered,
        mlir::shuttle::NumericalPolicy::Fast}) {
    SCOPED_TRACE(mlir::shuttle::stringifyNumericalPolicy(policy).str());
    std::unique_ptr<BuiltModule> built =
        build(*source, policy, kFusionTuning, false, true);
    ASSERT_TRUE(built);
    std::optional<PlanSummary> planSummary = summary(*built->module);
    ASSERT_TRUE(planSummary);
    EXPECT_TRUE(planSummary->hasContractRegion);
    EXPECT_TRUE(planSummary->hasFoldRegion);
    EXPECT_TRUE(planSummary->exactSourceCoverage);

    auto plan = *built->module->getOps<mlir::shuttle::MaterializationPlanOp>()
                     .begin();
    auto tasks = llvm::to_vector(
        plan.getBody()
            .front()
            .getOps<mlir::shuttle::MaterializationTaskOp>());
    auto foldTask = llvm::find_if(tasks, [](auto task) {
      return static_cast<bool>(task->template getAttrOfType<mlir::ArrayAttr>(
          "fold_region_sources"));
    });
    auto contractTask = llvm::find_if(tasks, [](auto task) {
      return static_cast<bool>(task->template getAttrOfType<mlir::ArrayAttr>(
          "contract_region_sources"));
    });
    ASSERT_NE(foldTask, tasks.end());
    ASSERT_NE(contractTask, tasks.end());
    mlir::shuttle::MaterializationTaskOp foldRegion = *foldTask;
    mlir::shuttle::MaterializationTaskOp contractRegion = *contractTask;
    auto foldRecomputed = foldRegion->getAttrOfType<mlir::ArrayAttr>(
        "fold_region_recomputed_sources");
    auto contractRecomputed = contractRegion->getAttrOfType<mlir::ArrayAttr>(
        "contract_region_recomputed_sources");
    ASSERT_TRUE(foldRecomputed);
    ASSERT_TRUE(contractRecomputed);
    EXPECT_EQ(sourceOperationOrdinals(foldRecomputed),
              llvm::ArrayRef<int64_t>({0}));
    EXPECT_EQ(sourceOperationOrdinals(contractRecomputed),
              llvm::ArrayRef<int64_t>({0}));

    mlir::FailureOr<llvm::SmallVector<uint8_t>> transport =
        mlir::shuttle::serializeGpuExecutableBundle(*built->module);
    ASSERT_TRUE(mlir::succeeded(transport));
    auto executable = mlir::shuttle::GpuExecutable::Load(*transport);
    ASSERT_TRUE(executable.ok()) << executable.status();
    EXPECT_EQ((*executable)->launches().size(), tasks.size());
  }
}

TEST(Abi22MapFusionContract,
     HybridNormalizedBoundaryRejectsOrderLifetimeAndStorageCorruption) {
  constexpr const char *fixtures[] = {
      "jax-0.10.1-bf16-rmsnorm_contract-5x19x7-composed.mlir",
      "jax-0.10.1-bf16-rmsnorm_contract-5x19x257-composed.mlir",
  };
  for (const char *fixture : fixtures) {
    SCOPED_TRACE(fixture);
    std::optional<std::string> source = fixtureSource(fixture);
    ASSERT_TRUE(source);
    for (llvm::StringRef mutation : {"output-order", "live-out", "storage"}) {
      SCOPED_TRACE(mutation.str());
      std::unique_ptr<BuiltModule> built = build(
          *source, mlir::shuttle::NumericalPolicy::SourceOrdered,
          kFusionTuning, false, false);
      ASSERT_TRUE(built);
      ASSERT_TRUE(mutateHybridNormalizedOutput(*built->module, mutation));
      EXPECT_FALSE(verifyMaterializationPlan(*built->module));
    }
  }
}

TEST(Abi22MapFusionContract,
     RejectsCorruptNormContractSchemaBuffersAndInternalSsaEdges) {
  struct Mutation {
    const char *name;
    const char *filename;
  };
  constexpr Mutation mutations[] = {
      {"strategy", "jax-0.10.1-bf16-rmsnorm_contract-5x19x7-forward.mlir"},
      {"cost", "jax-0.10.1-bf16-rmsnorm_contract-5x19x7-forward.mlir"},
      {"sources", "jax-0.10.1-bf16-rmsnorm_contract-5x19x7-forward.mlir"},
      {"fold-source", "jax-0.10.1-bf16-rmsnorm_contract-5x19x7-forward.mlir"},
      {"contract-source",
       "jax-0.10.1-bf16-rmsnorm_contract-5x19x7-forward.mlir"},
      {"boundary-source",
       "jax-0.10.1-bf16-rmsnorm_contract-5x19x7-forward.mlir"},
      {"normalized-operand",
       "jax-0.10.1-bf16-rmsnorm_contract-5x19x7-forward.mlir"},
      {"epilogue", "jax-0.10.1-bf16-rmsnorm_contract-5x19x7-forward.mlir"},
      {"buffers", "jax-0.10.1-bf16-rmsnorm_contract-5x19x7-forward.mlir"},
      {"ssa-edge", "jax-0.10.1-bf16-rmsnorm_contract-5x19x7-forward.mlir"},
      {"recomputation",
       "jax-0.10.1-bf16-rmsnorm_contract-5x19x257-forward.mlir"},
  };
  for (const Mutation &mutation : mutations) {
    SCOPED_TRACE(mutation.name);
    std::optional<std::string> source = fixtureSource(mutation.filename);
    ASSERT_TRUE(source);
    for (mlir::shuttle::NumericalPolicy policy :
         {mlir::shuttle::NumericalPolicy::SourceOrdered,
          mlir::shuttle::NumericalPolicy::Fast}) {
      SCOPED_TRACE(policy == mlir::shuttle::NumericalPolicy::Fast ? "fast"
                                                                  : "source");
      std::unique_ptr<BuiltModule> built =
          build(*source, policy, kFusionTuning, false, false);
      ASSERT_TRUE(built);
      ASSERT_TRUE(mutateContractRegionTask(*built->module, mutation.name));
      EXPECT_FALSE(verifyMaterializationPlan(*built->module));
    }
  }
}

TEST(Abi22MapFusionContract,
     NormContractFusionRejectsLiveOutFanoutWrongAxisAndMissingBf16Boundary) {
  std::optional<std::string> fixture = fixtureSource(
      "jax-0.10.1-bf16-rmsnorm_contract-5x19x7-forward.mlir");
  ASSERT_TRUE(fixture);
  std::vector<std::string> mutations;

  mutations.push_back(replaceAllText(
      replaceAllText(
          *fixture,
          ") -> (tensor<5x7xbf16> {jax.result_info = \"result\"}) {",
          ") -> (tensor<5x7xbf16>, tensor<5x19xbf16>) {"),
      "return %17 : tensor<5x7xbf16>",
      "return %17, %15 : tensor<5x7xbf16>, tensor<5x19xbf16>"));
  mutations.push_back(replaceAllText(
      replaceAllText(
          *fixture,
          ") -> (tensor<5x7xbf16> {jax.result_info = \"result\"}) {",
          ") -> (tensor<5x7xbf16>, tensor<5x1xf32>) {"),
      "return %17 : tensor<5x7xbf16>",
      "return %17, %8 : tensor<5x7xbf16>, tensor<5x1xf32>"));

  std::string fanout = replaceAllText(
      *fixture,
      "    %17 = stablehlo.convert %16 : (tensor<5x7xf32>) -> "
      "tensor<5x7xbf16>",
      "    %17 = stablehlo.convert %16 : (tensor<5x7xf32>) -> "
      "tensor<5x7xbf16>\n"
      "    %extra = stablehlo.dot_general %15, %arg2, contracting_dims = [1] "
      "x [0], precision = [DEFAULT, DEFAULT] : (tensor<5x19xbf16>, "
      "tensor<19x7xbf16>) -> tensor<5x7xf32>\n"
      "    %extra_bf16 = stablehlo.convert %extra : (tensor<5x7xf32>) -> "
      "tensor<5x7xbf16>");
  fanout = replaceAllText(
      replaceAllText(
          fanout,
          ") -> (tensor<5x7xbf16> {jax.result_info = \"result\"}) {",
          ") -> (tensor<5x7xbf16>, tensor<5x7xbf16>) {"),
      "return %17 : tensor<5x7xbf16>",
      "return %17, %extra_bf16 : tensor<5x7xbf16>, tensor<5x7xbf16>");
  mutations.push_back(std::move(fanout));

  std::string wrongAxis = replaceAllText(
      *fixture, "%arg2: tensor<19x7xbf16>", "%arg2: tensor<5x7xbf16>");
  wrongAxis = replaceAllText(
      wrongAxis,
      "%16 = stablehlo.dot_general %15, %arg2, contracting_dims = [1] x [0], "
      "precision = [DEFAULT, DEFAULT] : (tensor<5x19xbf16>, "
      "tensor<19x7xbf16>) -> tensor<5x7xf32>",
      "%16 = stablehlo.dot_general %15, %arg2, contracting_dims = [0] x [0], "
      "precision = [DEFAULT, DEFAULT] : (tensor<5x19xbf16>, "
      "tensor<5x7xbf16>) -> tensor<19x7xf32>");
  wrongAxis = replaceAllText(
      wrongAxis,
      "%17 = stablehlo.convert %16 : (tensor<5x7xf32>) -> tensor<5x7xbf16>",
      "%17 = stablehlo.convert %16 : (tensor<19x7xf32>) -> "
      "tensor<19x7xbf16>");
  wrongAxis = replaceAllText(
      wrongAxis,
      ") -> (tensor<5x7xbf16> {jax.result_info = \"result\"}) {",
      ") -> (tensor<19x7xbf16> {jax.result_info = \"result\"}) {");
  wrongAxis = replaceAllText(wrongAxis, "return %17 : tensor<5x7xbf16>",
                             "return %17 : tensor<19x7xbf16>");
  mutations.push_back(std::move(wrongAxis));

  std::string noBf16Boundary = replaceAllText(
      *fixture, "%arg2: tensor<19x7xbf16>", "%arg2: tensor<19x7xf32>");
  noBf16Boundary = replaceAllText(
      noBf16Boundary,
      "%16 = stablehlo.dot_general %15, %arg2, contracting_dims = [1] x [0], "
      "precision = [DEFAULT, DEFAULT] : (tensor<5x19xbf16>, "
      "tensor<19x7xbf16>) -> tensor<5x7xf32>",
      "%16 = stablehlo.dot_general %14, %arg2, contracting_dims = [1] x [0], "
      "precision = [DEFAULT, DEFAULT] : (tensor<5x19xf32>, "
      "tensor<19x7xf32>) -> tensor<5x7xf32>");

  for (const std::string &mutation : mutations) {
    for (mlir::shuttle::NumericalPolicy policy :
         {mlir::shuttle::NumericalPolicy::SourceOrdered,
          mlir::shuttle::NumericalPolicy::Fast}) {
      SCOPED_TRACE(policy == mlir::shuttle::NumericalPolicy::Fast ? "fast"
                                                                  : "source");
      std::unique_ptr<BuiltModule> built =
          build(mutation, policy, kFusionTuning, false, false);
      ASSERT_TRUE(built);
      auto plan = *built->module->getOps<mlir::shuttle::MaterializationPlanOp>()
                       .begin();
      for (auto task : plan.getBody().front().getOps<
               mlir::shuttle::MaterializationTaskOp>())
        EXPECT_FALSE(task->getAttr("contract_region_sources"));
    }
  }
  for (mlir::shuttle::NumericalPolicy policy :
       {mlir::shuttle::NumericalPolicy::SourceOrdered,
        mlir::shuttle::NumericalPolicy::Fast})
    EXPECT_FALSE(build(noBf16Boundary, policy, kFusionTuning, false, false));
}

} // namespace

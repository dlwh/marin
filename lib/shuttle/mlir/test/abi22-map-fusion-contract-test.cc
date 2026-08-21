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
buildAlgebra(llvm::StringRef source, mlir::shuttle::NumericalPolicy policy) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect, mlir::shuttle::ShuttleDialect>();
  auto result = std::make_unique<BuiltModule>();
  result->context = std::make_unique<mlir::MLIRContext>(registry);
  result->module =
      mlir::parseSourceString<mlir::ModuleOp>(source, result->context.get());
  if (!result->module)
    return {};
  result->module->walk(
      [&](mlir::shuttle::RegionOp region) { region.setPolicy(policy); });
  mlir::PassManager manager(result->context.get());
  manager.addPass(mlir::shuttle::createPlanRowFoldMaterializationPass());
  manager.addPass(mlir::shuttle::createVerifyMaterializationPlanPass());
  manager.addPass(mlir::shuttle::createPlanSimt32RowFoldSchedulePass());
  manager.addPass(mlir::shuttle::createVerifySimt32RowFoldSchedulePass());
  manager.addPass(mlir::shuttle::createBuildGpuExecutableBundlePass());
  manager.addPass(mlir::shuttle::createVerifyGpuExecutableBundlePass());
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

TEST(Abi22MapFusionContract,
     RejectsCorruptOrderedProvenanceAndComposedInputMaps) {
  std::optional<std::string> source = fixtureSource(kFixtures[2].filename);
  std::optional<std::string> foldSource = fixtureSource(kFixtures[2].filename);
  ASSERT_TRUE(source);
  ASSERT_TRUE(foldSource);
  constexpr llvm::StringLiteral mutations[] = {
      "duplicate-source", "reordered-source", "stale-source", "missing-map",
      "reordered-map"};
  for (llvm::StringRef mutation : mutations) {
    for (bool foldEpilogue : {false, true}) {
      SCOPED_TRACE(
          (llvm::Twine(foldEpilogue ? "Fold epilogue " : "Map ") + mutation)
              .str());
      std::unique_ptr<BuiltModule> built =
          build(foldEpilogue ? *foldSource : *source,
                mlir::shuttle::NumericalPolicy::SourceOrdered, kFusionTuning,
                false, false);
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
      auto tasks =
          llvm::to_vector(plan.getBody()
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

bool hasSourceOrderedContractRecurrence(llvm::StringRef ptx,
                                        int64_t reduction) {
  llvm::SmallVector<llvm::StringRef> lines;
  ptx.split(lines, '\n');
  auto tokens = [](llvm::StringRef line) {
    llvm::SmallVector<llvm::StringRef> result;
    line.split(result, " ,;\t", -1, false);
    return result;
  };
  auto loop = llvm::find(lines, "$contract_region_source_ordered_loop:");
  if (loop == lines.end())
    return false;
  std::optional<llvm::StringRef> coordinate;
  std::optional<llvm::StringRef> accumulator;
  bool increment = false;
  for (auto line = std::next(loop); line != lines.end(); ++line) {
    llvm::SmallVector<llvm::StringRef> instruction = tokens(*line);
    if (instruction.empty())
      continue;
    if (instruction.front() == "setp.ge.u64") {
      int64_t bound = 0;
      if (instruction.size() != 4 ||
          instruction.back().getAsInteger(10, bound) || bound != reduction)
        return false;
      coordinate = instruction[2];
    } else if (instruction.front() == "fma.rn.f32") {
      if (!coordinate || instruction.size() != 5 ||
          instruction[1] != instruction[4])
        return false;
      accumulator = instruction[1];
    } else if (instruction.front() == "add.u64" && coordinate) {
      if (instruction.size() == 4 && instruction[1] == *coordinate &&
          instruction[2] == *coordinate && instruction[3] == "1")
        increment = true;
    } else if (instruction.front() == "bra" && instruction.size() == 2 &&
               instruction[1] == "$contract_region_source_ordered_loop") {
      return accumulator.has_value() && increment;
    }
  }
  return false;
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
      auto tasks = llvm::to_vector(
          plan.getBody()
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
      EXPECT_NE(contractPtx.find("cvt.rn.bf16.f32"), llvm::StringRef::npos);
      EXPECT_NE(contractPtx.find("cvt.f32.bf16"), llvm::StringRef::npos);
      EXPECT_NE(contractPtx.find("fma.rn.f32"), llvm::StringRef::npos);
      if (policy == mlir::shuttle::NumericalPolicy::SourceOrdered) {
        EXPECT_TRUE(
            hasSourceOrderedContractRecurrence(contractPtx, boundary.reduction));
        EXPECT_EQ(contractPtx.find("shfl.sync.down.b32"), llvm::StringRef::npos);
      } else {
        EXPECT_NE(contractPtx.find("shfl.sync.down.b32"),
                  llvm::StringRef::npos);
        EXPECT_EQ(contractPtx.find("$contract_region_source_ordered_loop:"),
                  llvm::StringRef::npos);
      }
      EXPECT_EQ(contractPtx.count("st.global.u16"), 1u);

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

#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASSED=0
FAILED=0
SKIPPED=0

pass() {
    echo -e "${GREEN}✓ $1${NC}"
    ((PASSED++)) || true
}

fail() {
    echo -e "${RED}✗ $1${NC}"
    ((FAILED++)) || true
}

skip() {
    echo -e "${YELLOW}⊘ $1 (skipped)${NC}"
    ((SKIPPED++)) || true
}

header() {
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  $1${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Parse args
RUN_GPU_TESTS=false
RUN_MULTI_GPU=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) RUN_GPU_TESTS=true; shift ;;
        --multi-gpu) RUN_GPU_TESTS=true; RUN_MULTI_GPU=true; shift ;;
        --help)
            echo "Usage: ./test_release.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpu        Run GPU-dependent tests (model loading, training)"
            echo "  --multi-gpu  Run multi-GPU distributed tests (implies --gpu)"
            echo "  --help       Show this help message"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

header "CIRCUITS RELEASE TEST SUITE"
echo "Started at: $(date)"
echo "GPU tests: $RUN_GPU_TESTS"
echo "Multi-GPU tests: $RUN_MULTI_GPU"

# ============================================================================
header "1. PREREQUISITES"
# ============================================================================

# Check Python version
PYTHON_VERSION=$(uv run python --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
if [[ "$PYTHON_VERSION" == "3.12" ]]; then
    pass "Python version is 3.12.x"
else
    fail "Python version is $PYTHON_VERSION (expected 3.12.x)"
fi

# Check uv
if command -v uv &> /dev/null; then
    pass "uv is installed"
else
    fail "uv is not installed"
    echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check .env exists
if [[ -f .env ]]; then
    pass ".env file exists"
else
    if [[ -f .env.template ]]; then
        skip ".env file missing (copy from .env.template)"
    else
        fail ".env.template not found"
    fi
fi

# ============================================================================
header "2. DEPENDENCY INSTALLATION"
# ============================================================================

if uv sync 2>&1; then
    pass "uv sync completed"
else
    fail "uv sync failed"
    exit 1
fi

# ============================================================================
header "3. MODULE IMPORTS"
# ============================================================================

modules=(
    "circuits.core:Core circuit tracing"
    "circuits.analysis:Analysis module"
    "circuits.evals:Evaluation module"
    "circuits.utils:Utils module"
    "util:Lib util"
    "env_util:Environment util"
    "log_util:Logging util"
)

for module_entry in "${modules[@]}"; do
    module="${module_entry%%:*}"
    name="${module_entry##*:}"
    if uv run python -c "import $module" 2>/dev/null; then
        pass "$name ($module)"
    else
        fail "$name ($module)"
    fi
done

# ============================================================================
header "4. UNIT TESTS"
# ============================================================================

if uv run python -m unittest lib/util/tests/test_subject.py 2>&1; then
    pass "Subject unit tests"
else
    fail "Subject unit tests"
fi

# ============================================================================
header "5. SCRIPT ENTRYPOINTS"
# ============================================================================

if uv run python scripts/train.py --help > /dev/null 2>&1; then
    pass "train.py --help"
else
    fail "train.py --help"
fi

if uv run python scripts/eval.py --help > /dev/null 2>&1; then
    pass "eval.py --help"
else
    fail "eval.py --help"
fi

# ============================================================================
header "6. CONFIG LOADING"
# ============================================================================

configs=(
    "sweep/fc_evaluation/sva_pair.yaml"
    "sweep/fc_evaluation/sva_nopair.yaml"
    "sweep/fc_evaluation/sva_nopair_mean.yaml"
)

for config in "${configs[@]}"; do
    if [[ -f "$config" ]]; then
        pass "Config exists: $config"
    else
        fail "Config missing: $config"
    fi
done

# ============================================================================
header "7. DATA FILES"
# ============================================================================

if [[ -d "data/feature_circuits" ]]; then
    pass "Feature circuits data directory exists"
else
    skip "Feature circuits data directory missing"
fi

if [[ -d "data/wikipedia_datasets" ]]; then
    pass "Wikipedia datasets directory exists"
else
    skip "Wikipedia datasets directory missing"
fi

# ============================================================================
if [[ "$RUN_GPU_TESTS" == "true" ]]; then
header "8. GPU TESTS"
# ============================================================================

# Check CUDA available
if uv run python -c "import torch; assert torch.cuda.is_available(), 'No CUDA'" 2>/dev/null; then
    pass "CUDA is available"
    GPU_COUNT=$(uv run python -c "import torch; print(torch.cuda.device_count())")
    echo "   GPUs detected: $GPU_COUNT"
else
    fail "CUDA not available"
    echo "Skipping remaining GPU tests..."
    RUN_GPU_TESTS=false
fi

if [[ "$RUN_GPU_TESTS" == "true" ]]; then
    # Check HF_TOKEN
    if [[ -n "${HF_TOKEN:-}" ]] || grep -q "HF_TOKEN=" .env 2>/dev/null; then
        pass "HF_TOKEN is set"
    else
        skip "HF_TOKEN not set (needed for gated models)"
    fi

    # Check ARTIFACTS_DIR
    if [[ -n "${ARTIFACTS_DIR:-}" ]] || grep -q "ARTIFACTS_DIR=" .env 2>/dev/null; then
        pass "ARTIFACTS_DIR is set"
    else
        skip "ARTIFACTS_DIR not set"
    fi

    # Model loading test
    echo "Testing model loading (this may take a minute)..."
    if uv run python -c "
from util.subject import Subject
import torch
s = Subject('meta-llama/Llama-3.1-8B', device='cuda', dtype=torch.bfloat16)
print(f'Loaded model with {s.model.config.num_hidden_layers} layers')
del s
torch.cuda.empty_cache()
" 2>&1; then
        pass "Model loading (Llama-3.1-8B)"
    else
        fail "Model loading (Llama-3.1-8B)"
    fi

    # Training smoke test
    echo "Running minimal training test..."
    TMPDIR=$(mktemp -d)
    if ARTIFACTS_DIR="$TMPDIR" uv run python scripts/train.py \
        --model "meta-llama/Llama-3.1-8B" \
        --method nap \
        --dataset simple \
        --data_path data/feature_circuits \
        --num_train_examples 1 \
        --batch_size 1 \
        --device cuda \
        --nopair 2>&1; then
        pass "Training smoke test (1 example)"
    else
        fail "Training smoke test"
    fi
    rm -rf "$TMPDIR"
fi

# Multi-GPU tests
if [[ "$RUN_MULTI_GPU" == "true" && "$GPU_COUNT" -gt 1 ]]; then
    header "9. MULTI-GPU TESTS"

    echo "Running distributed training test..."
    TMPDIR=$(mktemp -d)
    if ARTIFACTS_DIR="$TMPDIR" uv run torchrun --nproc_per_node=2 scripts/train.py \
        --model "meta-llama/Llama-3.1-8B" \
        --method nap \
        --dataset simple \
        --data_path data/feature_circuits \
        --num_train_examples 2 \
        --batch_size 1 \
        --device cuda \
        --nopair 2>&1; then
        pass "Multi-GPU training (2 GPUs)"
    else
        fail "Multi-GPU training"
    fi
    rm -rf "$TMPDIR"
elif [[ "$RUN_MULTI_GPU" == "true" ]]; then
    skip "Multi-GPU tests (only $GPU_COUNT GPU available)"
fi

else
    header "8. GPU TESTS"
    skip "GPU tests (use --gpu flag to enable)"
fi

# ============================================================================
header "TEST SUMMARY"
# ============================================================================

TOTAL=$((PASSED + FAILED + SKIPPED))
echo ""
echo -e "  ${GREEN}Passed:  $PASSED${NC}"
echo -e "  ${RED}Failed:  $FAILED${NC}"
echo -e "  ${YELLOW}Skipped: $SKIPPED${NC}"
echo "  ─────────────"
echo "  Total:   $TOTAL"
echo ""

if [[ $FAILED -eq 0 ]]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi

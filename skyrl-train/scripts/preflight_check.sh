#!/bin/bash
# Pre-flight verification script for SkyRL + SGLang training
# Run this before starting RL training to verify your environment

set -e

echo "========================================"
echo "SkyRL + SGLang Pre-flight Check"
echo "========================================"
echo ""

ERRORS=0
WARNINGS=0

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ok() {
    echo -e "${GREEN}[OK]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    ((WARNINGS++))
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    ((ERRORS++))
}

# 1. Check Python version
echo "1. Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" == "3" ] && [ "$PYTHON_MINOR" == "12" ]; then
    ok "Python $PYTHON_VERSION"
else
    error "Python 3.12.x required, found $PYTHON_VERSION"
fi

# 2. Check PyTorch and CUDA
echo ""
echo "2. Checking PyTorch and CUDA..."
python -c "
import sys
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    if torch.cuda.is_available():
        print(f'CUDA: {torch.version.cuda}')
        print(f'GPUs: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f'  GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)')
    else:
        print('CUDA: NOT AVAILABLE')
        sys.exit(1)
except ImportError:
    print('PyTorch: NOT INSTALLED')
    sys.exit(1)
" && ok "PyTorch and CUDA" || error "PyTorch/CUDA check failed"

# 3. Check SGLang
echo ""
echo "3. Checking SGLang..."
python -c "
import sglang
print(f'SGLang version: {sglang.__version__}')
" && ok "SGLang installed" || warn "SGLang not installed (required for sglang backend)"

# 4. Check FlashInfer
echo ""
echo "4. Checking FlashInfer..."
python -c "
import flashinfer
print(f'FlashInfer version: {flashinfer.__version__}')
" && ok "FlashInfer installed" || warn "FlashInfer not installed (recommended for SGLang)"

# 5. Check Ray
echo ""
echo "5. Checking Ray..."
python -c "
import ray
print(f'Ray version: {ray.__version__}')
" && ok "Ray installed" || error "Ray not installed (required)"

# 6. Check SkyRL
echo ""
echo "6. Checking SkyRL..."
python -c "
import skyrl_train
print('SkyRL training package available')
" && ok "SkyRL installed" || error "SkyRL not installed"

# 7. Check transformers
echo ""
echo "7. Checking Transformers..."
python -c "
import transformers
print(f'Transformers version: {transformers.__version__}')
" && ok "Transformers installed" || error "Transformers not installed"

# 8. Check environment variables
echo ""
echo "8. Checking environment variables..."
if [ -n "$HF_TOKEN" ] || [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then
    ok "HuggingFace token set"
else
    warn "HF_TOKEN not set (required for gated models)"
fi

if [ -z "$RAY_RUNTIME_ENV_HOOK" ]; then
    ok "RAY_RUNTIME_ENV_HOOK not set (good for editable installs)"
else
    warn "RAY_RUNTIME_ENV_HOOK is set - may cause issues with editable SGLang"
fi

# 9. Quick model loading test (optional)
echo ""
echo "9. Testing model loading (small model)..."
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2', trust_remote_code=True)
print('Tokenizer loading: OK')
" && ok "Model loading works" || warn "Model loading test failed"

# 10. Check disk space
echo ""
echo "10. Checking disk space..."
DISK_FREE=$(df -BG . | tail -1 | awk '{print $4}' | tr -d 'G')
if [ "$DISK_FREE" -gt 50 ]; then
    ok "Disk space: ${DISK_FREE}GB free"
else
    warn "Disk space: ${DISK_FREE}GB free (recommend 50GB+)"
fi

# Summary
echo ""
echo "========================================"
echo "Pre-flight Check Summary"
echo "========================================"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}All checks passed! Ready for training.${NC}"
    echo ""
    echo "Quick start:"
    echo "  python -m skyrl_train.entrypoints.main_base \\"
    echo "    +experiment=grpo_qwen2.5-0.5b_math500 \\"
    echo "    generator.backend=sglang"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}Passed with $WARNINGS warning(s). Training should work.${NC}"
    exit 0
else
    echo -e "${RED}Failed with $ERRORS error(s) and $WARNINGS warning(s).${NC}"
    echo ""
    echo "Fix the errors above before training."
    echo "See: docs/VERSION_COMPATIBILITY.md for solutions"
    exit 1
fi

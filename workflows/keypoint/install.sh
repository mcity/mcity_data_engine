#!/bin/bash
# Complete clean installation for MMPose lower body training
# This script installs EXACT TESTED VERSIONS that work together

set -e

cd /home/dataengine/Mcity/mcity_data_engine/workflows/keypoint

echo "==================================================================="
echo "MMPOSE LOWER BODY - COMPLETE CLEAN INSTALLATION"
echo "==================================================================="
echo
echo "This will:"
echo "  1. Remove old venv_kp environment"
echo "  2. Create fresh Python 3.10 environment"
echo "  3. Install exact tested versions"
echo "  4. Verify all imports work"
echo
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Step 1: Backup and remove old environment
echo
echo "==================================================================="
echo "Step 1: Removing old environment"
echo "==================================================================="
deactivate 2>/dev/null || true
if [ -d "venv_kp" ]; then
    echo "Backing up old environment to venv_kp.backup..."
    rm -rf venv_kp.backup 2>/dev/null || true
    mv venv_kp venv_kp.backup
    echo "✓ Old environment backed up"
else
    echo "No existing environment found"
fi
echo

# Step 2: Create fresh Python 3.10 environment
echo "==================================================================="
echo "Step 2: Creating fresh Python 3.10 environment"
echo "==================================================================="

# Check if python3.10 is available
if ! command -v python3.10 &> /dev/null; then
    echo "Error: python3.10 not found"
    echo "Please install Python 3.10:"
    echo "  sudo apt-get install python3.10 python3.10-venv"
    exit 1
fi

python3.10 -m venv venv_kp
source venv_kp/bin/activate
echo "✓ New environment created and activated"
echo "  Python: $(python --version)"
echo

# Step 3: Upgrade pip
echo "==================================================================="
echo "Step 3: Upgrading pip, setuptools, wheel"
echo "==================================================================="
pip install --upgrade pip setuptools wheel
echo "✓ pip upgraded to $(pip --version)"
echo

# Step 4: Install PyTorch 2.0.1 with CUDA 11.8
echo "==================================================================="
echo "Step 4: Installing PyTorch 2.0.1 + CUDA 11.8"
echo "==================================================================="
echo "This may take a few minutes..."
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch
echo
echo "PyTorch verification:"
python -c "import torch; print(f'  ✓ PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'  ✓ CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'  ✓ CUDA version: {torch.version.cuda}')"
echo

# Step 5: Install openmim
echo "==================================================================="
echo "Step 5: Installing openmim"
echo "==================================================================="
pip install -U openmim
echo "✓ openmim installed"
echo

# Step 6: Install mmengine
echo "==================================================================="
echo "Step 6: Installing mmengine 0.10.3"
echo "==================================================================="
pip install mmengine==0.10.3
python -c "import mmengine; print(f'  ✓ mmengine {mmengine.__version__}')"
echo

# Step 7: Install MMCV 2.1.0 for PyTorch 2.0 + CUDA 11.8
echo "==================================================================="
echo "Step 7: Installing MMCV 2.1.0 (prebuilt wheel)"
echo "==================================================================="
echo "Using prebuilt wheel for PyTorch 2.0 + CUDA 11.8"
echo "This avoids compilation and CUDA_HOME issues..."
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

# Verify MMCV
echo
echo "MMCV verification:"
python -c "import mmcv; print(f'  ✓ mmcv {mmcv.__version__}')"

# Critical test: Load CUDA extensions
python -c "import mmcv._ext; print('  ✓ mmcv CUDA extensions loaded')" || {
    echo "  ✗ ERROR: MMCV extensions failed to load"
    echo
    echo "This usually means version mismatch. Please check:"
    echo "  python -c 'import torch; print(torch.__version__)'"
    echo "  python -c 'import mmcv; print(mmcv.__version__)'"
    exit 1
}

# Test MultiScaleDeformableAttention
python -c "from mmcv.ops import MultiScaleDeformableAttention; print('  ✓ mmcv.ops works')" || {
    echo "  ✗ WARNING: mmcv.ops import failed"
    echo "  This may cause issues with some models"
}
echo

# Step 8: Install mmdet
echo "==================================================================="
echo "Step 8: Installing mmdet 3.3.0"
echo "==================================================================="
pip install mmdet==3.3.0
python -c "import mmdet; print(f'  ✓ mmdet {mmdet.__version__}')"
echo

# Step 9: Install mmpose
echo "==================================================================="
echo "Step 9: Installing mmpose 1.3.1"
echo "==================================================================="

# Install chumpy separately first (mmpose dependency that causes issues)
echo "Installing chumpy (mmpose dependency)..."
pip install chumpy || {
    echo "⚠️  chumpy installation failed, trying workaround..."
    # Install numpy first, then chumpy
    pip install "numpy<2.0"
    pip install --no-build-isolation chumpy || {
        echo "⚠️  chumpy still failing, installing mmpose without it..."
        pip install --no-deps mmpose==1.3.1
        pip install json_tricks munkres opencv-python xtcocotools
    }
}

# Now install mmpose
pip install mmpose==1.3.1 || {
    echo "⚠️  mmpose 1.3.1 failed, trying mmpose 1.3.0..."
    pip install mmpose==1.3.0
}

python -c "import mmpose; print(f'  ✓ mmpose {mmpose.__version__}')"
echo

# Step 10: Install xtcocotools (fix numpy compatibility)
echo "==================================================================="
echo "Step 10: Installing xtcocotools"
echo "==================================================================="
pip install xtcocotools --no-cache-dir
python -c "from xtcocotools.coco import COCO; print('  ✓ xtcocotools works')"
echo

# Step 11: Install remaining dependencies
echo "==================================================================="
echo "Step 11: Installing remaining dependencies"
echo "==================================================================="
pip install fiftyone matplotlib seaborn pyyaml numpy pandas opencv-python pillow albumentations tqdm tensorboard scikit-learn scipy
echo "✓ All dependencies installed"
echo

# Step 12: Final comprehensive verification
echo "==================================================================="
echo "FINAL VERIFICATION"
echo "==================================================================="
echo

python << 'EOF'
import sys

def test_import(module_name, import_stmt):
    try:
        exec(import_stmt)
        print(f"✓ {module_name}")
        return True
    except Exception as e:
        print(f"✗ {module_name}")
        print(f"  Error: {str(e)[:100]}")
        return False

print("Testing all critical imports...")
print()

success = True
success &= test_import("torch", "import torch")
success &= test_import("mmengine", "import mmengine")
success &= test_import("mmcv", "import mmcv")
success &= test_import("mmcv._ext", "import mmcv._ext")
success &= test_import("mmcv.ops.MultiScaleDeformableAttention", "from mmcv.ops import MultiScaleDeformableAttention")
success &= test_import("mmdet", "import mmdet")
success &= test_import("mmpose", "import mmpose")
success &= test_import("mmpose.apis", "from mmpose.apis import init_model, inference_topdown")
success &= test_import("xtcocotools.coco", "from xtcocotools.coco import COCO")
success &= test_import("fiftyone", "import fiftyone")

print()
print("=================================================================")
if success:
    print("✓✓✓ ALL TESTS PASSED - INSTALLATION SUCCESSFUL! ✓✓✓")
else:
    print("✗✗✗ SOME TESTS FAILED ✗✗✗")
print("=================================================================")
print()

if success:
    import torch, mmengine, mmcv, mmdet, mmpose
    print("Installed versions:")
    print(f"  Python:   {sys.version.split()[0]}")
    print(f"  PyTorch:  {torch.__version__}")
    print(f"  CUDA:     {torch.version.cuda}")
    print(f"  mmengine: {mmengine.__version__}")
    print(f"  mmcv:     {mmcv.__version__}")
    print(f"  mmdet:    {mmdet.__version__}")
    print(f"  mmpose:   {mmpose.__version__}")
    print()
    print("Next steps:")
    print("  1. Edit config_unified.yaml with your dataset name")
    print("  2. Run: python train_mmpose_with_oks_validation.py --config config_unified.yaml")
    print()
    sys.exit(0)
else:
    print("Please check the errors above.")
    print()
    print("Common fixes:")
    print("  - Ensure Python 3.10 is being used")
    print("  - Check CUDA driver: nvidia-smi")
    print("  - See TROUBLESHOOTING_INDEX.md")
    print()
    sys.exit(1)
EOF

echo "==================================================================="
echo "Installation script completed!"
echo "==================================================================="
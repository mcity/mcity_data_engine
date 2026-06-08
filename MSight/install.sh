#!/usr/bin/env bash
# Install MSight localization dependencies into the current Python environment.
# Run this script before using run_localization = True in config/config.py.
#
# Usage:
#   bash MSIGHT/install.sh
#
# If you are using a virtual environment, activate it first:
#   source venv/bin/activate && bash MSIGHT/install.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"

echo "=== MSight localization install ==="
echo "Requirements file: $REQUIREMENTS"
echo "Python: $(python --version 2>&1)"
echo "pip:    $(pip --version 2>&1)"
echo ""

pip install -r "$REQUIREMENTS"

echo ""
echo "=== Install complete ==="
echo "MSight localization is ready. Set run_localization=True in config/config.py to use it."

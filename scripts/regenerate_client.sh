#!/bin/bash
set -e  # Exit on error

# FAIM Client Regeneration Script
# This script regenerates the faim_client from openapi.json and cleans up unused API endpoints
# Usage: ./scripts/regenerate_client.sh

echo "========================================="
echo "FAIM Client Regeneration Script"
echo "========================================="
echo ""

# Check if openapi.json exists
if [ ! -f "openapi.json" ]; then
    echo "❌ Error: openapi.json not found in current directory"
    exit 1
fi

# Check if client.config.yaml exists
if [ ! -f "client.config.yaml" ]; then
    echo "❌ Error: client.config.yaml not found in current directory"
    exit 1
fi

echo "📝 Step 1: Generating client from OpenAPI spec..."
echo "   Using: openapi-python-client"
echo ""

# Run openapi-python-client
openapi-python-client generate \
  --path openapi.json \
  --config client.config.yaml \
  --overwrite \
  --meta none

if [ $? -ne 0 ]; then
    echo "❌ Client generation failed"
    exit 1
fi

echo "✅ Client generated successfully"
echo ""

echo "🧹 Step 2: Cleaning up unused API endpoints..."
echo ""

# Track what we're removing
removed_count=0

# Remove user management API endpoints
if [ -d "faim_client/api/user" ]; then
    echo "   Removing: faim_client/api/user/ (session management - unused)"
    rm -rf faim_client/api/user
    removed_count=$((removed_count + 1))
fi

# Remove API keys management endpoints
if [ -d "faim_client/api/api_keys" ]; then
    echo "   Removing: faim_client/api/api_keys/ (API key management - unused)"
    rm -rf faim_client/api/api_keys
    removed_count=$((removed_count + 1))
fi

echo ""
echo "🧹 Step 3: Cleaning up unused model files..."
echo ""

# Remove API key related models
# Note: http_validation_error.py and validation_error.py are kept as they may be used by FastAPI validation
models_to_remove=(
    "api_key_info.py"
    "create_api_key_request.py"
    "create_api_key_response.py"
    "get_all_api_keys_response.py"
    "revoke_api_key_request.py"
)

for model_file in "${models_to_remove[@]}"; do
    model_path="faim_client/models/$model_file"
    if [ -f "$model_path" ]; then
        echo "   Removing: $model_path"
        rm "$model_path"
        removed_count=$((removed_count + 1))
    fi
done

echo ""
echo "✅ Cleanup complete ($removed_count items removed)"
echo ""

echo "🔧 Step 4: Fixing faim_client/models/__init__.py..."
echo ""

# Fix the __init__.py to remove imports of deleted models
cat > faim_client/models/__init__.py << 'EOF'
"""Contains all the data models used in inputs/outputs"""

from .error_code import ErrorCode
from .error_response import ErrorResponse
from .http_validation_error import HTTPValidationError
from .model_name import ModelName
from .validation_error import ValidationError

__all__ = (
    "ErrorCode",
    "ErrorResponse",
    "HTTPValidationError",
    "ModelName",
    "ValidationError",
)
EOF

echo "   ✅ Updated faim_client/models/__init__.py"
echo ""

echo "📊 Step 5: Verification..."
echo ""

# Verify key files exist
echo "   Checking generated files..."
if [ -f "faim_client/models/model_name.py" ]; then
    echo "   ✅ ModelName enum generated"
else
    echo "   ❌ ModelName enum missing"
fi

if [ -f "faim_client/models/error_code.py" ]; then
    echo "   ✅ ErrorCode enum generated"
else
    echo "   ❌ ErrorCode enum missing"
fi

if [ -f "faim_client/models/error_response.py" ]; then
    echo "   ✅ ErrorResponse model generated"
else
    echo "   ❌ ErrorResponse model missing"
fi

if [ -d "faim_client/api/forecast" ]; then
    echo "   ✅ Forecast API endpoint retained"
else
    echo "   ❌ Forecast API endpoint missing"
fi

if [ -d "faim_client/api/health" ]; then
    echo "   ✅ Health API endpoint retained"
else
    echo "   ⚠️  Health API endpoint missing (optional)"
fi

echo ""
echo "========================================="
echo "✅ Regeneration Complete!"
echo "========================================="
echo ""
echo "Generated client is ready at: faim_client/"
echo ""
echo "Next steps:"
echo "  1. Review changes in faim_client/"
echo "  2. Update faim_sdk/ if needed"
echo "  3. Run tests to verify compatibility"
echo ""
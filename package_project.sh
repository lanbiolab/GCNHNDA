#!/bin/bash
# FREEDOM Project Packaging Script
# Usage: bash package_project.sh [option]
# Options:
#   core    - Source code and configs only (~1MB)
#   standard - Code + tsRNA-disease2 data (~35MB)
#   full    - Everything except .pixi (~360MB)

set -e

PROJECT_DIR="/mnt/d/project/FREEDOM-master-v3/FREEDOM-master"
OUTPUT_DIR="/mnt/d/project/FREEDOM-master-v3"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_NAME="FREEDOM-tsRNA"

# Parse option
OPTION=${1:-standard}

echo "=========================================="
echo "FREEDOM Project Packaging"
echo "=========================================="
echo "Option: $OPTION"
echo "Project: $PROJECT_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Create temp directory
TEMP_DIR=$(mktemp -d)
PACKAGE_NAME="${BASE_NAME}_${OPTION}_${TIMESTAMP}"
PACKAGE_PATH="$TEMP_DIR/$PACKAGE_NAME"

echo "Creating package directory..."
mkdir -p "$PACKAGE_PATH"

# Copy function with exclusions
copy_with_excludes() {
    local src=$1
    local dest=$2

    rsync -av --progress \
        --exclude='.pixi' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.git' \
        --exclude='*.log' \
        --exclude='saved/' \
        $src/ $dest/
}

case $OPTION in
    core)
        echo "Packaging CORE version (code only)..."
        # Copy source code
        cp -r "$PROJECT_DIR/src" "$PACKAGE_PATH/"
        # Copy configs
        cp -r "$PROJECT_DIR/pixi.toml" "$PACKAGE_PATH/"
        cp -r "$PROJECT_DIR/pixi.lock" "$PACKAGE_PATH/"
        cp "$PROJECT_DIR/CLAUDE.md" "$PACKAGE_PATH/"
        cp "$PROJECT_DIR/.gitignore" "$PACKAGE_PATH/" 2>/dev/null || true
        # Create minimal data structure
        mkdir -p "$PACKAGE_PATH/data/tsRNA-disease2"
        ;;

    standard)
        echo "Packaging STANDARD version (code + tsRNA data)..."
        # Copy everything except large datasets
        rsync -av --progress \
            --exclude='.pixi' \
            --exclude='__pycache__' \
            --exclude='*.pyc' \
            --exclude='.git' \
            --exclude='*.log' \
            --exclude='backup' \
            --exclude='log' \
            --exclude='mytest' \
            --exclude='data/baby' \
            --exclude='data/denovo' \
            --exclude='data/denovo_tsRNA-disease2' \
            --exclude='data/tsRNA-disease' \
            --exclude='data/tsRNA_PlotPng' \
            --exclude='data/tsRNA_denovo' \
            --exclude='saved/' \
            "$PROJECT_DIR/" "$PACKAGE_PATH/"
        # Keep tsRNA-disease2 data
        mkdir -p "$PACKAGE_PATH/data/tsRNA-disease2"
        rsync -av --progress "$PROJECT_DIR/data/tsRNA-disease2/" "$PACKAGE_PATH/data/tsRNA-disease2/"
        ;;

    full)
        echo "Packaging FULL version (all data)..."
        rsync -av --progress \
            --exclude='.pixi' \
            --exclude='__pycache__' \
            --exclude='*.pyc' \
            --exclude='.git' \
            --exclude='saved/' \
            "$PROJECT_DIR/" "$PACKAGE_PATH/"
        ;;

    *)
        echo "Unknown option: $OPTION"
        echo "Usage: $0 [core|standard|full]"
        rm -rf "$TEMP_DIR"
        exit 1
        ;;
esac

# Create README for the package
cat > "$PACKAGE_PATH/README_PACKAGE.txt" << EOF
FREEDOM-tsRNA Project Package
================================
Package Type: $OPTION
Created: $(date)
Contents:
- Source code: src/
- Configuration: pixi.toml, pixi.lock
- Documentation: CLAUDE.md

Setup Instructions:
1. Install Pixi: curl -fsSL https://pixi.sh/install.sh | bash
2. Navigate to project: cd $PACKAGE_NAME
3. Install dependencies: pixi install
4. Run training: cd src && python main.py -m FREEDOM -d tsRNA-disease2

Package Contents:
$(du -sh "$PACKAGE_PATH"/* 2>/dev/null || ls -la "$PACKAGE_PATH")
EOF

# Show package size
echo ""
echo "Package contents:"
du -sh "$PACKAGE_PATH"/* 2>/dev/null || ls -la "$PACKAGE_PATH"
echo ""
echo "Total package size:"
du -sh "$PACKAGE_PATH"

# Create tar.gz archive
echo ""
echo "Creating TAR.GZ archive..."
cd "$TEMP_DIR"
tar -czf "${PACKAGE_NAME}.tar.gz" "$PACKAGE_NAME"

# Move to output directory
mv "${PACKAGE_NAME}.tar.gz" "$OUTPUT_DIR/"

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "=========================================="
echo "Packaging Complete!"
echo "=========================================="
echo "Output: $OUTPUT_DIR/${PACKAGE_NAME}.tar.gz"
ls -lh "$OUTPUT_DIR/${PACKAGE_NAME}.tar.gz"

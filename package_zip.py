#!/usr/bin/env python3
"""
FREEDOM Project Packaging Script (ZIP format)
Usage: python package_zip.py [option]
Options:
  core     - Source code and configs only (~1MB)
  standard - Code + tsRNA-disease2 data (~35MB)
  full     - Everything except .pixi (~360MB)
"""

import os
import sys
import zipfile
import shutil
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path("/mnt/d/project/FREEDOM-master-v3/FREEDOM-master")
OUTPUT_DIR = Path("/mnt/d/project/FREEDOM-master-v3")

# Exclusions common to all packages
COMMON_EXCLUDES = {
    '.pixi', '__pycache__', '.git', '.gitattributes',
    '*.pyc', '*.log', 'saved'
}

# Standard package excludes (in addition to common)
STANDARD_EXCLUDES = {
    'backup', 'log', 'mytest',
    'data/baby', 'data/denovo', 'data/denovo_tsRNA-disease2',
    'data/tsRNA-disease', 'data/tsRNA_PlotPng', 'data/tsRNA_denovo'
}

def should_exclude(path: Path, excludes: set, base_path: Path) -> bool:
    """Check if a path should be excluded from the package."""
    rel_path = str(path.relative_to(base_path))

    # Check common excludes
    for pattern in COMMON_EXCLUDES:
        if pattern.startswith('*'):
            if path.name.endswith(pattern[1:]):
                return True
        elif pattern in rel_path.split(os.sep) or pattern == path.name:
            return True

    # Check specific excludes
    for pattern in excludes:
        if rel_path.startswith(pattern) or rel_path == pattern:
            return True
        # Handle directory patterns
        if pattern.endswith('/'):
            if rel_path.startswith(pattern[:-1]):
                return True

    return False

def create_readme(package_name: str, option: str) -> str:
    """Create README content for the package."""
    return f"""FREEDOM-tsRNA Project Package
================================
Package Type: {option}
Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Contents:
- Source code: src/
- Configuration: pixi.toml, pixi.lock
- Documentation: CLAUDE.md

Setup Instructions:
1. Install Pixi: curl -fsSL https://pixi.sh/install.sh | bash
2. Navigate to project: cd {package_name}
3. Install dependencies: pixi install
4. Run training: cd src && python main.py -m FREEDOM -d tsRNA-disease2

Available Commands:
- Basic training: python src/main.py -m FREEDOM -d tsRNA-disease2
- 5-fold cross validation: python src/cross_validation.py -d tsRNA-disease2 --k 5
- Leave-one-disease validation: python src/denovo_cross_validation.py -d tsRNA-disease2

For more details, see CLAUDE.md
"""

def package_core(output_path: Path, package_name: str):
    """Create core package (code only)."""
    print("Packaging CORE version (code only)...")

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add source code
        for root, dirs, files in os.walk(PROJECT_DIR / "src"):
            dirs[:] = [d for d in dirs if d not in COMMON_EXCLUDES]
            for file in files:
                if file.endswith('.pyc') or file.endswith('.log'):
                    continue
                file_path = Path(root) / file
                arcname = f"{package_name}/src/{file_path.relative_to(PROJECT_DIR / 'src')}"
                zf.write(file_path, arcname)

        # Add configs
        zf.write(PROJECT_DIR / "pixi.toml", f"{package_name}/pixi.toml")
        zf.write(PROJECT_DIR / "pixi.lock", f"{package_name}/pixi.lock")
        zf.write(PROJECT_DIR / "CLAUDE.md", f"{package_name}/CLAUDE.md")

        # Add minimal data structure
        zf.writestr(f"{package_name}/data/tsRNA-disease2/.gitkeep", "")

        # Add README
        zf.writestr(f"{package_name}/README_PACKAGE.txt", create_readme(package_name, "core"))

def package_standard(output_path: Path, package_name: str):
    """Create standard package (code + tsRNA-disease2 data)."""
    print("Packaging STANDARD version (code + tsRNA data)...")

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add all files EXCEPT data directory (handle separately)
        for root, dirs, files in os.walk(PROJECT_DIR):
            # Skip .pixi and other common excludes
            dirs[:] = [d for d in dirs if d not in COMMON_EXCLUDES and d != 'data']

            for file in files:
                file_path = Path(root) / file

                # Skip excluded files
                if file_path.suffix in ['.pyc', '.log']:
                    continue

                # Skip backup and log directories
                rel_path = str(file_path.relative_to(PROJECT_DIR))
                if any(excl in rel_path.split(os.sep) for excl in ['backup', 'log', 'mytest', 'saved']):
                    continue

                # Create archive name
                arcname = f"{package_name}/{file_path.relative_to(PROJECT_DIR)}"
                zf.write(file_path, arcname)
                print(f"  Added: {file_path.relative_to(PROJECT_DIR)}", end='\r')

        # Add tsRNA-disease2 data directory
        data_dir = PROJECT_DIR / "data" / "tsRNA-disease2"
        if data_dir.exists():
            print(f"\nAdding tsRNA-disease2 data directory...")
            for root, dirs, files in os.walk(data_dir):
                dirs[:] = [d for d in dirs if d not in COMMON_EXCLUDES]
                for file in files:
                    file_path = Path(root) / file
                    arcname = f"{package_name}/data/tsRNA-disease2/{file_path.relative_to(data_dir)}"
                    zf.write(file_path, arcname)
                    print(f"  Added: data/tsRNA-disease2/{file_path.relative_to(data_dir)}", end='\r')

        # Add README
        zf.writestr(f"{package_name}/README_PACKAGE.txt", create_readme(package_name, "standard"))

    print()

def package_full(output_path: Path, package_name: str):
    """Create full package (all data)."""
    print("Packaging FULL version (all data)...")

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(PROJECT_DIR):
            # Filter directories
            dirs[:] = [d for d in dirs if d not in COMMON_EXCLUDES]

            for file in files:
                file_path = Path(root) / file

                # Skip excluded files
                if should_exclude(file_path, set(), PROJECT_DIR):
                    continue

                # Create archive name
                arcname = f"{package_name}/{file_path.relative_to(PROJECT_DIR)}"
                zf.write(file_path, arcname)
                print(f"  Added: {file_path.relative_to(PROJECT_DIR)}", end='\r')

        # Add README
        zf.writestr(f"{package_name}/README_PACKAGE.txt", create_readme(package_name, "full"))

    print()

def main():
    option = sys.argv[1] if len(sys.argv) > 1 else "standard"

    print("=" * 50)
    print("FREEDOM Project Packaging (ZIP)")
    print("=" * 50)
    print(f"Option: {option}")
    print(f"Project: {PROJECT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_name = f"FREEDOM-tsRNA_{option}_{timestamp}"
    output_file = OUTPUT_DIR / f"{package_name}.zip"

    if option == "core":
        package_core(output_file, package_name)
    elif option == "standard":
        package_standard(output_file, package_name)
    elif option == "full":
        package_full(output_file, package_name)
    else:
        print(f"Unknown option: {option}")
        print("Usage: python package_zip.py [core|standard|full]")
        sys.exit(1)

    # Get file size
    size_mb = output_file.stat().st_size / (1024 * 1024)

    print()
    print("=" * 50)
    print("Packaging Complete!")
    print("=" * 50)
    print(f"Output: {output_file}")
    print(f"Size: {size_mb:.1f} MB")

    # List contents summary
    with zipfile.ZipFile(output_file, 'r') as zf:
        print(f"Files: {len(zf.namelist())}")

if __name__ == "__main__":
    main()

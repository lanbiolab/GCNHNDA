FREEDOM-tsRNA Project Package
================================
Package Type: standard
Created: 2026-07-28 17:48:02

Contents:
- Source code: src/
- Configuration: pixi.toml, pixi.lock
- Documentation: CLAUDE.md

Setup Instructions:
1. Install Pixi: curl -fsSL https://pixi.sh/install.sh | bash
2. Navigate to project: cd FREEDOM-tsRNA_standard_20260728_174759
3. Install dependencies: pixi install
4. Run training: cd src && python main.py -m FREEDOM -d tsRNA-disease2

Available Commands:
- Basic training: python src/main.py -m FREEDOM -d tsRNA-disease2
- 5-fold cross validation: python src/cross_validation.py -d tsRNA-disease2 --k 5
- Leave-one-disease validation: python src/denovo_cross_validation.py -d tsRNA-disease2

For more details, see CLAUDE.md

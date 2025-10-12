#!/usr/bin/env python3
"""
Verification script for QuantumAI installation.
Checks that all files and modules are properly installed.
"""

import os
import sys
from pathlib import Path

def print_status(message, success=True):
    """Print status message with colored indicator."""
    symbol = "✓" if success else "✗"
    print(f"{symbol} {message}")

def check_file_exists(filepath):
    """Check if a file exists."""
    exists = os.path.exists(filepath)
    print_status(f"{filepath}", exists)
    return exists

def check_directory_exists(dirpath):
    """Check if a directory exists."""
    exists = os.path.exists(dirpath) and os.path.isdir(dirpath)
    print_status(f"{dirpath}/", exists)
    return exists

def main():
    """Main verification function."""
    print("=" * 70)
    print("  QUANTUMAI INSTALLATION VERIFICATION")
    print("=" * 70)
    
    all_checks_passed = True
    
    # Check core directories
    print("\n📁 Checking Core Directories:")
    directories = [
        "src",
        "src/models",
        "src/data",
        "src/utils",
        "src/config",
        "examples",
        "configs",
        "docs",
        "data",
        "tests",
    ]
    
    for directory in directories:
        if not check_directory_exists(directory):
            all_checks_passed = False
    
    # Check core files
    print("\n📄 Checking Core Files:")
    files = [
        "README.md",
        "LICENSE",
        "requirements.txt",
        "setup.py",
        ".gitignore",
        "QUICK_START.md",
        "IMPLEMENTATION_SUMMARY.md",
    ]
    
    for file in files:
        if not check_file_exists(file):
            all_checks_passed = False
    
    # Check source files
    print("\n🐍 Checking Python Source Files:")
    source_files = [
        "src/__init__.py",
        "src/models/__init__.py",
        "src/models/base_classifier.py",
        "src/models/lung_cancer_classifier.py",
        "src/models/brain_cancer_classifier.py",
        "src/data/__init__.py",
        "src/data/data_loader.py",
        "src/data/data_augmentation.py",
        "src/utils/__init__.py",
        "src/utils/preprocessing.py",
        "src/utils/evaluation.py",
        "src/utils/visualization.py",
        "src/config/__init__.py",
        "src/config/model_config.py",
    ]
    
    for source_file in source_files:
        if not check_file_exists(source_file):
            all_checks_passed = False
    
    # Check example scripts
    print("\n📝 Checking Example Scripts:")
    examples = [
        "examples/train_lung_cancer_model.py",
        "examples/train_brain_cancer_model.py",
        "examples/predict_lung_cancer.py",
        "examples/predict_brain_cancer.py",
        "examples/demo_system.py",
    ]
    
    for example in examples:
        if not check_file_exists(example):
            all_checks_passed = False
    
    # Check configuration files
    print("\n⚙️  Checking Configuration Files:")
    configs = [
        "configs/lung_cancer_config.yaml",
        "configs/brain_cancer_config.yaml",
    ]
    
    for config in configs:
        if not check_file_exists(config):
            all_checks_passed = False
    
    # Check Python syntax
    print("\n🔍 Checking Python Syntax:")
    try:
        import py_compile
        syntax_errors = []
        
        for source_file in source_files + examples:
            try:
                py_compile.compile(source_file, doraise=True)
                print_status(f"Syntax check: {source_file}", True)
            except py_compile.PyCompileError as e:
                print_status(f"Syntax error in: {source_file}", False)
                syntax_errors.append(source_file)
                all_checks_passed = False
        
        if not syntax_errors:
            print_status("All Python files have valid syntax", True)
    except Exception as e:
        print_status(f"Could not check syntax: {e}", False)
    
    # Try to import modules (without dependencies)
    print("\n📦 Checking Module Structure:")
    sys.path.insert(0, os.getcwd())
    
    try:
        # This will fail if dependencies aren't installed, but structure is checked
        import src
        print_status("src package structure", True)
    except ImportError as e:
        if "numpy" in str(e) or "tensorflow" in str(e) or "cv2" in str(e):
            print_status("src package structure (dependencies not installed)", True)
            print("  Note: TensorFlow/NumPy/OpenCV not installed, but structure is valid")
        else:
            print_status(f"src package import issue: {e}", False)
            all_checks_passed = False
    
    # Final summary
    print("\n" + "=" * 70)
    if all_checks_passed:
        print("  ✓ ALL CHECKS PASSED!")
        print("=" * 70)
        print("\n🎉 QuantumAI is properly installed!")
        print("\nNext Steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Prepare your dataset (see data/README.md)")
        print("  3. Run training: python examples/train_lung_cancer_model.py")
        print("  4. See QUICK_START.md for more details")
        return 0
    else:
        print("  ✗ SOME CHECKS FAILED")
        print("=" * 70)
        print("\n⚠️  Installation incomplete or corrupted")
        print("Please check the missing files/directories above")
        return 1

if __name__ == "__main__":
    sys.exit(main())

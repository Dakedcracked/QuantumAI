from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="quantumai",
    version="1.0.0",
    author="QuantumAI Team",
    description="Professional-grade medical image classification system for cancer detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dakedcracked/QuantumAI",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.10.0",
        "keras>=2.10.0",
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "opencv-python>=4.7.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "pillow>=9.3.0",
        "tqdm>=4.64.0",
        "PyYAML>=6.0",
    ],
)

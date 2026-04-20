from setuptools import setup, find_packages

setup(
    name="replens",
    version="0.1.0",
    description="Representation Engineering Toolkit for LLM Behavior Analysis",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "accelerate>=0.27.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "viz": ["matplotlib>=3.7.0", "seaborn>=0.13.0", "plotly>=5.18.0"],
        "dashboard": ["streamlit>=1.30.0"],
        "dev": ["pytest>=7.0.0", "pytest-cov>=4.0.0"],
    },
    entry_points={
        "console_scripts": [
            "replens=cli:main",
        ],
    },
)

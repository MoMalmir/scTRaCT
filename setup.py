from setuptools import setup, find_packages

setup(
    name="scTRaCT",
    version="0.1.0",
    description="A Python package for cell type classification using Transformer models.",
    author="Mostafa Malmir",
    author_email="malmir.edumail@gmail.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "scanpy>=1.9",
        "scikit-learn>=1.0",
        "numpy",
        "pandas",
        "scipy",
        "anndata",
    ],
    python_requires=">=3.7",
    include_package_data=True,
    license="MIT",
)

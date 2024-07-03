from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

with open("README.md", "r") as fh:
    long_description = fh.read()

ext_modules = []
ext_modules.append(
    CUDAExtension(
        "TriangleSearchCUDA",
        sources=["dspharmnet/csrc/triangle_search.cpp", "dspharmnet/csrc/triangle_search_kernel.cu"],
    )
)

setup(
    name="dspharmnet",
    version="0.1.0",
    license="Apache 2.0",
    author="Seungeun Lee",
    author_email="selee@unist.ac.kr",
    description="Leveraging Input-Level Feature Deformation with Guided-Attention for Sulcal Labeling",
    url="https://github.com/Shape-Lab/DSPHARM-Net",
    keywords=["spherical cnn", "feature deformation", "attention", "sulcal labeling"],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy==1.17.3",
        "scipy==1.8.0",
        "joblib>=0.14.1",
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)

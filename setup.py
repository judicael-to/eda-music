from setuptools import find_packages, setup

setup(
    name="eda-music",
    version="0.2.0",
    packages=find_packages(),
    include_package_data=True,
    description="Framework d'analyse exploratoire de données avec support pour dataset Spotify",
    author="",
    license="MIT",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "missingno>=0.5.0",
        "jupyter>=1.0.0",
        "notebook>=6.4.0",
        "ipywidgets>=7.6.0",
        "scikit-learn>=1.0.0",
        "pytest>=6.2.5",
        "nbconvert>=6.1.0",
    ],
    entry_points={
        "console_scripts": [
            "spotify-analysis=spotify_analysis:main",
            "eda-run=src.analysis.explore_data:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
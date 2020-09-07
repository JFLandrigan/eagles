from setuptools import setup, find_packages

setup(
    name="eagles",
    version="0.1.0",
    description="Data science utility package to help practitioners in their dev work.",
    url="https://github.com/JFLandrigan/eagles",
    author="Jon-Frederick Landrigan",
    author_email="jon.landrigan@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "ipython",
        "kneed",
        "matplotlib",
        "numpy",
        "pandas",
        "scikit-learn",
        "scikit-optimize",
        "scipy",
        "seaborn",
        "statsmodels",
    ],
    zip_safe=False,
)

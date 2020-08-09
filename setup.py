from setuptools import setup, find_packages

setup(
    name="eagles",
    version="0.0.1",
    description="Data science utility package to help practitioners in their dev work.",
    url="https://github.com/JFLandrigan/eagles",
    author="Jon-Frederick Landrigan",
    author_email="jon.landrigan@gmail.com",
    packages=find_packages(),
    install_requires=[
        "gensim",
        "imbalanced-learn",
        "kneed",
        "matplotlib",
        "nltk",
        "numpy",
        "pandas",
        "pingouin",
        "scikit-learn",
        "scikit-optimize",
        "scipy",
        "seaborn",
        "statsmodels",
    ],
    zip_safe=False,
)

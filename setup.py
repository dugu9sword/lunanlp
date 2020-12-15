from setuptools import setup, find_packages

setup(
    name="luna",
    version="inf",
    keywords=["luna", ],
    description="eds sdk",
    long_description="My tool for research",
    license="WTFPL Licence",

    url="https://github.com/dugu9sword/luna",
    author="dugu9sword",
    author_email="dugu9sword@163.com",

    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=[
        "colorama",
        "arrow",
        "psutil",
        "numpy",
        "tabulate",
    ],
    zip_safe=False,

    scripts=[],
    entry_points={}
)
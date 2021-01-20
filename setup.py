from setuptools import setup, find_packages

setup(
    name="lunanlp",
    version="1.0",
    keywords=["luna", ],
    description="eds sdk",
    long_description="My tool for research",
    license="WTFPL Licence",

    url="https://github.com/dugu9sword/lunanlp",
    author="dugu9sword",
    author_email="dugu9sword@163.com",

    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=open("requirements.txt").readlines(),
    zip_safe=False,

    scripts=[],
    entry_points={}
)
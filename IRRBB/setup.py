from setuptools import setup, find_packages
import os

about = {}
# with open("src/__about__.py") as f:
#     exec(f.read(), about)


here = os.path.abspath(os.path.dirname(__file__))

# with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
#     long_description = "\n" + fh.read()

VERSION = "0.0.1"
DESCRIPTION = "StressGAN"


setup(
    name="StressGAN",
    version=VERSION,
    author="Ilias Aarab",
    author_email="ilias_aarab@hotmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=DESCRIPTION,
    python_requires=">=3.7, <3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows",
        "License :: OSI Approved :: MIT License",
    ],
    package_dir={"":"src"},
    packages=find_packages(where= "src") )
"""
A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""
import distutils.command.clean
import os
import glob
import shutil
import subprocess
from pathlib import Path

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

PATH_ROOT = Path(__file__).parent.resolve()
VERSION = "0.1.0a0"

PACKAGE_NAME = "huluir"
sha = "Unknown"

try:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PATH_ROOT).decode("ascii").strip()
except Exception:
    pass

if os.getenv("BUILD_VERSION"):
    version_huluir = os.getenv("BUILD_VERSION")
elif sha != "Unknown":
    version_huluir = f"{VERSION}+{sha[:7]}"


def write_version_file():
    version_path = PATH_ROOT / PACKAGE_NAME / "version.py"
    with open(version_path, "w") as f:
        f.write(f"__version__ = '{version_huluir}'\n")
        f.write(f"git_version = {repr(sha)}\n")


def get_long_description():
    # Get the long description from the README file
    description = (PATH_ROOT / "README.md").read_text(encoding="utf-8")
    # replace relative repository path to absolute link to the release
    static_url = f"https://raw.githubusercontent.com/zhiqwang/huluir/v{VERSION}"
    description = description.replace("docs/source/_static/", f"{static_url}/docs/source/_static/")
    description = description.replace("notebooks/assets/", f"{static_url}/notebooks/assets/")
    return description


def load_requirements(path_dir=PATH_ROOT, file_name="requirements.txt", comment_char="#"):
    with open(path_dir / file_name, "r", encoding="utf-8", errors="ignore") as file:
        lines = [ln.rstrip() for ln in file.readlines() if not ln.startswith("#")]
    reqs = []
    for ln in lines:
        if comment_char in ln:  # filer all comments
            ln = ln[: ln.index(comment_char)].strip()
        if ln.startswith("http"):  # skip directly installed dependencies
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "huluir", "csrc")

    main_source = os.path.join(extensions_dir, "hulu.cpp")
    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"))

    sources = [main_source] + sources

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "huluir._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


class clean(distutils.command.clean.clean):
    def run(self):
        with open(".gitignore") as f:
            ignores = f.read()
            for wildcard in filter(None, ignores.split("\n")):
                for filename in glob.glob(wildcard):
                    try:
                        os.remove(filename)
                    except OSError:
                        shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)


if __name__ == "__main__":
    print(f"Building wheel {PACKAGE_NAME}-{VERSION}")

    write_version_file()

    setup(
        name=PACKAGE_NAME,
        version=version_huluir,
        description="Yet another IR",
        author="Zhiqiang Wang",
        author_email="zhiqwang@foxmail.com",
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        url="https://github.com/zhiqwang/huluir",
        license="Apache License 2.0",
        packages=find_packages(exclude=["test", "deployment", "notebooks"]),
        zip_safe=False,
        classifiers=[
            # Operation system
            "Operating System :: OS Independent",
            # How mature is this project? Common values are
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            "Development Status :: 3 - Alpha",
            # Indicate who your project is intended for
            "Intended Audience :: Developers",
            # Topics
            "Topic :: Education",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Image Recognition",
            # Pick your license as you wish
            "License :: OSI Approved :: Apache Software License",
            # Specify the Python versions you support here. In particular, ensure
            # that you indicate whether you support Python 2, Python 3 or both.
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
        ],
        install_requires=load_requirements(),
        # This field adds keywords for your project which will appear on the
        # project page. What does your project relate to?
        #
        # Note that this is a list of additional keywords, separated
        # by commas, to be used to assist searching for the distribution in a
        # larger catalog.
        keywords="machine-learning, deep-learning, ir",
        # Specify which Python versions you support. In contrast to the
        # 'Programming Language' classifiers above, 'pip install' will check this
        # and refuse to install the project if the version does not match. See
        # https://packaging.python.org/guides/distributing-packages-using-setuptools/#python-requires
        python_requires=">=3.6, <4",
        # List additional URLs that are relevant to your project as a dict.
        #
        # This field corresponds to the "Project-URL" metadata fields:
        # https://packaging.python.org/specifications/core-metadata/#project-url-multiple-use
        #
        # Examples listed include a pattern for specifying where the package tracks
        # issues, where the source is hosted, where to say thanks to the package
        # maintainers, and where to support the project financially. The key is
        # what's used to render the link text on PyPI.
        project_urls={  # Optional
            "Bug Reports": "https://github.com/zhiqwang/huluir/issues",
            "Funding": "https://zhiqwang.com",
            "Source": "https://github.com/zhiqwang/huluir/",
        },
        ext_modules=get_extensions(),
        python_requires=">=3.6",
        cmdclass={
            "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
            "clean": clean,
        },
    )

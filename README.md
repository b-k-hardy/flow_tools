# 4D Flow Tools

This a collection of Python tools written in Python that will aid in physics-based analysis of 4D Flow MRI. The most useful script is probably ``read_dicoms.py``, which reads in a 4D Flow DICOM dataset and exports it as a (.vti, .mat, .h5).

## Setup

1. Clone repository with ``git clone --recursive https://github.com/b-k-hardy/flow_tools.git``. Recursive option is necessary to ensure that the *v*WERP submodule is downloaded. If only the base repository is desired, omit the ``--recursive`` option.
2. The dependencies for this repository are managed with [uv](https://docs.astral.sh/uv/). I would recommend using uv (because it's cool!) but any Python package manager that can use a ``pyproject.toml`` will work.
3. Set paths in ``config/code_paths.yaml``. Note that these features are all optional -- I like to automate analysis in Python, but this codebase can also be used primarily for its DICOM reader + MAT file exporter for a pipeline implemented in MATLAB.

## *v*WERP

The *v*WERP repository is accessible on a [Umich EECS Gitlab instance](https://gitlab.eecs.umich.edu/david.marlevi/vwerp). There may be annoying permission errors based on how your ssh keys are set up. For reference, I included *v*WERP as a submodule to ensure that I always had a clean copy of the relative pressure estimators available. It is not required to download *v*WERP as a submodule; it is merely a convenience. Under you can write a custom path to your *v*WERP directory.

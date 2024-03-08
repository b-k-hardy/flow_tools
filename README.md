# 4D Flow Tools

This a collection of 4D flow tools written in Python that will aid in physics-based analysis of MRI/CT.

## Setup

1. Clone repository with ``git clone --recursive https://github.com/b-k-hardy/flow_tools.git``. Recursive option is necessary to ensure that vWERP submodule is downloaded
2. Navigate to local repository and run ``pip install -r requirements.txt`` or ``pip3 install -r requirements.txt``
3. Download DICOM data at _________ (waiting to de-identify patient data and make a secure, password-protected download link)

## vWERP

Repo accessible [here](https://gitlab.eecs.umich.edu/david.marlevi/vwerp). (Link to a repo hosted on UMich EECS Gitlab instance)

## To do list and planned features

1. Make I/O paths more robust and intuitive (There is currently too much string concatonation voodoo with weird rules.)
2. Add basic 3D visualization module
3. Add basic segmentation module with 3D Slicer compatibility
4. Flesh out git wiki further

import pydicom
import h5py
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from pyevtk.hl import imageToVTK
import nrrd


def import_segmentation(seg_path):
    segmentation, header = nrrd.read(seg_path)
    print(segmentation.shape)
    print(header)

    return segmentation


def import_flow(u_path, v_path, w_path, check=None):

    paths = [u_path, v_path, w_path]
    img5d = []

    for i in range(3):
        # load the DICOM files
        files = []
        print(f"glob: {paths[i]}")
        for fname in glob.glob(paths[i], recursive=False):
            print(f"loading: {fname}", end="\r")
            files.append(pydicom.dcmread(fname))

        print(f"\nfile count: {len(files)}")

        # skip files with no SliceLocation (eg scout views)
        slices = []
        skipcount = 0
        for f in files:
            if hasattr(f, "SliceLocation"):  # and f.AcquisitionNumber == '16':
                slices.append(f)
            else:
                skipcount = skipcount + 1

        print(f"skipped, no SliceLocation: {skipcount}")

        # ensure they are in the correct order
        slices.sort(key=lambda s: (s.TriggerTime, s.SliceLocation))

        # pixel aspects, assuming all slices are the same
        ps = slices[0].PixelSpacing
        ss = slices[0].SliceThickness
        ax_aspect = ps[1] / ps[0]
        sag_aspect = ps[1] / ss
        cor_aspect = ss / ps[0]
        max_slice = max(slices, key=lambda s: s.AcquisitionNumber)
        Nt = int(max_slice.AcquisitionNumber)

        # create 4D array
        img_shape = list(slices[0].pixel_array.shape)
        Nz = int(len(slices) / Nt)
        img_shape.append(Nz)
        img4d = np.zeros((img_shape[0], img_shape[1], Nz, Nt))

        for i in range(Nt):
            timestep = slices[Nz * i : Nz * (i + 1)]
            timestep.sort(key=lambda s: s.SliceLocation)
            for j in range(Nz):
                img_step = timestep[j]
                img2d = (
                    img_step.pixel_array * img_step.RescaleSlope
                    + img_step.RescaleIntercept
                )
                img4d[:, :, j, i] = img2d

        if check is not None:

            # plot 3 orthogonal slices to check for correct indexing
            a1 = plt.subplot(2, 2, 1)
            plt.imshow(img4d[:, :, img_shape[2] // 2, check], cmap="gray")
            a1.set_aspect(ax_aspect)

            a2 = plt.subplot(2, 2, 2)
            plt.imshow(img4d[:, img_shape[1] // 2, :, check], cmap="gray")
            a2.set_aspect(sag_aspect)

            a3 = plt.subplot(2, 2, 3)
            plt.imshow(img4d[img_shape[0] // 2, :, :, check].T, cmap="gray")
            a3.set_aspect(cor_aspect)

            plt.show()

        # fill 4D array with the images from the files

        img5d.append(img4d)

    return np.asarray(img5d)


def import_dicoms(dicom_path, check=None):
    # load the DICOM files
    files = []
    print(f"glob: {dicom_path}")
    for fname in glob.glob(dicom_path, recursive=False):
        print(f"loading: {fname}", end="\r")
        files.append(pydicom.dcmread(fname))

    print(f"\nfile count: {len(files)}")

    # skip files with no SliceLocation (eg scout views)
    slices = []
    skipcount = 0
    for f in files:
        if hasattr(f, "SliceLocation"):  # and f.AcquisitionNumber == '16':
            slices.append(f)
        else:
            skipcount = skipcount + 1

    print(f"skipped, no SliceLocation: {skipcount}")

    # ensure they are in the correct order
    slices.sort(key=lambda s: (s.TriggerTime, s.SliceLocation))

    # pixel aspects, assuming all slices are the same
    ps = slices[0].PixelSpacing
    ss = slices[0].SliceThickness
    ax_aspect = ps[1] / ps[0]
    sag_aspect = ps[1] / ss
    cor_aspect = ss / ps[0]
    max_slice = max(slices, key=lambda s: s.AcquisitionNumber)
    Nt = int(max_slice.AcquisitionNumber)

    # create 3D array
    img_shape = list(slices[0].pixel_array.shape)
    Nz = int(len(slices) / Nt)
    img_shape.append(Nz)
    img4d = np.zeros((img_shape[0], img_shape[1], Nz, Nt))

    # fill 3D array with the images from the files
    for i in range(Nt):
        timestep = slices[Nz * i : Nz * (i + 1)]
        timestep.sort(key=lambda s: s.SliceLocation)
        for j in range(Nz):
            img2d = timestep[j].pixel_array
            img4d[:, :, j, i] = img2d

    if check is not None:
        # plot 3 orthogonal slices to check for correct indexing
        a1 = plt.subplot(2, 2, 1)
        plt.imshow(img4d[:, :, img_shape[2] // 2, check], cmap="gray")
        a1.set_aspect(ax_aspect)

        a2 = plt.subplot(2, 2, 2)
        plt.imshow(img4d[:, img_shape[1] // 2, :, check], cmap="gray")
        a2.set_aspect(sag_aspect)

        a3 = plt.subplot(2, 2, 3)
        plt.imshow(img4d[img_shape[0] // 2, :, :, check].T, cmap="gray")
        a3.set_aspect(cor_aspect)

        plt.show()

    return img4d


def convert_vti(data_path, output_dir, output_filename):

    Nt = 0
    dx = 0

    for t in range(Nt):
        # open additional mat files mat file in series (silly string concatenation here...)
        with h5py.File(data_path + f"_{t+1}.mat", "r") as f:
            # get pointers for velocity struct
            v_pointers = f["v"][:]

            # access the images (matlab equivalent: v{1}.im)
            u = f[v_pointers[0, 0]]["im"][:].T
            v = f[v_pointers[1, 0]]["im"][:].T
            w = f[v_pointers[2, 0]]["im"][:].T

        # write velocity field one timestep at a time
        vel = (u, v, w)
        out_path = f"{output_dir}/{output_filename}_{t:03d}"
        imageToVTK(out_path, spacing=dx, cellData={"Velocity": vel})


def main():

    # NOTE: I'm currently having a user input the paths directly, but this could definitely get tedious (especially since DICOM paths are evil and not even close to being straightforward/intuitive).
    # I will definitely want to automate this process but unfortunately, again, DICOMs are evil and I don't know how to parse their metadata completely yet...
    print("this is now a methods module")

    # mag data in   00003965/*x
    # u data in     0000ED0F/*
    # v data in     0000D7F0/*
    # w data in     00004E45/*


if __name__ == "__main__":
    main()

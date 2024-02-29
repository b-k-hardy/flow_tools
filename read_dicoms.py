import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import nrrd
import numpy as np
import pydicom


def import_segmentation(seg_path):
    segmentation, header = nrrd.read(seg_path)
    print(segmentation.shape)
    print(header)

    return segmentation


def import_flow(
    paths: tuple[str, str, str],
    vencs: tuple[int, int, int],
    phase_range: int = 4096,
    check: None | int = None,
) -> np.ndarray:
    """Function that imports 4D flow phase data

    Args:
        paths (tuple[str, str, str]): _description_
        vencs (tuple[int, int, int]): _description_
        phase_range (int, optional): _description_. Defaults to 4096.
        check (None | int, optional): _description_. Defaults to None.

    Returns:
        np.ndarray: 5-dimensional array of flow data (vel component, x, y, z, t)
    """

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

        # NOTE I'M REPEATING INDICES WTF DUDE DUH THE ORDERING IS WRONG
        for t in range(Nt):
            timestep = slices[Nz * t : Nz * (t + 1)]
            timestep.sort(key=lambda s: s.SliceLocation)
            for j in range(Nz):
                img_step = timestep[j]
                img2d = (
                    img_step.pixel_array * img_step.RescaleSlope
                    + img_step.RescaleIntercept
                )
                img4d[:, :, j, t] = img2d

        img4d = (
            img4d / phase_range * vencs[i] / 100
        )  # convert from phase data to velocity data in m/s

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

    trigger_times = np.array([float(s.TriggerTime) for s in slices])
    trigger_times = np.unique(trigger_times)
    dt = np.mean(np.diff(trigger_times)) * 1e-3  # make sure dt is in seconds

    dx = list(img_step.PixelSpacing) + [img_step.SliceThickness]
    dx = np.array([float(dx_i) for dx_i in dx])

    return np.asarray(img5d), dx, dt


def import_mag(dicom_path: str, check: None | int = None) -> np.ndarray:
    """_summary_

    Args:
        dicom_path (str): _description_
        check (None | int, optional): _description_. Defaults to None.

    Returns:
        np.array: _description_
    """
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


def import_all_dicoms(dir_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Function that automatically walks through DICOM directory tree and imports 4D flow files.

    Args:
       dir_path (str): path to data directory

    Returns:
        tuple[np.ndarray, np.ndarray]: magnitude data, flow data
    """

    phase_encoding_IDs = ["ap", "rl", "fh", "in"]

    # 1. Find all DICOM directories
    dir_list = glob.glob(
        "*/", root_dir=dir_path
    )  # there might be a cleaner way to do this recursively, but for now this is okay -- or maybe I good use os module instead?
    for dir_name in dir_list:

        print(f"Checking {dir_name}")

        try:
            fname = os.listdir(dir_path + dir_name)[0]
        except IndexError:
            continue

        try:
            check_file = pydicom.dcmread(dir_path + dir_name + fname)
        except pydicom.errors.InvalidDicomError:
            continue

        if "4dflow" in check_file.SeriesDescription.lower() and hasattr(
            check_file, "SequenceName"
        ):
            wip = check_file.SequenceName
            print(f"Found {wip}!")  # in, ap, fh

            if wip[-2:] == "in":
                u_path = dir_name + "*"
                u_venc = int(wip[-5:-2])
            elif wip[-2:] == "ap":
                v_path = dir_name + "*"
                v_venc = int(wip[-5:-2])
            elif wip[-2:] == "fh":
                w_path = dir_name + "*"
                w_venc = int(wip[-5:-2])
            else:
                mag_path = dir_name + "*"

    flow_paths = (dir_path + u_path, dir_path + v_path, dir_path + w_path)
    vencs = (u_venc, v_venc, w_venc)

    mag_data = import_mag(dir_path + mag_path)
    flow_data, dx, dt = import_flow(flow_paths, vencs)

    # 3. Stop checking loop when all relevant directories are found (mag, u, v, w)

    return mag_data, flow_data, dx, dt


def main():

    # NEW GOAL: GO AHEAD AND TRY TO DEBUG A DICOM SEARCH TOOL

    # NOTE: I'm currently having a user input the paths directly, but this could definitely get tedious (especially since DICOM paths are evil and not even close to being straightforward/intuitive).
    # I will definitely want to automate this process but unfortunately, again, DICOMs are evil and I don't know how to parse their metadata completely yet...
    print("this is now a methods module")

    # mag data in   00003965/*x
    # u data in     0000ED0F/*
    # v data in     0000D7F0/*
    # w data in     00004E45/*

    import_all_dicoms(
        "/Users/bkhardy/Dropbox (University of Michigan)/MRI_1.22.24/DICOM/0000A628/AAD75E3C/AA62C567/"
    )


if __name__ == "__main__":
    main()

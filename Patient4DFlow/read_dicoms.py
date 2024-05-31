import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import nrrd
import numpy as np
import pydicom
from tqdm import tqdm


def import_segmentation(seg_path: str) -> np.ndarray:
    """_summary_

    Args:
        seg_path (str): _description_

    Returns:
        np.ndarray: _description_
    """
    return nrrd.read(seg_path)[0]


# FIXME: pass in base directory so glob paths aren't so annoying...
def import_flow(
    paths: tuple[str, str, str],
    vencs: tuple[int, int, int],
    phase_range: int = 4096,
    check: None | int = None,
) -> tuple[np.ndarray, np.ndarray, np.floating]:
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

        for fname in tqdm(glob.glob(paths[i], recursive=False)):
            print_out = "/".join(fname.split("/")[-2:])
            # print(f"loading: {print_out}", end="\r")
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
    dx = np.array([float(dx_i) for dx_i in dx]) * 1e-3  # make sure dx is in seconds

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
    for fname in tqdm(glob.glob(dicom_path, recursive=False)):
        print_out = "/".join(fname.split("/")[-2:])
        # print(f"loading: {print_out}", end="\r")
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


def import_ssfp(dicom_path: str, check: None | int = None) -> np.ndarray:
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
    for fname in tqdm(glob.glob(dicom_path, recursive=False)):
        print_out = "/".join(fname.split("/")[-2:])
        # print(f"loading: {print_out}", end="\r")
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
    slices.sort(key=lambda s: s.SliceLocation)

    # pixel aspects, assuming all slices are the same
    ps = slices[0].PixelSpacing
    ss = slices[0].SliceThickness
    ax_aspect = ps[1] / ps[0]
    sag_aspect = ps[1] / ss
    cor_aspect = ss / ps[0]
    max_slice = max(slices, key=lambda s: s.AcquisitionNumber)

    # create 3D array
    img_shape = list(slices[0].pixel_array.shape)
    Nz = int(len(slices))
    img_shape.append(Nz)
    img4d = np.zeros((img_shape[0], img_shape[1], Nz))

    # fill 3D array with the images from the files
    for i in range(Nz):
        img2d = slices[i].pixel_array
        img4d[:, :, i] = img2d

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
    wip_list = []
    ssfp_list = []

    # 1. Find all DICOM directories
    dir_list = glob.glob(
        "**/", root_dir=dir_path, recursive=True
    )  # there might be a cleaner way to do this recursively, but for now this is okay -- or maybe I good use os module instead?
    for dir_name in dir_list:

        print(f"Checking {dir_name}")

        try:
            fname = os.listdir(dir_path + dir_name)[0]
            check_file = pydicom.dcmread(dir_path + dir_name + fname)
        except IndexError:
            continue
        except pydicom.errors.InvalidDicomError:
            continue
        except IsADirectoryError:
            continue

        # I NEED TO CHECK FOR SSFP IN SERIES DESCRIPTION AND THEN IF SSFP IS FOUND I WILL CHECK FOR SLICE THICKNESS TO GRAB ACTUAL DATA
        # there can be multiple studies; therefore we should check for SeriesNumber as well
        if "4dflow" in check_file.SeriesDescription.lower() and hasattr(
            check_file, "SequenceName"
        ):
            wip = check_file.SequenceName
            series = check_file.SeriesNumber
            wip_list.append({"wip": wip, "series_num": int(series), "dir": dir_name})
            print(f"Found {wip}!")  # in, ap, fh
        elif (
            "ssfp" in check_file.SeriesDescription.lower()
            and hasattr(check_file, "SequenceName")
            and int(check_file.SliceThickness) != 0
        ):
            wip = (
                check_file.SequenceName
            )  # FIXME: why was this gone? How did appending a fake wip ever work even once??
            series = check_file.SeriesNumber
            ssfp_list.append({"wip": wip, "series_num": int(series), "dir": dir_name})
            print("Found SSFP!")  # NOTE WHAT IS HAPPENING

    wip_list.sort(key=lambda w: w["series_num"])
    wip_list = wip_list[-4:]
    for item in wip_list:
        dir_name = item["dir"]
        wip = item["wip"]

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
    for item in ssfp_list:
        dir_name = item["dir"]
        ssfp_path = dir_name + "*"

    flow_paths = (dir_path + u_path, dir_path + v_path, dir_path + w_path)
    vencs = (u_venc, v_venc, w_venc)

    mag_data = import_mag(dir_path + mag_path)
    ssfp_data = import_ssfp(dir_path + ssfp_path)
    flow_data, dx, dt = import_flow(flow_paths, vencs)

    return mag_data, ssfp_data, flow_data, dx, dt


def main():
    print("This isn't a script, but feel free to debug/run tests here!")


if __name__ == "__main__":
    main()

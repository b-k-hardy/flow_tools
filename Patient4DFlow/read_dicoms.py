"""Module for reading in DICOM files and converting them to numpy arrays."""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import nrrd
import numpy as np
import pydicom

logger = logging.getLogger(__name__)


def import_segmentation(seg_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Import segmentation from an NRRD file.

    Args:
        seg_path (str): path to the NRRD file

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: segmentation, origin, spacing

    """
    seg_data = nrrd.read(seg_path)
    segmentation = seg_data[0]
    header = seg_data[1]
    origin = np.array(header["space origin"])
    spacing = np.array(header["space directions"])

    return segmentation, origin, spacing


# FIXME: pass in base directory so glob paths aren't so annoying...
def import_flow(
    paths: tuple[str, str, str],
    vencs: tuple[int, int, int],
    phase_range: int = 4096,
    check: None | int = None,
) -> tuple[np.ndarray, np.ndarray, np.floating]:
    """Import 4D flow phase data, convert to velocity, and return as a 5D numpy array.

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
        logger.info("globbing %s", paths[i])

        files = [pydicom.dcmread(fname) for fname in Path(paths[i]).glob("*")]

        logger.info("file count: %d\n", len(files))

        # skip files with no SliceLocation (eg scout views)
        slices = []
        skipcount = 0
        for f in files:
            if hasattr(f, "SliceLocation"):  # and f.AcquisitionNumber == '16':
                slices.append(f)
            else:
                skipcount = skipcount + 1

        if skipcount > 0:
            logger.warning("skipped, no SliceLocation: %d", skipcount)

        # ensure they are in the correct order
        slices.sort(key=lambda s: (s.TriggerTime, s.SliceLocation))

        # pixel aspects, assuming all slices are the same
        ps = slices[0].PixelSpacing
        ss = slices[0].SliceThickness
        ax_aspect = ps[1] / ps[0]
        sag_aspect = ps[1] / ss
        cor_aspect = ss / ps[0]
        max_slice = max(slices, key=lambda s: s.AcquisitionNumber)
        nt = int(max_slice.AcquisitionNumber)

        # create 4D array
        img_shape = list(slices[0].pixel_array.shape)
        nz = int(len(slices) / nt)
        img_shape.append(nz)
        img4d = np.zeros((img_shape[0], img_shape[1], nz, nt))

        # NOTE I'M REPEATING INDICES WTF DUDE DUH THE ORDERING IS WRONG
        for t in range(nt):
            timestep = slices[nz * t : nz * (t + 1)]
            timestep.sort(key=lambda s: s.SliceLocation)
            for j in range(nz):
                img_step = timestep[j]
                img2d = img_step.pixel_array * img_step.RescaleSlope + img_step.RescaleIntercept
                img4d[:, :, j, t] = img2d

        # convert from phase data to velocity data in m/s
        img4d = img4d / phase_range * vencs[i] / 100

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

    dx = [*img_step.PixelSpacing, img_step.SliceThickness]
    dx = np.array([float(dx_i) for dx_i in dx]) * 1e-3  # make sure dx is in seconds

    return np.asarray(img5d), dx, dt


def import_mag(dicom_path: str, check: None | int = None) -> np.ndarray:
    """Import 4D magnitude data from DICOM files.

    Args:
        dicom_path (str): _description_
        check (None | int, optional): _description_. Defaults to None.

    Returns:
        np.array: _description_

    """
    # load the DICOM files
    files = []
    logger.info("globbing %s", dicom_path)

    files = [pydicom.dcmread(fname) for fname in Path(dicom_path).glob("*")]

    logger.info("file count: %d\n", len(files))

    # skip files with no SliceLocation (eg scout views)
    slices = []
    skipcount = 0
    for f in files:
        if hasattr(f, "SliceLocation"):  # and f.AcquisitionNumber == '16':
            slices.append(f)
        else:
            skipcount = skipcount + 1

    if skipcount > 0:
        logger.warning("skipped, no SliceLocation: %d", skipcount)

    # ensure they are in the correct order
    slices.sort(key=lambda s: (s.TriggerTime, s.SliceLocation))

    # pixel aspects, assuming all slices are the same
    ps = slices[0].PixelSpacing
    ss = slices[0].SliceThickness
    ax_aspect = ps[1] / ps[0]
    sag_aspect = ps[1] / ss
    cor_aspect = ss / ps[0]
    max_slice = max(slices, key=lambda s: s.AcquisitionNumber)
    nt = int(max_slice.AcquisitionNumber)

    # create 3D array
    img_shape = list(slices[0].pixel_array.shape)
    nz = int(len(slices) / nt)
    img_shape.append(nz)
    img4d = np.zeros((img_shape[0], img_shape[1], nz, nt))

    # fill 3D array with the images from the files
    for i in range(nt):
        timestep = slices[nz * i : nz * (i + 1)]
        timestep.sort(key=lambda s: s.SliceLocation)
        for j in range(nz):
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
    """Import 3D steady-state free precession (SSFP) data from DICOM files.

    Args:
        dicom_path (str): _description_
        check (None | int, optional): _description_. Defaults to None.

    Returns:
        np.array: _description_

    """
    # load the DICOM files
    files = []
    logger.info("globbing %s", dicom_path)

    files = [pydicom.dcmread(fname) for fname in Path(dicom_path).glob("*")]

    logger.info("file count: %d\n", len(files))

    # skip files with no SliceLocation (eg scout views)
    slices = []
    skipcount = 0
    for f in files:
        if hasattr(f, "SliceLocation"):  # and f.AcquisitionNumber == '16':
            slices.append(f)
        else:
            skipcount = skipcount + 1

    if skipcount > 0:
        logger.warning("skipped, no SliceLocation: %d", skipcount)

    # ensure they are in the correct order
    slices.sort(key=lambda s: s.SliceLocation)

    # pixel aspects, assuming all slices are the same
    ps = slices[0].PixelSpacing
    ss = slices[0].SliceThickness
    ax_aspect = ps[1] / ps[0]
    sag_aspect = ps[1] / ss
    cor_aspect = ss / ps[0]

    # create 3D array
    img_shape = list(slices[0].pixel_array.shape)
    nz = int(len(slices))
    img_shape.append(nz)
    img4d = np.zeros((img_shape[0], img_shape[1], nz))

    # fill 3D array with the images from the files
    for i in range(nz):
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
    """Automatically walk through DICOM directory tree and imports 4D flow files.

    Args:
       dir_path (str): path to data directory

    Returns:
        tuple[np.ndarray, np.ndarray]: magnitude data, flow data

    """
    wip_list = []
    ssfp_list = []

    # 1. Find all DICOM directories
    # there might be a cleaner way to do this recursively? Without all the try except
    dir_list = glob.glob(
        "**/",
        root_dir=dir_path,
        recursive=True,
    )
    for dir_name in dir_list:
        logger.info("Checking %s", dir_name)

        try:
            fname = os.listdir(dir_path + dir_name)[0]
            check_file = pydicom.dcmread(dir_path + dir_name + fname)
        except IndexError:
            continue
        except pydicom.errors.InvalidDicomError:
            continue
        except IsADirectoryError:
            continue

        # I NEED TO CHECK FOR SSFP IN SERIES DESCRIPTION AND THEN IF SSFP IS FOUND I WILL CHECK
        # FOR SLICE THICKNESS TO GRAB ACTUAL DATA... huh
        # there can be multiple studies; therefore we should check for SeriesNumber as well
        if "4dflow" in check_file.SeriesDescription.lower() and hasattr(
            check_file,
            "SequenceName",
        ):
            wip = check_file.SequenceName
            series = check_file.SeriesNumber
            wip_list.append({"wip": wip, "series_num": int(series), "dir": dir_name})
            logger.info("Found %s!", wip)  # in, ap, fh
        elif (
            "ssfp" in check_file.SeriesDescription.lower()
            and hasattr(check_file, "SequenceName")
            and int(check_file.SliceThickness) != 0
        ):
            wip = check_file.SequenceName
            series = check_file.SeriesNumber
            ssfp_list.append({"wip": wip, "series_num": int(series), "dir": dir_name})
            logger.info("Found SSFP!")

    wip_list.sort(key=lambda w: w["series_num"])
    wip_list = wip_list[-4:]

    # stupid ass initialization so linter doesn't freak out
    vencs = np.zeros(3)
    flow_paths = [""] * 3
    mag_path = ""

    for item in wip_list:
        dir_name = item["dir"]
        wip = item["wip"]

        if wip[-2:] == "in":  # or rl?
            flow_paths[0] = dir_path + dir_name
            vencs[0] = int(wip[-5:-2])
        elif wip[-2:] == "ap":
            flow_paths[1] = dir_path + dir_name
            vencs[1] = int(wip[-5:-2])
        elif wip[-2:] == "fh":
            flow_paths[2] = dir_path + dir_name
            vencs[2] = int(wip[-5:-2])
        else:
            mag_path = dir_path + dir_name

    for item in ssfp_list:
        dir_name = item["dir"]
        ssfp_path = dir_name

    mag_data = import_mag(mag_path)
    ssfp_data = import_ssfp(dir_path + ssfp_path)
    flow_data, dx, dt = import_flow(flow_paths, vencs)

    return mag_data, ssfp_data, flow_data, dx, dt

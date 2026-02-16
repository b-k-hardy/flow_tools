"""Module for reading in DICOM files and converting them to numpy arrays."""

import logging
import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import nrrd
import numpy as np
import pydicom
import pyvista as pv
import scipy.io as sio
from ruamel.yaml import YAML
from tqdm import tqdm

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


def import_phase(
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
    nz = len(slices)
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


def find_dicom_dirs(dir_path: str) -> tuple[list[str], list[str]]:
    """Find the relevant DICOM sub-directories.

    This includes sub-directories for 4D flow phase images, 4D flow magnitude images, and SSFP images.

    Args:
        dir_path (str): Absolute path to the data directory (root where ALL of the DICOMs are stored)

    Returns:
        tuple[list[str], list[str]]: list of 4D flow directories (given by WIP), list of SSFP directories

    """
    wip_list = []
    ssfp_list = []

    # 1. Find all DICOM directories
    # there might be a cleaner way to do this recursively? Without all the try except
    for dir_name in Path(dir_path).glob("**/"):
        logger.info("Checking %s", dir_name)

        try:
            fname = os.listdir(dir_name)[0]
            check_file = pydicom.dcmread(dir_name / fname)
        except IndexError:
            continue
        except pydicom.errors.InvalidDicomError:
            continue
        except IsADirectoryError:
            continue

        # I NEED TO CHECK FOR SSFP IN SERIES DESCRIPTION AND THEN IF SSFP IS FOUND I WILL CHECK
        # FOR SLICE THICKNESS TO GRAB ACTUAL DATA... huh
        # there can be multiple studies; therefore we should check for SeriesNumber as well
        if ("4d" and "flow") in check_file.SeriesDescription.lower() and hasattr(
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

    return wip_list, ssfp_list


def import_all_dicoms(dir_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Automatically walk through DICOM directory tree and imports 4D flow files.

    Args:
       dir_path (str): path to data directory

    Returns:
        tuple[np.ndarray, np.ndarray]: magnitude data, flow data

    """
    # wip_list, ssfp_list = find_dicom_dirs(dir_path)
    # NOTE: I THINK I CAN REWRITE THIS TO DO IMPORTING BASED ON THE FILEPATH ONLY
    wip_fname_list = [
        "/Users/bkhardy/Downloads/Dissection_2025_karolinska/CR_20250611/4DFlow/DICOM/00009AD4/AA9D5841/AAD51FE5/0000B33F/EE0A2F30",
        "/Users/bkhardy/Downloads/Dissection_2025_karolinska/CR_20250611/4DFlow/DICOM/00009AD4/AA9D5841/AAD51FE5/000004A5/EE0A6D97",
        "/Users/bkhardy/Downloads/Dissection_2025_karolinska/CR_20250611/4DFlow/DICOM/00009AD4/AA9D5841/AAD51FE5/000051BA/EE0A3B55",
        "/Users/bkhardy/Downloads/Dissection_2025_karolinska/CR_20250611/4DFlow/DICOM/00009AD4/AA9D5841/AAD51FE5/0000215E/EE0A7CFB",
    ]

    wip_dir_list = [
        "/Users/bkhardy/Downloads/Dissection_2025_karolinska/CR_20250611/4DFlow/DICOM/00009AD4/AA9D5841/AAD51FE5/0000B33F/",
        "/Users/bkhardy/Downloads/Dissection_2025_karolinska/CR_20250611/4DFlow/DICOM/00009AD4/AA9D5841/AAD51FE5/000004A5/",
        "/Users/bkhardy/Downloads/Dissection_2025_karolinska/CR_20250611/4DFlow/DICOM/00009AD4/AA9D5841/AAD51FE5/000051BA/",
        "/Users/bkhardy/Downloads/Dissection_2025_karolinska/CR_20250611/4DFlow/DICOM/00009AD4/AA9D5841/AAD51FE5/0000215E/",
    ]

    wip_file_list = [pydicom.dcmread(wip_dir) for wip_dir in wip_fname_list]
    wip_list = [{"wip": wip.SequenceName, "dir": wip_dir_list[wip_file_list.index(wip)]} for wip in wip_file_list]

    ssfp_list = []

    # stupid ass initialization so linter doesn't freak out
    vencs = np.zeros(3)
    flow_paths = [""] * 3
    mag_path = ""

    for item in wip_list:
        dir_name = item["dir"]
        wip = item["wip"]

        if wip[-2:] == "in":  # or rl?
            flow_paths[0] = dir_name
            vencs[0] = int(wip[-5:-2])
        elif wip[-2:] == "ap":
            flow_paths[1] = dir_name
            vencs[1] = int(wip[-5:-2])
        elif wip[-2:] == "fh":
            flow_paths[2] = dir_name
            vencs[2] = int(wip[-5:-2])
        else:
            mag_path = dir_name

    # NOTE: what the hell is this... I'm just overwriting with the most recent dir_name
    for item in ssfp_list:
        dir_name = item["dir"]
        ssfp_path = dir_name

    mag_data = import_mag(mag_path)
    # ssfp_data = import_ssfp(dir_path / ssfp_path)
    ssfp_data = []
    flow_data, dx, dt = import_phase(flow_paths, vencs)

    return mag_data, ssfp_data, flow_data, dx, dt


def import_flow(flow_paths: tuple[Path, Path, Path, Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.floating]:
    dicom_objects = [pydicom.dcmread(dcm_file) for dcm_file in flow_paths]
    wip_list = [{"wip": image.SequenceName, "dir": Path(image.filename).parent} for image in dicom_objects]

    vencs = np.zeros(3)
    phase_paths = [""] * 3
    mag_path = ""

    for item in wip_list:
        dir_name = item["dir"]
        wip = item["wip"]

        if wip[-2:] == "in":  # or rl?
            phase_paths[0] = dir_name
            vencs[0] = int(wip[-5:-2])
        elif wip[-2:] == "ap":
            phase_paths[1] = dir_name
            vencs[1] = int(wip[-5:-2])
        elif wip[-2:] == "fh":
            phase_paths[2] = dir_name
            vencs[2] = int(wip[-5:-2])
        else:
            mag_path = dir_name

    mag_data = import_mag(mag_path)
    flow_data, dx, dt = import_phase(phase_paths, vencs)

    return mag_data, flow_data, dx, dt


def _export_to_mat(
    data_id: str,
    flow_data: np.ndarray,
    mask: np.ndarray,
    dx: np.ndarray,
    dt: np.floating,
    res: np.ndarray,
) -> None:
    """Export flow velocity struct and mask data to MATLAB .mat files.

    Args:
        output_dir (None | str, optional): Path to mat output directory. Defaults to None to autogenerate directory.

    """
    # navigate MATLAB instance to current working directory to call custom function
    logger.info("Exporting MATLAB velocity structs...")

    vx = flow_data[0, :, :, :, :].copy()
    vy = flow_data[1, :, :, :, :].copy()
    vz = flow_data[2, :, :, :, :].copy()
    vx_dict = {"im": vx, "PixDim": dx, "dt": dt, "res": res}
    vy_dict = {"im": vy, "PixDim": dx, "dt": dt, "res": res}
    vz_dict = {"im": vz, "PixDim": dx, "dt": dt, "res": res}
    # this weird format is to make sure the struct is preserved for MATLAB
    vel_output = {"v": np.array([vx_dict, vy_dict, vz_dict], dtype=object).T}

    sio.savemat(f"data/{data_id}/mat/{data_id}_vel.mat", vel_output)

    logger.info("Exporting MATLAB mask...")
    sio.savemat(f"data/{data_id}/mat/{data_id}_mask.mat", {"mask": mask})


def _export_to_h5(
    data_id: str,
    flow_data: np.ndarray,
    mask: np.ndarray,
    dx: np.ndarray,
    dt: np.floating,
    res: np.ndarray,
) -> None:
    """Export flow velocity and mask data to HDF5 format.

    Args:
        output_dir (None | str, optional): Path to h5 output directory. Defaults to None to autogenerate directory.

    """
    logger.info("Exporting HDF5 velocity...")

    with h5py.File(f"data/{data_id}/h5/{data_id}_vel.h5", "w") as f:
        f.create_dataset("v", data=flow_data)
        f.create_dataset("dx", data=dx)
        f.create_dataset("dt", data=dt)
        f.create_dataset("res", data=res)

    logger.info("Exporting HDF5 mask...")
    with h5py.File(f"data/{data_id}/h5/{data_id}_mask.h5", "w") as f:
        f.create_dataset("mask", data=mask)


def _export_to_vti(
    data_id: str,
    flow_data: np.ndarray,
    mask: np.ndarray,
    dx: np.ndarray,
    dt: np.floating,  # unused, but keeping in API just in case I ever add time series info
    res: np.ndarray,
) -> None:
    """Export flow velocity data to VTK ImageData format.

    Args:
        output_dir (None | str, optional): Path to vti output directory. Defaults to None to autogenerate directory.

    """
    logger.info("Exporting to VTI...")
    for t in tqdm(range(res[-1])):
        # write velocity field one timestep at a time
        vel = (flow_data[:, :, :, :, t] * mask).reshape(3, -1, order="F").T

        out_path = f"data/{data_id}/vti/{data_id}_flow_{t:03d}.vti"

        frame = pv.ImageData(dimensions=(res[:-1] + 1), spacing=dx)
        frame.cell_data["Velocity"] = vel
        frame.save(out_path)


def export_data(
    data_id: str,
    flow_data: np.ndarray,
    mask: np.ndarray,
    dx: np.ndarray,
    dt: np.floating,
    res: np.ndarray,
    filetype: tuple[str, str, str] = ("mat", "h5", "vti"),
) -> None:
    output_dir = f"data/{data_id}"
    # make sure output path exists, create directory if not
    base_data_path = Path(output_dir)
    if "mat" in filetype:
        mat_output_dir = base_data_path / "mat"
        mat_output_dir.mkdir(parents=True, exist_ok=True)
        _export_to_mat(data_id, flow_data, mask, dx, dt, res)
    if "h5" in filetype:
        h5_output_dir = base_data_path / "h5"
        h5_output_dir.mkdir(parents=True, exist_ok=True)
        _export_to_h5(data_id, flow_data, mask, dx, dt, res)
    if "vti" in filetype:
        vti_output_dir = base_data_path / "vti"
        vti_output_dir.mkdir(parents=True, exist_ok=True)
        _export_to_vti(data_id, flow_data, mask, dx, dt, res)


def main():
    # set up logging to print to stdout
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # read in config file for DICOM paths and other parameters
    yaml = YAML(typ="safe")
    config_path = Path(__file__).resolve().parent / "config" / "dicom_paths.yaml"
    with config_path.open(encoding="utf-8") as f:
        config = yaml.load(f)

    data_id = config["data_id"]
    base_dicom_directory = config["flow"]["dicom_dir"]
    dicom_extensions = list(config["flow"]["dicom_extensions"])
    seg_path = config["segmentation"]

    # Main reader and exporter
    flow_paths = tuple([Path(f"{base_dicom_directory}/{filename}") for filename in dicom_extensions])
    mag_data, flow_data, dx, dt = import_flow(flow_paths)
    flow_data = np.flip(flow_data, axis=0)
    flow_data[0] *= -1
    flow_data = flow_data.copy()  # make sure it's contiguous in memory prior to export
    res = np.array(mag_data.shape)

    segmentation, seg_origin, seg_spacing = import_segmentation(seg_path)
    segmentation = np.transpose(segmentation, (1, 0, 2))
    segmentation = np.flip(segmentation, axis=2)

    export_data(
        data_id,
        flow_data,
        segmentation,
        dx,
        dt,
        res,
        filetype=("mat", "h5", "vti"),
    )


if __name__ == "__main__":
    main()

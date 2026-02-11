"""Module that definds the Patient4DFlow class."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pyvista as pv
import scipy.io as sio
from scipy import ndimage
from tqdm import tqdm

import patient4Dflow.plot_results as pr
import patient4Dflow.read_dicoms as rd
import patient4Dflow.seg_module as sm

PA_TO_MMHG = 0.00750061683
PVPYTHON_PATH = "/Applications/ParaView-5.13.1.app/Contents/bin/pvbatch"

logger = logging.getLogger(__name__)


class Patient4DFlow:
    """Class to store and analyze 4D flow MRI data for a single patient.

    Specialized helper functions are contained in additional modules of the patient4Dflow library.
    """

    def __init__(
        self,
        patient_id: str,
        data_directory: str,
        seg_path: str = "user",
    ) -> None:
        """Initialize Patient4DFlow object.

        Args:
            patient_id (str): Anonymized patient ID.
            data_directory (str): Absolute path to patient data directory.
            seg_path (str, optional): Name of segmentation or relative path to segmentation from data directory.
            Defaults to "user" to prompt user for input.

        """
        self.id = patient_id
        self.data_directory = data_directory
        self.mag_data, self.ssfp_data, self.flow_data, self.dx, self.dt = rd.import_all_dicoms(self.data_directory)

        self.flow_data = np.flip(self.flow_data, axis=0)
        self.flow_data[0] *= -1
        self.flow_data = self.flow_data.copy()
        self.segmentation, self.seg_origin, self.seg_spacing = self.add_segmentation(
            seg_path,
        )
        self.mask = np.array(self.segmentation != 0).astype(np.float64).copy()

        # NOTE: TEMPORARY VALUES BEFORE I FIX EVERYTHING
        self.res = np.array(self.mag_data.shape)

    def __str__(self) -> str:
        """Print patient ID and data directory."""
        return f"Patient ID: {self.id} @ location {self.data_directory}"

    def add_segmentation(self, path_input: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Add segmentation from 3D Slicer.

        Args:
            path_input (str): path to NRRD segmentation file

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: segmentation array, origin coordinates, and voxel spacing/affine

        """
        seg_path = input("Enter relative path to segmentation: ") if path_input == "user" else path_input

        segmentation, origin, spacing = rd.import_segmentation(
            self.data_directory + seg_path,
        )

        # NOTE: PROBABLY GOING TO BE TEMPORARY CODE AS I WORK THINGS OUT
        segmentation = np.transpose(segmentation, (1, 0, 2))
        segmentation = np.flip(segmentation, axis=2)

        return segmentation, origin, spacing

    def check_orientation(self) -> None:
        """Quick check to make sure 4D flow data is oriented correctly after import.

        Data from one timeframe is exported to VTK format for visualization in ParaView.
        The goal would be to check for any orientation issues in the data, including vector direction.
        """
        mag = self.mag_data[:, :, :, 6].copy()

        u = self.flow_data[0, :, :, :, 6].copy() * self.mask
        v = self.flow_data[1, :, :, :, 6].copy() * self.mask
        w = self.flow_data[2, :, :, :, 6].copy() * self.mask
        vel = np.stack([u.flatten(), v.flatten(), w.flatten()], axis=-1)

        vel_frame = pv.ImageData(dimensions=self.flow_data.shape[1:4], spacing=self.dx)
        vel_frame.cell_data["Velocity"] = vel
        vel_frame.save(f"{self.id}_check_vel.vti")

        mag_frame = pv.ImageData(dimensions=self.mag_data.shape)  # I don't have spacing info read
        mag_frame.cell_data["Magnitude"] = mag.flatten()
        mag_frame.save(f"{self.id}_check_mag.vti")

        # unfortunately it seems like the only solution here is to write a timeframe to disk then load
        # back in. That sucks and is inefficient but whatever.

    def export_vel_to_vti(self, output_dir: None | str = None) -> None:
        """Export flow velocity data to VTK ImageData format.

        Args:
            output_dir (None | str, optional): Path to vti output directory. Defaults to None to autogenerate directory.

        """
        if output_dir is not None:
            output_dir = f"{self.data_directory}/{output_dir}"
        else:
            output_dir = f"{self.data_directory}/{self.id}_flow_vti"

        # make sure output path exists, create directory if not
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Exporting to VTI...")
        for t in tqdm(range(self.flow_data.shape[-1])):
            # write velocity field one timestep at a time
            u = self.flow_data[0, :, :, :, t].copy() * self.mask
            v = self.flow_data[1, :, :, :, t].copy() * self.mask
            w = self.flow_data[2, :, :, :, t].copy() * self.mask
            vel = np.stack([u.flatten(), v.flatten(), w.flatten()], axis=-1)

            out_path = f"{output_dir}/{self.id}_flow_{t:03d}.vti"

            frame = pv.ImageData(dimensions=self.flow_data.shape[1:4], spacing=self.dx)
            frame.cell_data["Velocity"] = vel
            frame.save(out_path)

    def export_to_mat(self, output_dir: None | str = None) -> None:
        """Export flow velocity struct and mask data to MATLAB .mat files.

        Args:
            output_dir (None | str, optional): Path to mat output directory. Defaults to None to autogenerate directory.

        """
        if output_dir is not None:
            output_dir = f"{self.data_directory}/{output_dir}"
        else:
            output_dir = f"{self.data_directory}/{self.id}_mat_files"

        # make sure output path exists, create directory if not
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # navigate MATLAB instance to current working directory to call custom function
        logger.info("Exporting velocity structs...")

        vx = self.flow_data[0, :, :, :, :].copy()
        vy = self.flow_data[1, :, :, :, :].copy()
        vz = self.flow_data[2, :, :, :, :].copy()
        vx_dict = {"im": vx, "PixDim": self.dx, "dt": self.dt, "res": self.res}
        vy_dict = {"im": vy, "PixDim": self.dx, "dt": self.dt, "res": self.res}
        vz_dict = {"im": vz, "PixDim": self.dx, "dt": self.dt, "res": self.res}
        # this weird format is to make sure the struct is preserved for MATLAB
        vel_output = {"v": np.array([vx_dict, vy_dict, vz_dict], dtype=object).T}

        sio.savemat(output_dir + f"/{self.id}_vel.mat", vel_output)

        logger.info("Exporting masks...")
        sio.savemat(
            output_dir + f"/{self.id}_masks.mat",
            {"mask": self.mask, "inlet": self.inlet, "outlet": self.outlet},
        )

    def export_to_h5(self, output_dir: None | str = None) -> None:
        """Export flow velocity and mask data to HDF5 format.

        Args:
            output_dir (None | str, optional): Path to h5 output directory. Defaults to None to autogenerate directory.

        """
        if output_dir is not None:
            output_dir = f"{self.data_directory}/{output_dir}"
        else:
            output_dir = f"{self.data_directory}/{self.id}_h5_files"

        # make sure output path exists, create directory if not
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Exporting velocity to HDF5...")

        with h5py.File(f"{output_dir}/{self.id}_vel.h5", "w") as f:
            f.create_dataset("velocity", data=self.flow_data)

        with h5py.File(f"{output_dir}/{self.id}_masks.h5", "w") as f:
            f.create_dataset("mask", data=self.mask)
            f.create_dataset("inlet", data=self.inlet)
            f.create_dataset("outlet", data=self.outlet)

    def add_skeleton(self) -> None:
        """Create a skeleton from the segmentation and plot it."""
        skel_image, skel_rough, self.skeleton, self.skeleton_ddx = sm.smooth_skeletonize(self.segmentation)
        pr.plot_seg_skeleton(self.segmentation, skel_image, skel_rough, self.skeleton)

    def draw_planes(self) -> None:
        """Call plane drawing tool and add inlet/outlet masks."""
        self.outlet, self.inlet = sm.plane_drawer(self.segmentation, self.skeleton, self.skeleton_ddx)
        self.inlet = ndimage.binary_dilation(self.inlet) * self.mask
        self.outlet = ndimage.binary_dilation(self.outlet) * self.mask

    def get_ste_drop(self) -> None:
        """Use matlabengine to call STE MATLAB function and estimate a pressure drop.

        It is significantly more reliable to load the .mat data in the MATLAB function than to pass it as an argument
        in this function. Make sure to export the data to .mat files with export_to_mat() before running this function.
        """
        output = run_matlab_function(
            "/Applications/MATLAB_R2025b.app/bin/matlab",
            "/Users/bkhardy/Developer/GitHub/flow_tools",
            "out.mat",
            "ste",
            f"{self.data_directory}/{self.id}_mat_files/{self.id}_vel.mat",
            f"{self.data_directory}/{self.id}_mat_files/{self.id}_masks.mat",
            "resample",
            2,
        )

        self.times = np.array(output["times"]).flatten()
        self.dp_STE = np.array(output["dP"]).flatten() * PA_TO_MMHG
        self.p_STE = np.array(output["P"]) * PA_TO_MMHG

    def export_p_field_to_vti(self, output_dir: None | str = None) -> None:
        """Export pressure field to VTK ImageData format.

        Args:
            output_dir (None | str, optional): Path to vti output directory. Defaults to None to autogenerate directory.

        """
        if output_dir is not None:
            output_dir = f"{self.data_directory}/{output_dir}"
        else:
            output_dir = f"{self.data_directory}/{self.id}_STE_vti"

        # make sure output path exists, create directory if not
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Exporting pressure to VTI...")
        for t in tqdm(range(self.p_STE.shape[-1])):
            # write pressure field one timestep at a time
            p = self.p_STE[:, :, :, t].copy()  # * self.mask
            out_path = f"{output_dir}/{self.id}_p_STE_{t:03d}.vti"

            frame = pv.ImageData(dimensions=self.p_STE.shape[:-1], spacing=self.dx / 2)
            frame.cell_data["Relative Pressure"] = p.flatten()
            frame.save(out_path)

            # NOTE: CURRENTLY ASSUMING RESAMPLING FACTOR of 2...

    def plot_dp(self) -> None:
        """Plot estimated pressure drop over time. Exports as a pdf file."""
        figure = pr.plot_dp(self.times, self.dp_STE, self.id)
        figure.savefig(f"{self.id}_STE_dP.pdf")

    def paraview_analysis(self) -> None:
        """Run pvpython script for post-processing.

        This script will output a video showing the velocity and pressure fields over time.
        """
        subprocess.run(
            [
                PVPYTHON_PATH,
                "../paraview_scripts/paraview_trace.py",
                self.id,
                self.data_directory,
                str(self.mag_data.shape[-1]),
            ],
            check=False,
        )

    def export_to_nifti(self) -> None:
        """Export SSFP image to Nifti format for testing segmentation U-Net."""
        self.mask = np.transpose(self.mask, (2, 1, 0))
        self.flow_data = np.transpose(self.flow_data, (0, 3, 2, 1, 4))
        self.ssfp_data = np.transpose(self.ssfp_data, (2, 1, 0))
        self.ssfp_data = np.flip(self.ssfp_data, axis=(1, 2))
        self.flow_data = np.flip(self.flow_data, axis=0)

        self.convert_to_vti()

        img = nib.Nifti1Image(self.ssfp_data.astype(np.int16).copy(), np.eye(4))

        nib.save(img, "test.nii.gz")


def run_matlab_function(
    matlab_exe: str,
    matlab_path: str | Path,
    out_mat: str | Path,
    *args: str,
    wrapper: str = "pressure_wrapper",
    timeout_s: int = 300,
) -> dict:
    """Run MATLAB wrapper via subprocess and return loaded outputs.

    matlab_exe: path to matlab executable (or just "matlab" if on PATH)
    matlab_path: folder containing pressure_wrapper.m and dependencies
    out_mat: output .mat file path
    args: arguments forwarded to run_myfunc
    """
    matlab_path = str(Path(matlab_path).resolve())
    out_mat = str(Path(out_mat).resolve())

    # Build MATLAB command string safely.
    # Use double quotes in MATLAB strings; escape embedded quotes.
    def mstr(x: str | int | bool) -> str:
        if isinstance(x, str):
            return '"' + x.replace('"', '""') + '"'
        if isinstance(x, bool):
            return "true" if x else "false"
        if isinstance(x, int):
            return str(x)
        raise TypeError(f"Unexpected type {type(x)} for mstr")

    # Add paths and call wrapper
    matlab_cmd = (
        f"addpath(genpath({mstr(matlab_path)})); {wrapper}({mstr(out_mat) + ''.join([', ' + mstr(a) for a in args])});"
    )
    print(matlab_cmd)

    cmd = [matlab_exe, "-batch", matlab_cmd]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )

    # If MATLAB errors, you'll often still get stderr + (maybe) out_mat with err struct.
    if proc.returncode != 0:
        extra = ""
        if Path(out_mat).exists():
            try:
                m = sio.loadmat(out_mat, simplify_cells=True)
                if "err" in m:
                    extra = f"\nMATLAB err.message: {m['err'].get('message')}"
            except Exception:
                pass
        raise RuntimeError(
            f"MATLAB failed (code {proc.returncode}).\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}{extra}",
        )

    if not Path(out_mat).exists():
        raise FileNotFoundError(f"MATLAB finished but did not create output file: {out_mat}")

    try:
        mat = {}
        with h5py.File(out_mat, "r") as f:
            for key in f:
                mat[key] = np.asarray(f[key]).T
    except OSError:
        # Fall back to scipy for older MATLAB versions that don't support HDF5 output
        mat = sio.loadmat(out_mat, simplify_cells=True)

    if "err" in mat:
        raise RuntimeError(f"MATLAB saved an error struct: {mat['err']}")

    # if "out" not in mat:
    #    raise KeyError(f"Expected variable 'out' in {out_mat}, got keys: {list(mat.keys())}")

    return mat

import subprocess
from pathlib import Path

import matlab.engine
import nibabel as nib
import numpy as np
import plot_results as pr
import read_dicoms as rd
import seg_module as sm  # init seems to have fixed these paths??? very interesting...
from pyevtk.hl import imageToVTK
from scipy import ndimage
from tqdm import tqdm

PA_TO_MMHG = 0.00750061683
PVPYTHON_PATH = "/Applications/ParaView-5.13.1.app/Contents/bin/pvbatch"


class Patient4DFlow:
    def __init__(
        self, patient_id: str, data_directory: str, seg_path: str = "user"
    ) -> None:
        self.id = patient_id
        self.data_directory = data_directory
        self.mag_data, self.ssfp_data, self.flow_data, self.dx, self.dt = (
            rd.import_all_dicoms(self.data_directory)
        )

        self.flow_data = np.flip(self.flow_data, axis=0)
        self.flow_data[0] *= -1
        self.flow_data = self.flow_data.copy()

        # NOTE: currently not following segmentation pattern shown below... inlet/outlet numbers are temp values
        self.segmentation, self.seg_origin, self.seg_spacing = self.add_segmentation(
            seg_path
        )
        self.mask = np.array(self.segmentation != 0).astype(np.float64).copy()
        self.inlet = np.array(self.segmentation == 2).astype(np.float64).copy()
        self.inlet = ndimage.binary_dilation(self.inlet) * self.mask
        self.outlet = np.array(self.segmentation == 3).astype(np.float64).copy()
        self.outlet = ndimage.binary_dilation(self.outlet) * self.mask

        # NOTE: TEMPORARY VALUES BEFORE I FIX EVERYTHING
        self.res = np.array(self.mag_data.shape)

    def __str__(self):
        return f"Patient ID: {self.id} @ location {self.data_directory}"

    def add_segmentation(self, path_input):
        if path_input == "user":
            seg_path = input("Enter relative path to segmentation: ")
        else:
            seg_path = path_input

        segmentation, origin, spacing = rd.import_segmentation(
            self.data_directory + seg_path
        )

        # PROBABLY GOING TO BE TEMPORARY CODE AS I WORK THINGS OUT
        segmentation = np.transpose(segmentation, (1, 0, 2))
        segmentation = np.flip(segmentation, axis=2)

        return segmentation, origin, spacing

    def check_orientation(self):
        mag = self.mag_data[:, :, :, 6].copy()

        u = self.flow_data[0, :, :, :, 6].copy() * self.mask
        v = self.flow_data[1, :, :, :, 6].copy() * self.mask
        w = self.flow_data[2, :, :, :, 6].copy() * self.mask
        vel = (u, v, w)

        imageToVTK(f"{self.id}_check_mag", cellData={"Magnitude": mag})
        imageToVTK(f"{self.id}_check_vel", cellData={"Velocity": vel})

        # unfortunately it seems like the only solution here is to write a timeframe to disk then load
        # back in. That sucks and is inefficient but whatever.

    def convert_to_vti(self, output_dir: None | str = None) -> None:
        if output_dir is not None:
            output_dir = f"{self.data_directory}/{output_dir}"
        else:
            output_dir = f"{self.data_directory}/{self.id}_flow_vti"

        # make sure output path exists, create directory if not
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print("\nExporting to VTI...")
        for t in tqdm(range(self.flow_data.shape[-1])):
            # write velocity field one timestep at a time
            u = self.flow_data[0, :, :, :, t].copy() * self.mask
            v = self.flow_data[1, :, :, :, t].copy() * self.mask
            w = self.flow_data[2, :, :, :, t].copy() * self.mask
            vel = (u, v, w)

            out_path = f"{output_dir}/{self.id}_flow_{t:03d}"

            imageToVTK(out_path, spacing=self.dx.tolist(), cellData={"Velocity": vel})

    def export_to_mat(self, output_dir: None | str = None) -> None:
        eng = matlab.engine.start_matlab()

        if output_dir is not None:
            output_dir = f"{self.data_directory}/{output_dir}"
        else:
            output_dir = f"{self.data_directory}/{self.id}_mat_files"

        # make sure output path exists, create directory if not
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # navigate MATLAB instance to current working directory to call custom function
        print("Exporting velocity structs...")
        eng.addpath(eng.genpath("Patient4DFlow"))
        eng.export_struct(
            output_dir + f"/{self.id}_vel.mat",
            self.flow_data,
            self.dx,
            self.dt,
            self.res,
            nargout=0,
        )

        print("Exporting masks...")
        eng.export_masks(
            output_dir + f"/{self.id}_masks.mat",
            self.mask,
            self.inlet,
            self.outlet,
            nargout=0,
        )

        eng.quit()

    # FIXME: Add interactivity! Add plane drawing!!! MAYBE SPLIT THIS UP
    def add_skeleton(self):
        skel_image, skel_rough, self.skeleton, self.skeleton_ddx = (
            sm.smooth_skeletonize(self.segmentation)
        )
        pr.plot_seg_skeleton(self.segmentation, skel_image, skel_rough, self.skeleton)

    def draw_planes(self):
        sm.plane_drawer(self.segmentation, self.skeleton, self.skeleton_ddx)

    def get_ste_drop(self):
        """_summary_"""
        eng = matlab.engine.start_matlab()

        eng.addpath(eng.genpath("../vwerp"))

        times, dp_drop, dp_field = eng.get_ste_pressure_estimate_py(
            f"{self.data_directory}/{self.id}_mat_files/{self.id}_vel.mat",
            f"{self.data_directory}/{self.id}_mat_files/{self.id}_masks.mat",
            nargout=3,
        )

        eng.quit()

        self.times = np.array(times).flatten()
        self.dp_STE = np.array(dp_drop).flatten() * PA_TO_MMHG
        self.p_STE = np.array(dp_field) * PA_TO_MMHG

    def export_p_field(self, output_dir: None | str = None) -> None:
        if output_dir is not None:
            output_dir = f"{self.data_directory}/{output_dir}"
        else:
            output_dir = f"{self.data_directory}/{self.id}_STE_vti"

        # make sure output path exists, create directory if not
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print("\nExporting pressure to VTI...")
        for t in tqdm(range(self.p_STE.shape[-1])):
            # write pressure field one timestep at a time
            p = self.p_STE[:, :, :, t].copy()  # * self.mask
            out_path = f"{output_dir}/{self.id}_p_STE_{t:03d}"

            # NOTE: CURRENTLY ASSUMING RESAMPLING FACTOR of 2...
            imageToVTK(
                out_path, spacing=(self.dx / 2).tolist(), cellData={"Pressure": p}
            )

    def plot_dp(self) -> None:
        """Function to plot the pressure drop over time. Exports as a pdf file."""
        figure = pr.plot_dp(self.times, self.dp_STE, self.id)
        figure.savefig(f"{self.id}_STE_dP.pdf")

    def paraview_analysis(self) -> None:
        """Function to run pvparaview script. This script will output a video showing the
        velocity and pressure fields over time.
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

    def export_to_nifti(self):
        self.mask = np.transpose(self.mask, (2, 1, 0))
        self.flow_data = np.transpose(self.flow_data, (0, 3, 2, 1, 4))
        self.ssfp_data = np.transpose(self.ssfp_data, (2, 1, 0))
        self.ssfp_data = np.flip(self.ssfp_data, axis=(1, 2))
        self.flow_data = np.flip(self.flow_data, axis=0)

        self.convert_to_vti()

        img = nib.Nifti1Image(self.ssfp_data.astype(np.int16).copy(), np.eye(4))

        nib.save(img, "test.nii.gz")


def full_run(patient_id, data_path, seg_path):
    patient = Patient4DFlow(patient_id, data_path, seg_path)
    patient.add_skeleton()
    patient.convert_to_vti()
    patient.export_to_mat()
    patient.get_ste_drop()
    patient.export_p_field()
    patient.plot_dp()


def main():
    patient = Patient4DFlow(
        "Prab",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/",
        "Segmentation.nrrd",
    )

    patient.add_skeleton()
    patient.draw_planes()


if __name__ == "__main__":
    main()

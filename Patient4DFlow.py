from pathlib import Path

import matlab.engine
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from pyevtk.hl import imageToVTK
from scipy import ndimage
from tqdm import tqdm

import plot_results as pr
import read_dicoms as rd
import seg_module as sm

PA_TO_MMHG = 0.00750061683


class Patient4DFlow:
    def __init__(self, ID: str, data_directory: str, seg_path: str = "user") -> None:
        self.ID = ID
        self.dir = data_directory
        self.mag_data, self.flow_data, self.dx, self.dt = rd.import_all_dicoms(self.dir)

        self.flow_data = np.flip(self.flow_data, axis=0)
        self.flow_data[0] *= -1
        self.flow_data = self.flow_data.copy()

        self.segmentation = self.add_segmentation(seg_path)
        self.mask = (
            np.array(self.segmentation != 0).astype(np.float64).copy()
        )  # ALL NON-ZERO VALUES

        self.inlet = (
            np.array(self.segmentation == 2).astype(np.float64).copy()
        )  # ALL TWO'S
        self.inlet = ndimage.binary_dilation(self.inlet) * self.mask

        self.outlet = (
            np.array(self.segmentation == 3).astype(np.float64).copy()
        )  # ALL THREE'S
        self.outlet = ndimage.binary_dilation(self.outlet) * self.mask

        # NOTE: TEMPORARY VALUES BEFORE I FIX EVERYTHING
        self.res = np.array(self.mag_data.shape)

        # self.mag_data = self.add_mag(path_input="user")
        # self.flow_data = self.add_flow(path_input="user")
        # self.segmentation = self.add_segmentation(path_input="user")

    def __str__(self):
        return f"Patient ID: {self.ID} @ location {self.dir}"

    def add_mag(self, path_input):
        if path_input == "user":
            mag_path = input("Enter relative path to magnitude data: ")
        else:
            mag_path = path_input

        mag_data = rd.import_mag(self.dir + mag_path)

        return mag_data

    def add_flow(
        self, path_input: str = "user"
    ) -> tuple[np.ndarray, np.ndarray, np.floating]:

        if path_input == "user":
            u_path = input("Enter relative path to u data: ")
            v_path = input("Enter relative path to v data: ")
            w_path = input("Enter relative path to w data: ")
        else:
            u_path, v_path, w_path = path_input

        flow_data, dx, dt = rd.import_flow(
            (self.dir + u_path, self.dir + v_path, self.dir + w_path)
        )

        return flow_data, dx, dt

    def add_segmentation(self, path_input):

        if path_input == "user":
            seg_path = input("Enter relative path to segmentation: ")
        else:
            seg_path = path_input

        segmentation = rd.import_segmentation(self.dir + seg_path)

        # PROBABLY GOING TO BE TEMPORARY CODE AS I WORK THINGS OUT
        segmentation = np.transpose(segmentation, (1, 0, 2))
        segmentation = np.flip(segmentation, axis=2)

        return segmentation

    def check_orientation(self):

        mag = self.mag_data[:, :, :, 6].copy()

        u = self.flow_data[0, :, :, :, 6].copy() * self.mask
        v = self.flow_data[1, :, :, :, 6].copy() * self.mask
        w = self.flow_data[2, :, :, :, 6].copy() * self.mask
        vel = (u, v, w)

        imageToVTK(f"{self.ID}_check_mag", cellData={"Magnitude": mag})
        imageToVTK(f"{self.ID}_check_vel", cellData={"Velocity": vel})

        # unfortunately it seems like the only solution here is to write a timeframe to disk then load
        # back in. That sucks and is inefficient but whatever.

    # NOTE: I'm currently having a user input the paths directly, but this could definitely get tedious (especially since DICOM paths are evil and not even close to being straightforward/intuitive).
    # I will definitely want to automate this process but unfortunately, again, DICOMs are evil and I don't know how to parse their metadata completely yet...

    def convert_to_vti(self, output_dir: None | str = None) -> None:

        if output_dir is not None:
            output_dir = f"{self.dir}/{output_dir}"
        else:
            output_dir = f"{self.dir}/{self.ID}_flow_vti"

        # make sure output path exists, create directory if not
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print("\nExporting to VTI...")
        for t in tqdm(range(self.flow_data.shape[-1])):

            # write velocity field one timestep at a time
            u = self.flow_data[0, :, :, :, t].copy() * self.mask
            v = self.flow_data[1, :, :, :, t].copy() * self.mask
            w = self.flow_data[2, :, :, :, t].copy() * self.mask
            vel = (u, v, w)

            out_path = f"{output_dir}/{self.ID}_flow_{t:03d}"

            imageToVTK(out_path, spacing=self.dx.tolist(), cellData={"Velocity": vel})

    def export_to_mat(self, output_dir: None | str = None) -> None:
        eng = matlab.engine.start_matlab()

        if output_dir is not None:
            output_dir = f"{self.dir}/{output_dir}"
        else:
            output_dir = f"{self.dir}/{self.ID}_mat_files"

        # make sure output path exists, create directory if not
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # navigate MATLAB instance to current working directory to call custom function
        print("Exporting velocity structs...")
        eng.export_struct(
            output_dir + f"/{self.ID}_vel.mat",
            self.flow_data,
            self.dx,
            self.dt,
            self.res,
            nargout=0,
        )

        print("Exporting masks...")
        eng.export_masks(
            output_dir + f"/{self.ID}_masks.mat",
            self.mask,
            self.inlet,
            self.outlet,
            nargout=0,
        )

        eng.quit()

        # now call matlab to more easily assemble vWERP/STE/PPE compatible structs
        # function should not return anything...

    # FIXME: Add interactivity! Add plane drawing!!! MAYBE SPLIT THIS UP
    def add_skeleton(self):
        skel_image, skel_rough, self.skeleton, self.skeleton_derivative = (
            sm.smooth_skeletonize(self.segmentation)
        )  # FEEDING IN SEGMENTATION INSTEAD... WEIRD? NOTE: feeding in the full segmentation (with data range from 1 to 3) works better when visualizing... weird.
        pr.plot_seg_skeleton(self.segmentation, skel_image, skel_rough, self.skeleton)

    def draw_planes(self):
        sm.plane_drawer(self.segmentation, self.skeleton, self.skeleton_derivative)

    def get_ste_drop(self):
        """_summary_"""
        eng = matlab.engine.start_matlab()

        eng.addpath(eng.genpath("vwerp"))

        times, dP, P = eng.get_ste_pressure_estimate_py(
            f"{self.dir}/{self.ID}_mat_files/{self.ID}_vel.mat",
            f"{self.dir}/{self.ID}_mat_files/{self.ID}_masks.mat",
            nargout=3,
        )

        eng.quit()

        self.times = np.array(times).flatten()
        self.dp_STE = np.array(dP).flatten() * PA_TO_MMHG
        self.p_STE = np.array(P) * PA_TO_MMHG

    def plot_dp(self):
        figure = pr.plot_dp(self.times, self.dp_STE, self.ID)
        plt.show()


def um19_check():
    patient_UM19 = Patient4DFlow(
        "UM19",
        "/Users/bkhardy/Dropbox (University of Michigan)/MRI_1.22.24/DICOM/0000A628/AAD75E3C/AA62C567/",
        "Segmentation.nrrd",
    )

    patient_UM19.check_orientation()
    patient_UM19.convert_to_vti()
    patient_UM19.export_to_mat()


def full_run(patient_id, data_path, seg_path):
    patient = Patient4DFlow(patient_id, data_path, seg_path)

    patient.add_skeleton()
    patient.convert_to_vti()
    patient.export_to_mat()
    patient.get_ste_drop()
    patient.plot_dp()


def main():
    prab = Patient4DFlow(
        "Prab",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/",
        "Segmentation.nrrd",
    )

    prab.add_skeleton()
    prab.draw_planes()


if __name__ == "__main__":
    main()

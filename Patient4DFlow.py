from pathlib import Path

import h5py
import matlab.engine
import nrrd
import numpy as np
import pyvista as pv
from pyevtk.hl import imageToVTK

import read_dicoms as rd


class Patient4DFlow:
    def __init__(self, ID, data_directory):
        self.ID = ID
        self.dir = data_directory
        self.mag_data, self.flow_data = rd.import_all_dicoms(self.dir)
        self.segmentation = self.add_segmentation("Segmentation.nrrd")

        # NOTE: TEMPORARY VALUES BEFORE I FIX EVERYTHING
        self.dt = 0.050
        self.dx = np.array([0.002, 0.002, 0.002])
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

        mag_data = rd.import_dicoms(self.dir + mag_path)

        return mag_data

    def add_flow(self, path_input):

        if path_input == "user":
            u_path = input("Enter relative path to u data: ")
            v_path = input("Enter relative path to v data: ")
            w_path = input("Enter relative path to w data: ")
        else:
            u_path, v_path, w_path = path_input

        flow_data = rd.import_flow(
            (self.dir + u_path, self.dir + v_path, self.dir + w_path)
        )

        return flow_data

    def add_segmentation(self, path_input):

        if path_input == "user":
            seg_path = input("Enter relative path to segmentation: ")
        else:
            seg_path = path_input

        segmentation = rd.import_segmentation(self.dir + seg_path)

        # PROBABLY GOING TO BE TEMPORARY CODE AS I WORK THINGS OUT
        segmentation = np.transpose(segmentation, (1, 0, 2))
        segmentation = np.flip(segmentation, axis=2)
        nrrd.write("Segmentation_transposed.nrrd", segmentation)

        return segmentation

    def check_orientation(self):

        mag = self.mag_data[:, :, :, 6].copy()

        u = self.flow_data[0, :, :, :, 6].copy() * self.segmentation
        v = self.flow_data[1, :, :, :, 6].copy() * self.segmentation
        w = self.flow_data[2, :, :, :, 6].copy() * self.segmentation
        vel = (-w, v, u)

        imageToVTK("UM19_check_mag", cellData={"Magnitude": mag})
        imageToVTK("UM19_check_vel", cellData={"Velocity": vel})

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

        for t in range(self.flow_data.shape[-1]):

            # write velocity field one timestep at a time
            u = self.flow_data[0, :, :, :, t].copy() * self.segmentation
            v = self.flow_data[1, :, :, :, t].copy() * self.segmentation
            w = self.flow_data[2, :, :, :, t].copy() * self.segmentation
            vel = (-w, v, u)

            out_path = f"{output_dir}/{self.ID}_flow_{t:03d}"

            imageToVTK(out_path, spacing=[1, 1, 1], cellData={"Velocity": vel})

    def export_to_mat_struct(self, output_dir: None | str = None) -> None:
        eng = matlab.engine.start_matlab()

        if output_dir is not None:
            output_dir = f"{self.dir}/{output_dir}"
        else:
            output_dir = f"{self.dir}/{self.ID}_mat_files"

        # make sure output path exists, create directory if not
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # navigate MATLAB instance to current working directory to call custom function
        eng.cd(str(Path.cwd()))
        eng.export_struct(
            output_dir + f"/{self.ID}_vel.mat",
            self.flow_data,
            self.dx,
            self.dt,
            self.res,
            nargout=0,
        )

        eng.quit()

        # now call matlab to more easily assemble vWERP/STE/PPE compatible structs
        # function should not return anything...


def main():

    patient_UM19 = Patient4DFlow(
        "UM19",
        "/Users/bkhardy/Dropbox (University of Michigan)/MRI_1.22.24/DICOM/0000A628/AAD75E3C/AA62C567/",
    )

    patient_UM19.check_orientation()
    patient_UM19.convert_to_vti()
    patient_UM19.export_to_mat_struct()


if __name__ == "__main__":
    main()

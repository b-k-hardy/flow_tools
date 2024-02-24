from pathlib import Path

import h5py
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

        # self.mag_data = self.add_mag(path_input="user")
        # self.flow_data = self.add_flow(path_input="user")
        self.segmentation = self.add_segmentation(path_input="user")

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

    def convert_to_vti(self, output_dir=None):

        for t in range(self.flow_data.shape[-1]):

            # write velocity field one timestep at a time
            u = self.flow_data[0, :, :, :, t].copy() * self.segmentation
            v = self.flow_data[1, :, :, :, t].copy() * self.segmentation
            w = self.flow_data[2, :, :, :, t].copy() * self.segmentation
            vel = (-w, v, u)

            if output_dir is not None:
                out_path = f"{self.dir}/{output_dir}/{self.ID}_flow_{t:03d}"
            else:
                out_path = f"{self.dir}/{self.ID}_flow_vti/{self.ID}_vel_{t:03d}"

            # make sure output path exists, create directory if not
            Path(out_path).mkdir(parents=True, exist_ok=True)

            imageToVTK(out_path, spacing=[1, 1, 1], cellData={"Velocity": vel})


def main():

    patient_UM19 = Patient4DFlow(
        "UM19",
        "/Users/bkhardy/Dropbox (University of Michigan)/MRI_1.22.24/DICOM/0000A628/AAD75E3C/AA62C567/",
    )

    print(patient_UM19)
    patient_UM19.check_orientation()
    patient_UM19.convert_to_vti()


if __name__ == "__main__":
    main()

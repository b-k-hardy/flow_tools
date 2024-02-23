from pyevtk.hl import imageToVTK
import read_dicoms as rd
import nrrd
import numpy as np


class Patient4DFlow:
    def __init__(self, ID, data_directory):
        self.ID = ID
        self.dir = data_directory
        self.mag_data = ""
        self.flow_data = ""
        self.segmentation = ""

    def add_mag(self):
        mag_path = input("Enter relative path to magnitude data: ")
        self.mag_data = rd.import_dicoms(self.dir + mag_path)

    def add_flow(self):
        u_path = input("Enter relative path to u data: ")
        v_path = input("Enter relative path to v data: ")
        w_path = input("Enter relative path to w data: ")
        self.flow_data = rd.import_flow(
            self.dir + u_path, self.dir + v_path, self.dir + w_path
        )

    def add_segmentation(self):
        seg_path = input("Enter relative path to segmentation: ")
        self.segmentation = rd.import_segmentation(self.dir + seg_path)

        # PROBABLY GOING TO BE TEMPORARY CODE AS I WORK THINGS OUT
        self.segmentation = np.transpose(self.segmentation, (1, 0, 2))
        self.segmentation = np.flip(self.segmentation, axis=2)
        nrrd.write("Segmentation_transposed.nrrd", self.segmentation)

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

    def __str__(self):
        return f"Patient ID: {self.ID} @ location {self.dir}"

    # NOTE: I'm currently having a user input the paths directly, but this could definitely get tedious (especially since DICOM paths are evil and not even close to being straightforward/intuitive).
    # I will definitely want to automate this process but unfortunately, again, DICOMs are evil and I don't know how to parse their metadata completely yet...


def main():

    patient_UM19 = Patient4DFlow(
        "UM19",
        "/Users/bkhardy/Dropbox (University of Michigan)/MRI_1.22.24/DICOM/0000A628/AAD75E3C/AA62C567/",
    )

    print(patient_UM19)
    patient_UM19.add_segmentation()
    patient_UM19.add_mag()
    patient_UM19.add_flow()
    patient_UM19.check_orientation()

    # mag data in   00003965/*x
    # u data in     0000ED0F/*
    # v data in     0000D7F0/*
    # w data in     00004E45/*


if __name__ == "__main__":
    main()

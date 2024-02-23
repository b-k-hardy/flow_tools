import pydicom
import h5py
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from pyevtk.hl import imageToVTK
import nrrd


def import_segmentation(seg_path):
    segmentation, header = nrrd.read(seg_path)
    print(segmentation.shape)
    print(header)

    return segmentation


def import_flow(u_path, v_path, w_path):

    paths = [u_path, v_path, w_path]
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

        for i in range(Nt):
            timestep = slices[Nz * i : Nz * (i + 1)]
            timestep.sort(key=lambda s: s.SliceLocation)
            for j in range(Nz):
                img_step = timestep[j]
                img2d = (
                    img_step.pixel_array * img_step.RescaleSlope
                    + img_step.RescaleIntercept
                )
                img4d[:, :, j, i] = img2d

        # plot 3 orthogonal slices to check for correct indexing
        a1 = plt.subplot(2, 2, 1)
        plt.imshow(img4d[:, :, img_shape[2] // 2, 5], cmap="gray")
        a1.set_aspect(ax_aspect)

        a2 = plt.subplot(2, 2, 2)
        plt.imshow(img4d[:, img_shape[1] // 2, :, 5], cmap="gray")
        a2.set_aspect(sag_aspect)

        a3 = plt.subplot(2, 2, 3)
        plt.imshow(img4d[img_shape[0] // 2, :, :, 5].T, cmap="gray")
        a3.set_aspect(cor_aspect)

        plt.show()

        # fill 4D array with the images from the files

        img5d.append(img4d)

    return np.asarray(img5d)


def import_dicoms(dicom_path):
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

    # plot 3 orthogonal slices to check for correct indexing
    a1 = plt.subplot(2, 2, 1)
    plt.imshow(img4d[:, :, img_shape[2] // 2, 5], cmap="gray")
    a1.set_aspect(ax_aspect)

    a2 = plt.subplot(2, 2, 2)
    plt.imshow(img4d[:, img_shape[1] // 2, :, 5], cmap="gray")
    a2.set_aspect(sag_aspect)

    a3 = plt.subplot(2, 2, 3)
    plt.imshow(img4d[img_shape[0] // 2, :, :, 5].T, cmap="gray")
    a3.set_aspect(cor_aspect)

    plt.show()

    return img4d


class Patient4DFlow:
    def __init__(self, ID, data_directory):
        self.ID = ID
        self.dir = data_directory
        self.mag_data = ""
        self.flow_data = ""
        self.segmentation = ""

    def add_mag(self):
        mag_path = input("Enter relative path to magnitude data: ")
        self.mag_data = import_dicoms(self.dir + mag_path)

    def add_flow(self):
        u_path = input("Enter relative path to u data: ")
        v_path = input("Enter relative path to v data: ")
        w_path = input("Enter relative path to w data: ")
        self.flow_data = import_flow(
            self.dir + u_path, self.dir + v_path, self.dir + w_path
        )

    def add_segmentation(self):
        seg_path = input("Enter relative path to segmentation: ")
        self.segmentation = import_segmentation(self.dir + seg_path)

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


def convert_vti(data_path, output_dir, output_filename):

    Nt = 0
    dx = 0

    for t in range(Nt):
        # open additional mat files mat file in series (silly string concatenation here...)
        with h5py.File(data_path + f"_{t+1}.mat", "r") as f:
            # get pointers for velocity struct
            v_pointers = f["v"][:]

            # access the images (matlab equivalent: v{1}.im)
            u = f[v_pointers[0, 0]]["im"][:].T
            v = f[v_pointers[1, 0]]["im"][:].T
            w = f[v_pointers[2, 0]]["im"][:].T

        # write velocity field one timestep at a time
        vel = (u, v, w)
        out_path = f"{output_dir}/{output_filename}_{t:03d}"
        imageToVTK(out_path, spacing=dx, cellData={"Velocity": vel})


def main():

    # NOTE: I'm currently having a user input the paths directly, but this could definitely get tedious (especially since DICOM paths are evil and not even close to being straightforward/intuitive).
    # I will definitely want to automate this process but unfortunately, again, DICOMs are evil and I don't know how to parse their metadata completely yet...

    patient_UM19 = Patient4DFlow(
        "UM19",
        "/Users/bkhardy/Dropbox (University of Michigan)/MRI_1.22.24/DICOM/0000A628/AAD75E3C/AA62C567/",
    )

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

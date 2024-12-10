"""Driver code to run analysis of 4D flow patient data and log code execution."""

import logging
import sys

from patient4Dflow import patient_class

logger = logging.getLogger(__name__)


def full_run(patient_id: str, data_path: str, seg_path: str) -> None:
    """Run full analysis pipeline for a single patient.

    This function can be run directly in the driver script to run the full analysis pipeline. Additionally, this
    function can be used as inspiration to show how the sub-functions are generally ordered. Of note:
    1. The first step is always to initialize the Patient4DFlow object.
    2. For any analysis, a skeleton will need to be added
    3. Exporting the velocity data to VTK format is a common step, but not required
    4. Exporting the data to .mat MATLAB format is required for the STE pressure drop estimation
    5. The STE pressure drop estimation is a key part of the analysis and required for the remaining steps
    6. Exporting the pressure field to VTK format is optional, but can be useful for visualization
    7. Plotting the pressure drop over time is a key part of the analysis

    Args:
        patient_id (str): Anonymized patient ID
        data_path (str): Absolute path to patient data directory
        seg_path (str): Relative path from data_path to segmentation file

    """
    patient = patient_class.Patient4DFlow(patient_id, data_path, seg_path)
    patient.add_skeleton()
    patient.export_vel_to_vti()
    patient.export_to_mat()
    patient.get_ste_drop()
    patient.export_p_field()
    patient.plot_dp()


def main() -> None:
    """Run and log analysis of 4D flow data."""
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    patient = patient_class.Patient4DFlow(
        "Prab",
        "/Users/bkhardy/Dropbox (University of Michigan)/4D Flow Test Data/Prab 9.27.23/",
        "Segmentation.nrrd",
    )

    patient.add_skeleton()
    patient.draw_planes()
    patient.export_to_mat()
    patient.get_ste_drop()
    patient.export_p_field_to_vti()
    patient.plot_dp()


if __name__ == "__main__":
    main()

"""Driver code to run analysis of 4D flow patient data and log code execution."""

import logging
import sys

import patient4Dflow.patient_class

logger = logging.getLogger(__name__)


def main() -> None:
    """Run and log analysis of 4D flow data."""
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    patient = patient4Dflow.patient_class.Patient4DFlow(
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

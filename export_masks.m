function export_masks(save_path, mask, inlet, outlet)
% EXPORT_STRUCT  Export DICOM data in Python class to vWERP/STE/PPE -compatible struct.
%   Note that this function purely exists to be called from Python
%   Output: None
%   Input:  mask, inlet, and outlet from 3D Slicer

% NOTE: I HAVE HARDCODED IN OUR vWERP DIRECTORY... USER WILL NEED TO CHANGE THIS
addpath(genpath("../../vwerp"))

outlet = DilateMask(outlet) .* mask;
inlet = DilateMask(inlet) .* mask;

save(save_path, "mask", "inlet", "outlet", "-v7.3")

end
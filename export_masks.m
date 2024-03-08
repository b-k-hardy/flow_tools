function export_masks(save_path, mask, inlet, outlet)
% EXPORT_STRUCT  Export DICOM data in Python class to vWERP/STE/PPE -compatible struct.
%   Note that this function purely exists to be called from Python
%   Output: None
%   Input:  mask, inlet, and outlet from 3D Slicer

save(save_path, "mask", "inlet", "outlet", "-v7.3")

end
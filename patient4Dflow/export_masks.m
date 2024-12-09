function export_masks(save_path, mask, inlet, outlet)
% EXPORT_MASKS  Export segmentation data in Python class to MATLAB
%   Note that this function purely exists to be called from Python
%   Output: None
%   Input:  mask, inlet, and outlet from 3D Slicer

save(save_path, "mask", "inlet", "outlet", "-v7.3")

end
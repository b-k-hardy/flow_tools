function export_struct(save_path, vel_data, dx, dt, res)
% EXPORT_STRUCT  Export DICOM data in Python class to vWERP/STE/PPE -compatible struct.
%   Note that this function purely exists to be called from Python
%   Output: None
%   Input:  Velocity data, voxel size, temporal resolution, spatial resolution (grid size)

v = cell(1,3);

for i=1:3

    v{i}.im = squeeze(vel_data(i,:,:,:,:));
    v{i}.PixDim = dx;
    v{i}.dt = dt;
    v{i}.res = res;

end

save(save_path, "v", "-v7.3")

end
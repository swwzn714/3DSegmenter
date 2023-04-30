"""visualize tsdf as mesh"""
import os
import random
import numpy as np
import open3d as o3d
import glob
import fusion

if __name__ == "__main__":
    filedir = "Path to PanoRooms"
    filelist = glob.glob(os.path.join(filedir, "*.npz"))
    random.shuffle((filelist))
    for filename in filelist:
        vol_file = np.load(filename)
        print("Estimating voxel volume bounds...")
        vol_bnds = np.zeros((3,2))
        print("Initializing voxel volume...")
        voxel_size=0.04
        vol = vol_file["sdf128"]
        vol_bnds[:,1]=voxel_size*np.asarray(vol.shape)
        tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_size)
        tsdf_vol.write_volume(vol)

        rgb_int = vol_file["color128"]
        tsdf_vol.write_color(rgb_int)
        print("Saving mesh to mesh.ply...")
        verts, faces, norms, colors = tsdf_vol.get_mesh()
        fusion.meshwrite("mesh.ply", verts, faces, norms, colors)
        mesh = o3d.io.read_triangle_mesh("mesh.ply")
        o3d.visualization.draw_geometries([mesh],mesh_show_back_face=True)
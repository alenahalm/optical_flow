import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


pcd = o3d.io.read_point_cloud('./pointCloud.ply')
o3d.visualization.draw_geometries([pcd])
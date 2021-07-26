import loader
from KITTI_loader import LoaderKITTI
from Lidar_Mapping import Mapper
from odometry_estimator import OdometryEstimator
from utils import get_pcd_from_numpy
import open3d as o3d


FEATURES_REGION = 5
filter_for_convolve = [1, 1, 1, 1, 1, -10, 1, 1, 1, 1,
                       1]  # paper V. LIDAR ODOMETRY equation (1) => SUM(X_(k,i) - X_(k,j)) # 10개인 이유는 heuristic
assert len(filter_for_convolve) == 2 * FEATURES_REGION + 1
print(len(filter_for_convolve))

import numpy as np

arr = np.arange(10).reshape(2,-1)
print(arr)
print(arr[:,0])
print(arr[0,:])

x = np.pad(filter_for_convolve, 5)
print(x)

folder = '/home/bryan/PycharmProjects/LOAM/KITTI_LIDAR_dataset'
loader = LoaderKITTI(folder, '00')

odometry = OdometryEstimator()
global_transform = np.eye(4)
pcds = []
mapper = Mapper()

vis = o3d.visualization.Visualizer()
vis.create_window()

import numpy as np

a = np.arange(16).reshape(4,4)
# print(a)
b = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
c = np.array(b).reshape(4,4)
# print(c)

print(a @ c)
print(c @ a)
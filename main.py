import copy
import numpy as np
import open3d as o3d

# from loader_vlp16 import LoaderVLP16
import utils
from KITTI_loader import LoaderKITTI
from Lidar_Mapping import Mapper
from odometry_estimator import OdometryEstimator


def find_transformation(source, target, trans_init):
    threshold = 0.2
    if not source.has_normals():
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=50))
    if not target.has_normals():
        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=50))
    transformation = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,
                                                       o3d.pipelines.registration.TransformationEstimationPointToPlane()).transformation
    return transformation


if __name__ == '__main__':

    folder = '/home/bryan/PycharmProjects/LOAM/KITTI_LIDAR_dataset'
    loader = LoaderKITTI(folder, '00')

    odometry = OdometryEstimator()
    global_transform = np.eye(4)
    pcds = []
    mapper = Mapper()

    vis = o3d.visualization.Visualizer()
    vis.create_window()


    # for i in range(loader.length()):
    for i in range(80, 150):
        if i >= 50:
            pcd_np_1 = utils.get_pcd_from_numpy(loader.get_item(i)[0]).voxel_down_sample(voxel_size=0.2)
            pcd_np_2 = utils.get_pcd_from_numpy(loader.get_item(i + 1)[0]).voxel_down_sample(voxel_size=0.2)

            # pcd_np_1 = utils.get_pcd_from_numpy(loader.get_item(i)[0])
            # pcd_np_2 = utils.get_pcd_from_numpy(loader.get_item(i + 1)[0])

            T = find_transformation(pcd_np_2, pcd_np_1, np.eye(4))
            print(T.shape)


            global_transform = global_transform @ T  # 변경 : 원래는 T @ global_transform 으로 되어있었는데, 오류 발생.
            print(global_transform.shape)
            pcds.append(copy.deepcopy(pcd_np_2).transform(global_transform))
            # pcds.append(copy.deepcopy(pcd_np_1))

            vis.add_geometry(pcds[-1])
            vis.poll_events()
            vis.update_renderer()

    vis.destroy_window()


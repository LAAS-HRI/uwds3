#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import rospy
import uuid
from pyuwds3.reasoning.simulation.internal_simulator import InternalSimulator
from pyuwds3.types.grid import Grid
from pyuwds3.types.camera import HumanCamera


class DataGenerator(object):
    """
    """
    def __init__(self):
        """
        """
        cad_models_additional_search_path = rospy.get_param("~cad_models_additional_search_path", "")
        use_simulation_gui = rospy.get_param("~use_simulation_gui", True)
        static_entities_config_filename = rospy.get_param("~static_entities_config_filename", "")
        ouput_data_directory = rospy.get_param("~ouput_data_directory", "/tmp/")
        global_frame_id = rospy.get_param("~global_frame_id", "map")
        base_frame_id = rospy.get_param("~global_frame_id", "base_link")

        max_samples = rospy.get_param("~max_samples", 5)

        simulator = InternalSimulator(use_simulation_gui,
                                      cad_models_additional_search_path,
                                      static_entities_config_filename,
                                      "",
                                      global_frame_id,
                                      base_frame_id,
                                      load_robot=False)

        xmin = rospy.get_param("~xmin", 0.0)
        xmax = rospy.get_param("~xmax", 20.0)

        ymin = rospy.get_param("~ymin", 0.0)
        ymax = rospy.get_param("~ymax", 15.0)

        zmin = rospy.get_param("~zmin", 1.4)
        zmax = rospy.get_param("~zmax", 1.80)

        xdim = rospy.get_param("~xdim", 100)
        ydim = rospy.get_param("~ydim", 100)
        zdim = rospy.get_param("~zdim", 5)

        width = rospy.get_param("~width", 128)
        height = rospy.get_param("~height", 128)

        camera = HumanCamera()
        camera.width = width
        camera.height = height

        grid = Grid(xmin, ymin, zmin, xmax, ymax, zmax, xdim, ydim, zdim)

        nb_samples = 0
        while nb_samples < max_samples:
            if rospy.is_shutdown() is True or nb_samples > max_samples:
                break
            x, y, z = grid.random_cell_indices()
            pose = grid.cell_random_pose(x, y, z)
            xmin_camera = pose.pos.x-0.5
            xmax_camera = pose.pos.x+0.5
            ymin_camera = pose.pos.y-0.5
            ymax_camera = pose.pos.y+0.5
            if simulator.test_aabb_collision(xmin_camera, ymin_camera, 0.1, xmax_camera, ymax_camera, zmax):
                continue
            rgb_image, depth_image, mask_image, tracks = simulator.get_camera_view(pose, camera)
            if len(tracks) != 0:
                nb_samples += 1
                sample_uuid = str(uuid.uuid4()).replace("-", "")
                print((nb_samples, sample_uuid))
                #np.save(sample_uuid+".jpg", depth_image)
                #scipy.misc.imsave("{}{}_depth.jpg".format(ouput_data_directory, sample_uuid), depth_image)
                #scipy.misc.imsave("{}{}_rgb.jpg".format(ouput_data_directory, sample_uuid), rgb_image)

if __name__ == "__main__":
    rospy.init_node("data_generator", anonymous=False)
    generator = DataGenerator()

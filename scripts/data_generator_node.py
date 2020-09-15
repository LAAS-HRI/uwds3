#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import cv2
import os
import rospy
import uuid
from pyuwds3.reasoning.simulation.internal_simulator import InternalSimulator
from pyuwds3.reasoning.sampling.grid_sampler import GridSampler
from pyuwds3.types.camera import HumanCamera

np.random.seed(123)


class DataGenerator(object):
    """
    """
    def __init__(self):
        """
        """
        cad_models_additional_search_path = rospy.get_param("~cad_models_additional_search_path", "")
        use_simulation_gui = rospy.get_param("~use_simulation_gui", True)
        static_entities_config_filename = rospy.get_param("~static_entities_config_filename", "")

        global_frame_id = rospy.get_param("~global_frame_id", "map")
        base_frame_id = rospy.get_param("~global_frame_id", "base_link")
        simulation_config_filename = rospy.get_param("~simulation_config_filename", "")

        self.output_data_directory = rospy.get_param("~ouput_data_directory", "/tmp/")

        self.max_samples = rospy.get_param("~max_samples", 600)

        simulator = InternalSimulator(use_simulation_gui,
                                      simulation_config_filename,
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

        self.data_path = self.output_data_directory

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        camera = HumanCamera()
        camera.width = int(width/2.0)
        camera.height = int(height/2.0)

        grid = GridSampler(xmin, ymin, zmin, xmax, ymax, zmax, xdim, ydim, zdim)

        self.nb_sample = 0
        while not rospy.is_shutdown() is True and self.nb_sample < self.max_samples:
            x, y, z = grid.random_cell_indices()
            pose = grid.cell_random_pose(x, y, z)
            xmin_camera = pose.pos.x-0.5
            xmax_camera = pose.pos.x+0.5
            ymin_camera = pose.pos.y-0.5
            ymax_camera = pose.pos.y+0.5
            if simulator.test_overlap(xmin_camera, ymin_camera, 0.1, xmax_camera, ymax_camera, zmax):
                continue
            rgb_image, depth_image, mask_image, tracks = simulator.get_camera_view(pose, camera)
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            if len(tracks) != 0:
                self.nb_sample += 1
                sample_uuid = str(uuid.uuid4()).replace("-", "")
                rospy.loginfo("[data_generator] sample: {} id: {}".format(self.nb_sample, sample_uuid))
                cv2.imwrite(self.data_path+"/"+sample_uuid+"-rgb.png", bgr_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
                cv2.imwrite(self.data_path+"/"+sample_uuid+"-depth.png", depth_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        rospy.loginfo("[data_generator] Finished !")


if __name__ == "__main__":
    rospy.init_node("data_generator", anonymous=False)
    generator = DataGenerator()

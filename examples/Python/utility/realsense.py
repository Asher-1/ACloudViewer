#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
================================================================
Copyright (C) 2019 * Ltd. All rights reserved.
PROJECT      :  tutorial
FILE_NAME    :  realsense
AUTHOR       :  DAHAI LU
TIME         :  2020/12/31 上午10:31
PRODUCT_NAME :  PyCharm
================================================================
"""

import json
import cloudViewer as cv3d


def bag_test(bag_filename):
    bag_reader = cv3d.t.io.RSBagReader()
    bag_reader.open(bag_filename)
    while not bag_reader.is_eof():
        im_rgbd = bag_reader.next_frame()
        # process im_rgbd.depth and im_rgbd.color
        cv3d.visualization.draw_geometries([im_rgbd])

    bag_reader.close()


def read_from_real_sense(config_filename, bag_filename):
    with open(config_filename) as cf:
        rs_cfg = cv3d.t.io.RealSenseSensorConfig(json.load(cf))

    rs = cv3d.t.io.RealSenseSensor()
    rs.init_sensor(rs_cfg, 0, bag_filename)
    rs.start_capture(True)  # true: start recording with capture
    for fid in range(150):
        im_rgbd = rs.capture_frame(True, True)  # wait for frames and align them
        # process im_rgbd.depth and im_rgbd.color

    rs.stop_capture()


if __name__ == '__main__':
    BAG_FILENAME = ""
    CONFIG_FILENAME = ""
    bag_test(BAG_FILENAME)
    read_from_real_sense(CONFIG_FILENAME, BAG_FILENAME)

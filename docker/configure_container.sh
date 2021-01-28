#!/bin/bash

docker run -dit --runtime=nvidia --name=cloudViewer -m 4g -p 2222:22 detection_ai:latest
import cloudViewer as cv3d

import os

os.environ["WEBRTC_IP"] = "127.0.0.1"
os.environ["WEBRTC_PORT"] = "8882"

if __name__ == "__main__":
    cv3d.visualization.webrtc_server.enable_webrtc()
    cube_red = cv3d.geometry.ccMesh.create_box(1, 2, 4)
    cube_red.compute_vertex_normals()
    cube_red.paint_uniform_color((1.0, 0.0, 0.0))
    cv3d.visualization.draw(cube_red)

import cloudViewer as cv3d

if __name__ == "__main__":
    cv3d.visualization.webrtc_server.enable_webrtc()
    cube_red = cv3d.geometry.TriangleMesh.create_box(1, 2, 4)
    cube_red.compute_vertex_normals()
    cube_red.paint_uniform_color((1.0, 0.0, 0.0))
    cv3d.visualization.draw(cube_red)

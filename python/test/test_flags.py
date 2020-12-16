import cloudViewer as cv3d


def test_global_flags():
    assert cv3d.pybind._GLIBCXX_USE_CXX11_ABI in (True, False)
    assert cv3d.pybind._GLIBCXX_USE_CXX11_ABI == cv3d._build_config[
        'GLIBCXX_USE_CXX11_ABI']
    assert cv3d._build_config['GLIBCXX_USE_CXX11_ABI'] in (True, False)
    assert cv3d._build_config['ENABLE_HEADLESS_RENDERING'] in (True, False)
    assert cv3d._build_config['BUILD_CUDA_MODULE'] in (True, False)

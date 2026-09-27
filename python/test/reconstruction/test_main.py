"""Port of upstream pycolmap main_test.py (Open3D style).

Fork adaptation: version/build info resolves through recon.util.get_version /
get_build_info (the fork has no module-level __version__ attributes)."""


def test_version_is_str(recon):
    version = recon.util.get_version()
    assert isinstance(version, str)
    assert len(version) > 0


def test_build_info_is_str(recon):
    build_info = recon.util.get_build_info()
    assert isinstance(build_info, str)
    assert len(build_info) > 0


def test_device_enum(pipeline):
    assert pipeline.Device.AUTO is not None
    assert pipeline.Device.CPU is not None
    assert pipeline.Device.CUDA is not None

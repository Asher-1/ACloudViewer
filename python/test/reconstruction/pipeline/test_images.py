"""Port of upstream pycolmap pipeline/images_test.py (Open3D style).

Fork adaptations: FileCopyType lives on recon.util with the COPY/HARD_LINK/
SOFT_LINK value names, and the import/undistort engine entry points are
recon.image.register_image / undistort_image.
"""
import pytest


def test_image_reader_options_init(pipeline):
    assert pipeline.ImageReaderOptions() is not None


def test_image_reader_options_camera_model(pipeline):
    options = pipeline.ImageReaderOptions()
    options.camera_model = "SIMPLE_PINHOLE"
    assert options.camera_model == "SIMPLE_PINHOLE"


def test_image_reader_options_check(pipeline):
    assert pipeline.ImageReaderOptions().check()


def test_file_copy_type_enum(recon):
    assert recon.util.COPY is not None
    assert recon.util.HARD_LINK is not None
    assert recon.util.SOFT_LINK is not None


@pytest.mark.parametrize(
    "module_and_name",
    [
        ("image", "register_image"),
        ("image", "undistort_image"),
    ],
)
def test_public_api_callable(recon, module_and_name):
    module, name = module_and_name
    assert callable(getattr(getattr(recon, module), name))

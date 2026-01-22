# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------


def torch_available():
    try:
        import torch
        import torch.utils.dlpack
    except ImportError:
        return False
    return True


def list_devices(enable_cuda=True, enable_sycl=False):
    """
    Returns a list of devices that are available for CloudViewer to use:
    - Device("CPU:0")
    - Device("CUDA:0") if built with CUDA support and a CUDA device is available.
    - Device("SYCL:0") if built with SYCL support and a SYCL GPU device is available.
    """
    import cloudViewer as cv3d

    devices = [cv3d.core.Device("CPU:0")]
    if enable_cuda and cv3d.core.cuda.device_count() > 0:
        devices.append(cv3d.core.Device("CUDA:0"))
    # Ignore fallback SYCL CPU device
    if enable_sycl and hasattr(cv3d.core, 'sycl') and len(
            cv3d.core.sycl.get_available_devices()) > 1:
        devices.append(cv3d.core.Device("SYCL:0"))
    return devices


def list_devices_with_torch():
    """
    Similar to list_devices(), but take PyTorch available devices into account.
    The returned devices are compatible on both PyTorch and CloudViewer.

    If PyTorch is not available at all, empty list will be returned, thus the
    test is effectively skipped.
    """
    if torch_available():
        import cloudViewer as cv3d
        import torch
        devices = [cv3d.core.Device("CPU:0")]
        if (cv3d.core.cuda.device_count() > 0 and
                torch.cuda.device_count() > 0):
            devices += [cv3d.core.Device("CUDA:0")]
        return devices
    else:
        return []

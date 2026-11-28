#!/usr/bin/env python3
"""Export the qOpenPCDet multi-frame test-data bundle (v2).

Every frame lives under frames/<id>/ and keeps its unique original
basenames so the shared test-data repository's recursive basename lookup
stays unambiguous:

  openpcdet_test_data_v2/
    frames/
      kitti_000008/
        kitti_000008.bin                raw KITTI Velodyne frame (4 feats)
        kitti_000008_sampled16384.bin   PointRCNN pre-sampled variant
        scene/                          CaDDN / focal multimodal scene
      nuscenes_n015/
        nuscenes_n015_lidar.bin         OpenPCDet 5-feature single sweep
        scene/                          BEVFusion 6-camera scene

The scene payloads are byte-identical to the pinned v1 members: the
verify subcommand regenerates the 000008 / n015 frames from the upstream
OpenPCDet-GGML validation assets and compares every inference member.
scene.json gains an optional camera_names array (the engine scene loader
only reads the four dimension keys, so unknown keys are ignored).

The PointRCNN variant of kitti_000008 is a frozen legacy member: its
original generator is not reproducible from the published upstream
recipe, so it is carried over verbatim and authenticated by its pinned
SHA-256. New KITTI frames use the documented upstream seed-0 near/far
sampling (prepare_pointrcnn_sample.py).

Subcommands:
  kitti       generate KITTI frames from a raw drive directory + calib
  kitti-asset generate the pinned 000008 frame from upstream assets
  nuscenes    generate one nuScenes frame from a staged asset directory
  stage-mini  stage nuScenes mini-split samples into asset directories
  verify      regenerate pinned frames and byte-compare against v1
  package     zip a staging tree into the publishable v2 archive

Only numpy + Pillow are required; stage-mini additionally needs
nuscenes-devkit. No torch / pcdet dependency.
"""

import argparse
import hashlib
import json
import pickle
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

# Upstream infer_bevfusion_reference.py camera order (the BEVFusion
# checkpoint's camera ring layout).
CAMERAS = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK",
           "CAM_BACK_LEFT", "CAM_BACK_RIGHT")

# Frozen legacy PointRCNN sample of KITTI 000008 (pinned in the v1
# bundle and the openpcdet_input_pointrcnn.zip manifest entry).
FROZEN_SAMPLED_SHA256 = (
    "bab0a849a45b7509085aa68c37458c9c24e7bab1269e0d0bb3619c710088b667")

BUNDLE_GROUP = "openpcdet_test_data_v2"

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], np.float32)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_f32(path, array):
    np.ascontiguousarray(array, dtype=np.float32).tofile(path)


def write_scene(scene_dir, images_nhwc, lidar_to_image, camera_to_lidar,
                camera_intrinsics, image_aug, lidar_aug, camera_names):
    """Write the engine scene payload (aicore_openpcdet_load_scene_dir).

    images_nhwc: [N, H, W, C] float32; the matrix arguments are [N, 4, 4]
    (lidar_aug is a single [4, 4]). camera_names is the optional display
    metadata consumed by the plugin preview grid.
    """
    scene_dir = Path(scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)
    num_cameras, height, width, channels = images_nhwc.shape
    write_f32(scene_dir / "images.bin", images_nhwc)
    write_f32(scene_dir / "lidar_to_image.bin", lidar_to_image)
    write_f32(scene_dir / "camera_to_lidar.bin", camera_to_lidar)
    write_f32(scene_dir / "camera_intrinsics.bin", camera_intrinsics)
    write_f32(scene_dir / "image_aug.bin", image_aug)
    write_f32(scene_dir / "lidar_aug.bin", lidar_aug)
    (scene_dir / "scene.json").write_text(
        json.dumps({
            "num_cameras": int(num_cameras),
            "image_height": int(height),
            "image_width": int(width),
            "image_channels": int(channels),
            "camera_names": list(camera_names),
        }) + "\n")


def sample_pointrcnn(points, count=16384, seed=0):
    """Deterministic PointRCNN input sampling (upstream
    prepare_pointrcnn_sample.py): keep every far point (depth >= 40 m),
    fill the remainder from the near points, then shuffle with seed 0."""
    if len(points) < count:
        raise ValueError("input frame has fewer than %d points" % count)
    random = np.random.RandomState(seed)
    depth = np.linalg.norm(points[:, :3], axis=1)
    far = np.flatnonzero(depth >= 40.0)
    near = np.flatnonzero(depth < 40.0)
    if count > len(far):
        near_choice = random.choice(near, count - len(far), replace=False)
        choice = (np.concatenate(
            (near_choice, far)) if len(far) else near_choice)
    else:
        choice = random.choice(np.arange(len(points), dtype=np.int32),
                               count,
                               replace=False)
    random.shuffle(choice)
    return np.ascontiguousarray(points[choice], dtype=np.float32)


def kitti_scene_from_matrices(image_png, lidar2img, cam2img):
    """Assemble the single-camera KITTI scene (CaDDN convention): the
    image stored 0..1, identity augmentations, identity camera_to_lidar
    (the KITTI checkpoints never consume it — only the BEVFusion LSS
    splat reads camera_to_lidar)."""
    image = np.asarray(Image.open(image_png).convert("RGB"), np.float32) / 255.0
    images = image[None, ...]  # [1, H, W, 3]
    lidar_to_image = np.asarray(lidar2img, np.float32).reshape(1, 4, 4)
    intrinsics = np.asarray(cam2img, np.float32).reshape(1, 4, 4)
    eye = np.eye(4, dtype=np.float32)
    return (images, lidar_to_image, eye[None, ...].copy(), intrinsics,
            eye[None, ...].copy(), eye.copy(), ["CAM2"])


def parse_kitti_calib(calib_path):
    """Read P2 / R0_rect / Tr_velo_to_cam from a raw KITTI calib file and
    return (lidar2img, cam2img) as 4x4 float32 matrices."""
    rows = {}
    for line in Path(calib_path).read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        rows[key.strip()] = np.asarray([float(v) for v in value.split()],
                                       np.float64)
    p2 = np.eye(4, dtype=np.float64)
    p2[:3, :4] = rows["P2"].reshape(3, 4)
    r0 = np.eye(4, dtype=np.float64)
    r0[:3, :3] = rows["R0_rect"].reshape(3, 3)
    tr = np.eye(4, dtype=np.float64)
    tr[:3, :4] = rows["Tr_velo_to_cam"].reshape(3, 4)
    lidar2img = (p2 @ r0 @ tr).astype(np.float32)
    return lidar2img, p2.astype(np.float32)


def cmd_kitti(args):
    drive = Path(args.drive_dir)
    if args.calib_pkl:
        # KITTI calibration is constant per drive: reuse the pinned frame's
        # cam2 matrices verbatim so every generated frame shares the exact
        # authenticated calibration (no float64 re-multiplication drift).
        meta = pickle.loads(Path(args.calib_pkl).read_bytes())
        cam2 = meta["data_list"][0]["images"]["CAM2"]
        lidar2img = np.asarray(cam2["lidar2img"], np.float32).reshape(4, 4)
        cam2img = np.asarray(cam2["cam2img"], np.float32).reshape(4, 4)
    else:
        calib = Path(args.calib) if args.calib else drive / "calib.txt"
        lidar2img, cam2img = parse_kitti_calib(calib)
    out_root = Path(args.out)
    for token in args.frames.split(","):
        idx = int(token)
        frame_id = "kitti_%06d" % idx
        frame_dir = out_root / "frames" / frame_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        stem = "%010d" % idx
        raw = drive / "velodyne_points" / "data" / (stem + ".bin")
        points = np.fromfile(raw, np.float32).reshape(-1, 4)
        shutil.copyfile(raw, frame_dir / (frame_id + ".bin"))
        sampled_path = frame_dir / (frame_id + "_sampled16384.bin")
        if args.frozen_sampled and idx == 8:
            frozen = Path(args.frozen_sampled)
            if sha256(frozen) != FROZEN_SAMPLED_SHA256:
                raise ValueError("frozen sampled member failed SHA-256")
            shutil.copyfile(frozen, sampled_path)
        else:
            write_f32(sampled_path, sample_pointrcnn(points))
        png = drive / "image_02" / "data" / (stem + ".png")
        write_scene(frame_dir / "scene",
                    *kitti_scene_from_matrices(png, lidar2img, cam2img))
        print("frame %s: %d points" % (frame_id, len(points)))


def cmd_kitti_asset(args):
    """Regenerate the pinned 000008 frame from the upstream validation
    asset directory (000008.bin / 000008.png / 000008.pkl)."""
    asset = Path(args.asset_dir)
    meta = pickle.loads((asset / "000008.pkl").read_bytes())
    cam2 = meta["data_list"][0]["images"]["CAM2"]
    out_root = Path(args.out)
    frame_dir = out_root / "frames" / "kitti_000008"
    frame_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(asset / "000008.bin", frame_dir / "kitti_000008.bin")
    frozen = Path(args.frozen_sampled)
    if sha256(frozen) != FROZEN_SAMPLED_SHA256:
        raise ValueError("frozen sampled member failed SHA-256")
    shutil.copyfile(frozen, frame_dir / "kitti_000008_sampled16384.bin")
    write_scene(
        frame_dir / "scene",
        *kitti_scene_from_matrices(asset / "000008.png", cam2["lidar2img"],
                                   cam2["cam2img"]))
    print("frame kitti_000008 regenerated from upstream assets")


def prepare_cameras(asset_dir, metadata, final_size=(256, 704), resize=0.48):
    """Port of the upstream infer_bevfusion_reference.py prepare_cameras
    (resize / crop / ImageNet normalize + calibration matrices)."""
    images, intrinsics, camera2lidar, lidar2image = [], [], [], []
    image_info = metadata["data_list"][0]["images"]
    final_h, final_w = final_size
    crop_x = crop_y = 0
    for name in CAMERAS:
        info = image_info[name]
        image = Image.open(Path(asset_dir) / info["img_path"]).convert("RGB")
        new_w = int(image.width * resize)
        new_h = int(image.height * resize)
        crop_x = int(max(0, new_w - final_w) / 2)
        crop_y = new_h - final_h
        image = image.resize((new_w, new_h)).crop(
            (crop_x, crop_y, crop_x + final_w, crop_y + final_h))
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - IMAGENET_MEAN) / IMAGENET_STD
        images.append(np.moveaxis(array, -1, 0))

        intrinsic = np.eye(4, dtype=np.float32)
        intrinsic[:3, :3] = np.asarray(info["cam2img"], np.float32)
        lidar_to_camera = np.asarray(info["lidar2cam"], np.float32)
        intrinsics.append(intrinsic)
        camera2lidar.append(np.linalg.inv(lidar_to_camera).astype(np.float32))
        lidar2image.append((intrinsic @ lidar_to_camera).astype(np.float32))

    image_aug = np.tile(np.eye(4, dtype=np.float32), (len(CAMERAS), 1, 1))
    image_aug[:, 0, 0] = resize
    image_aug[:, 1, 1] = resize
    image_aug[:, 0, 3] = -crop_x
    image_aug[:, 1, 3] = -crop_y
    return (np.stack(images), np.stack(intrinsics), np.stack(camera2lidar),
            np.stack(lidar2image), image_aug)


def cmd_nuscenes(args):
    asset = Path(args.asset_dir)
    metadata = pickle.loads((asset / "metadata.pkl").read_bytes())
    camera_images, intrinsics, camera2lidar, lidar2image, image_aug = \
        prepare_cameras(asset, metadata)
    nhwc = np.moveaxis(camera_images, 1, -1)  # [6, 256, 704, 3]
    frame_dir = Path(args.out) / "frames" / args.frame_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    lidar_src = asset / "lidar.pcd.bin"
    lidar_dst = frame_dir / (args.frame_id + "_lidar.bin")
    if args.lidar_raw:
        # Raw nuScenes sweep [x,y,z,intensity,ring] -> the OpenPCDet
        # single-sweep representation [x,y,z,intensity/255,time_lag=0]
        # (upstream prepare_nuscenes_sample.py).
        raw = np.fromfile(lidar_src, np.float32).reshape(-1, 5)
        converted = np.column_stack((raw[:, :3], raw[:, 3] / 255.0,
                                     np.zeros(len(raw),
                                              np.float32))).astype(np.float32)
        write_f32(lidar_dst, converted)
    else:
        shutil.copyfile(lidar_src, lidar_dst)
    write_scene(frame_dir / "scene", nhwc,
                lidar2image, camera2lidar, intrinsics, image_aug,
                np.eye(4, dtype=np.float32), list(CAMERAS))
    print("frame %s: %d cameras" % (args.frame_id, len(CAMERAS)))


def _rigid(rotation, translation):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def cmd_stage_mini(args):
    """Stage nuScenes mini-split samples into the asset-directory layout
    cmd_nuscenes consumes (metadata.pkl + camera JPEGs + raw lidar)."""
    from nuscenes.nuscenes import NuScenes
    from pyquaternion import Quaternion

    nusc = NuScenes(version=args.version,
                    dataroot=args.mini_root,
                    verbose=False)
    out_root = Path(args.out)
    for token in args.samples.split(","):
        sample = nusc.get("sample", token)
        frame_id = "nuscenes_" + token[:8]
        asset = out_root / frame_id
        asset.mkdir(parents=True, exist_ok=True)
        lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        lidar_path = Path(nusc.dataroot) / lidar_sd["filename"]
        shutil.copyfile(lidar_path, asset / "lidar.pcd.bin")
        lidar_cs = nusc.get("calibrated_sensor",
                            lidar_sd["calibrated_sensor_token"])
        lidar_ep = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        lidar_to_global = (_rigid(
            Quaternion(lidar_ep["rotation"]).rotation_matrix,
            lidar_ep["translation"]) @ _rigid(
                Quaternion(lidar_cs["rotation"]).rotation_matrix,
                lidar_cs["translation"]))
        images = {}
        for name in CAMERAS:
            cam_sd = nusc.get("sample_data", sample["data"][name])
            cam_path = Path(nusc.dataroot) / cam_sd["filename"]
            jpg_name = "%s__%s__%d.jpg" % (frame_id, name, cam_sd["timestamp"])
            shutil.copyfile(cam_path, asset / jpg_name)
            cam_cs = nusc.get("calibrated_sensor",
                              cam_sd["calibrated_sensor_token"])
            cam_ep = nusc.get("ego_pose", cam_sd["ego_pose_token"])
            global_to_cam = np.linalg.inv(
                _rigid(
                    Quaternion(cam_ep["rotation"]).rotation_matrix,
                    cam_ep["translation"]) @ _rigid(
                        Quaternion(cam_cs["rotation"]).rotation_matrix,
                        cam_cs["translation"]))
            images[name] = {
                "img_path":
                    jpg_name,
                "cam2img":
                    np.asarray(cam_cs["camera_intrinsic"], np.float64),
                "lidar2cam":
                    (global_to_cam @ lidar_to_global).astype(np.float64),
                "sample_data_token":
                    cam_sd["token"],
                "timestamp":
                    cam_sd["timestamp"] / 1e6,
            }
        metadata = {
            "metainfo": {
                "dataset": "nuscenes-mini"
            },
            "data_list": [{
                "token": token,
                "timestamp": sample["timestamp"] / 1e6,
                "images": images,
                "lidar_points": {
                    "num_pts_feats": 5,
                    "lidar_path": "lidar.pcd.bin",
                },
            }],
        }
        (asset / "metadata.pkl").write_bytes(pickle.dumps(metadata))
        print("staged %s -> %s" % (token, asset))


def _compare_member(label, generated, reference):
    gen_bytes = Path(generated).read_bytes()
    ref_bytes = Path(reference).read_bytes()
    if gen_bytes == ref_bytes:
        print("PASS %s (%d B)" % (label, len(gen_bytes)))
        return True
    print("FAIL %s: generated %d B vs reference %d B" %
          (label, len(gen_bytes), len(ref_bytes)))
    return False


def cmd_verify(args):
    """Regenerate the pinned 000008 / n015 frames from the upstream
    validation assets and byte-compare every inference member against
    the reference (v1) bundle. scene.json is compared on its four
    dimension keys (camera_names is a v2 display-only addition). The
    frozen PointRCNN sample is authenticated by its pinned SHA-256."""
    assets = Path(args.assets_root)
    reference = Path(args.reference)
    staging = Path(args.staging)
    ok = True

    # KITTI 000008 from the upstream pkl/png/bin assets.
    meta = pickle.loads((assets / "kitti_000008" / "000008.pkl").read_bytes())
    cam2 = meta["data_list"][0]["images"]["CAM2"]
    gen_kitti = staging / "verify" / "kitti_000008"
    gen_kitti.mkdir(parents=True, exist_ok=True)
    write_scene(
        gen_kitti / "scene",
        *kitti_scene_from_matrices(assets / "kitti_000008" / "000008.png",
                                   cam2["lidar2img"], cam2["cam2img"]))
    ok &= _compare_member("kitti_000008.bin",
                          assets / "kitti_000008" / "000008.bin",
                          reference / "kitti_000008.bin")
    for member in ("images.bin", "lidar_to_image.bin", "camera_to_lidar.bin",
                   "camera_intrinsics.bin", "image_aug.bin", "lidar_aug.bin"):
        ok &= _compare_member("scene_caddn/" + member,
                              gen_kitti / "scene" / member,
                              reference / "scene_caddn" / member)
    gen_json = json.loads((gen_kitti / "scene" / "scene.json").read_text())
    ref_json = json.loads(
        (reference / "scene_caddn" / "scene.json").read_text())
    for key in ("num_cameras", "image_height", "image_width", "image_channels"):
        if gen_json[key] != ref_json[key]:
            print("FAIL scene_caddn/scene.json key %s" % key)
            ok = False
    frozen = reference / "kitti_000008_sampled16384.bin"
    if sha256(frozen) == FROZEN_SAMPLED_SHA256:
        print("PASS kitti_000008_sampled16384.bin (frozen SHA-256)")
    else:
        print("FAIL kitti_000008_sampled16384.bin frozen SHA-256")
        ok = False

    # nuScenes n015 from the upstream metadata/jpg/lidar assets.
    asset = assets / "nuscenes_n015"
    metadata = pickle.loads((asset / "metadata.pkl").read_bytes())
    camera_images, intrinsics, camera2lidar, lidar2image, image_aug = \
        prepare_cameras(asset, metadata)
    gen_nusc = staging / "verify" / "nuscenes_n015"
    gen_nusc.mkdir(parents=True, exist_ok=True)
    write_scene(gen_nusc / "scene", np.moveaxis(camera_images, 1, -1),
                lidar2image, camera2lidar, intrinsics, image_aug,
                np.eye(4, dtype=np.float32), list(CAMERAS))
    ok &= _compare_member("nuscenes_n015_lidar.bin", asset / "lidar.pcd.bin",
                          reference / "nuscenes_n015_lidar.bin")
    for member in ("images.bin", "lidar_to_image.bin", "camera_to_lidar.bin",
                   "camera_intrinsics.bin", "image_aug.bin", "lidar_aug.bin"):
        ok &= _compare_member("scene_bevfusion/" + member,
                              gen_nusc / "scene" / member,
                              reference / "scene_bevfusion" / member)
    gen_json = json.loads((gen_nusc / "scene" / "scene.json").read_text())
    ref_json = json.loads(
        (reference / "scene_bevfusion" / "scene.json").read_text())
    for key in ("num_cameras", "image_height", "image_width", "image_channels"):
        if gen_json[key] != ref_json[key]:
            print("FAIL scene_bevfusion/scene.json key %s" % key)
            ok = False

    print("VERIFY %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def cmd_package(args):
    staging = Path(args.staging)
    frames = staging / "frames"
    if not frames.is_dir() or not any(frames.iterdir()):
        raise ValueError("staging tree carries no frames: %s" % staging)
    zip_path = Path(args.zip)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    members = []
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        # Only the frames/ tree is publishable; verify/ scratch output and
        # earlier archives in the staging root must never ride along.
        for path in sorted(frames.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(staging).as_posix()
            info = zipfile.ZipInfo(BUNDLE_GROUP + "/" + rel,
                                   date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
            members.append({
                "relative_path": BUNDLE_GROUP + "/" + rel,
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            })
    entry = {
        "id": "openpcdet-plugin-test-data",
        "tasks": ["openpcdet"],
        "relative_path": zip_path.name,
        "url": ("https://github.com/Asher-1/cloudViewer_downloads/releases/"
                "download/OpenPCDet-GGUF/" + zip_path.name),
        "sha256": sha256(zip_path),
        "size_bytes": zip_path.stat().st_size,
        "archive": True,
        "extracted_files": members,
    }
    print(json.dumps(entry, indent=1))


def main():
    parser = argparse.ArgumentParser(
        description="Export the qOpenPCDet multi-frame test-data bundle.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("kitti", help="generate KITTI frames from raw data")
    p.add_argument("--drive-dir",
                   required=True,
                   help="extracted 2011_09_26_drive_XXXX_sync directory")
    p.add_argument("--calib",
                   default=None,
                   help="KITTI calib file (default: <drive-dir>/calib.txt)")
    p.add_argument("--calib-pkl",
                   default=None,
                   help="pinned frame pkl whose cam2 matrices are reused "
                   "verbatim (drive-constant calibration)")
    p.add_argument("--frames",
                   required=True,
                   help="comma-separated frame indices, e.g. 0,2,5,8,10")
    p.add_argument("--frozen-sampled",
                   default=None,
                   help="frozen PointRCNN sample carried over for 000008")
    p.add_argument("--out", required=True, help="staging root")
    p.set_defaults(func=cmd_kitti)

    p = sub.add_parser("kitti-asset",
                       help="regenerate pinned 000008 from upstream assets")
    p.add_argument("--asset-dir", required=True)
    p.add_argument("--frozen-sampled", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_kitti_asset)

    p = sub.add_parser("nuscenes",
                       help="generate one nuScenes frame from staged assets")
    p.add_argument("--asset-dir", required=True)
    p.add_argument("--frame-id", required=True)
    p.add_argument("--lidar-raw",
                   action="store_true",
                   help="lidar.pcd.bin is a raw sweep (convert to OpenPCDet)")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_nuscenes)

    p = sub.add_parser("stage-mini",
                       help="stage nuScenes mini samples (needs devkit)")
    p.add_argument("--mini-root", required=True)
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--samples",
                   required=True,
                   help="comma-separated sample tokens")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_stage_mini)

    p = sub.add_parser("verify",
                       help="byte-compare regenerated pinned frames vs v1")
    p.add_argument("--assets-root",
                   required=True,
                   help="upstream validation/assets directory")
    p.add_argument("--reference",
                   required=True,
                   help="extracted v1 bundle root (scene_caddn/...)")
    p.add_argument("--staging", required=True, help="scratch output root")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("package", help="zip the staging tree for publishing")
    p.add_argument("--staging", required=True)
    p.add_argument("--zip", required=True)
    p.set_defaults(func=cmd_package)

    args = parser.parse_args()
    result = args.func(args)
    sys.exit(result if isinstance(result, int) else 0)


if __name__ == "__main__":
    main()

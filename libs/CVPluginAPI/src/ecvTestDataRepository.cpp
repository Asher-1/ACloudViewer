// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvTestDataRepository.h"

#include <QCryptographicHash>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QImageReader>
#include <algorithm>
#include <functional>

#include "ecvModelDownloader.h"

extern "C" {
#include "ioapi.h"
#include "unzip.h"
}

#include <memory>

// ----------------------------------------------------------------------------
// Constants
// ----------------------------------------------------------------------------

namespace {

// Monstree dataset. Content anchor verified once at ingestion (streamed
// digest) and recorded in the <zip>.cvintegrity ledger; later access
// checks are stat-only. SHA-256 pins were computed from MD5-verified
// release artifacts.
constexpr const char* kMonstreeZipName = "dataset_monstree.zip";
constexpr const char* kMonstreeExtractDir = "dataset_monstree";
constexpr const char* kMonstreeDownloadUrl =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "reconstruction_data/dataset_monstree.zip";
constexpr const char* kMonstreeSha256 =
        "db890a8f64780ef3c491a088a3be9d34fa4cbb26eeca31394e715b0cd4cce46c";

// FriendsFaces dataset
constexpr const char* kFriendsZipName = "friends_faces.zip";
constexpr const char* kFriendsExtractDir = "friends_faces";
constexpr const char* kFriendsDownloadUrl =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "qFaceDetect/friends_faces.zip";
constexpr const char* kFriendsSha256 =
        "eb2c2daff249f8bf50e9f95ae1a37c3ce7de06bc7e2a798c4512a179265a637b";

// Shared object detection / background removal / line detection samples.
// SHA-256 of the 2026-09-20 release refresh (105537126 bytes): adds the raw
// tracking videos people-walking.mp4 / vehicles.mp4 (un-annotated evaluation
// material; supervision_demo.mp4 keeps its baked-in overlays).
constexpr const char* kObjectsDetectionZipName = "objects_detection_data.zip";
constexpr const char* kObjectsDetectionExtractDir = "objects_detection_data";
constexpr const char* kObjectsDetectionDownloadUrl =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "objects_detection_data/objects_detection_data.zip";
constexpr const char* kObjectsDetectionSha256 =
        "f6dc7755a124d3ffa4454e5d0b52c8f97d149f2807ac877420c4fd319feaddd2";

// Single-image-to-3D samples (qTrellis, qSAM3D): 33 curated images in
// examples_images/ plus multi-view (mv/), texture (example_texturing/),
// HDRI and webp extras, and the official SAM 3D Objects scene set in
// sam3d_images/ (RGBA image.png with the object mask as alpha).
constexpr const char* kImage2MeshZipName = "image_to_mesh_data.zip";
constexpr const char* kImage2MeshExtractDir = "image_to_mesh_data";
constexpr const char* kImage2MeshDownloadUrl =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "Image2MeshData/image_to_mesh_data.zip";
constexpr const char* kImage2MeshSha256 =
        "490f85a45c37f278bff8011109e91f66488d155c438015e3fdd7b95bbc297cac";

// SAM3 segmentation samples (qSAM3): 14 images in images/ + 7 tracking
// videos in videos/.
constexpr const char* kSam3ZipName = "sam_test_data.zip";
constexpr const char* kSam3ExtractDir = "sam_test_data";
constexpr const char* kSam3DownloadUrl =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "sam_test_data/sam_test_data.zip";
constexpr const char* kSam3Sha256 =
        "3c1d97fddc540dfbc134738aa29fa1607bba329c03aa5c0df37f1e6e13317e87";

// Official GKDT demo images (qGKD): the 15 images behind the upstream
// General-Keypoint-Detection README examples, incl. the text-prompt demo
// (2007_007524.jpg) and its 1-shot support image (2007_003778.jpg).
constexpr const char* kGkdZipName = "general_keypoint_detection_data.zip";
constexpr const char* kGkdExtractDir = "general_keypoint_detection_data";
constexpr const char* kGkdDownloadUrl =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "general_keypoint_detection_data/general_keypoint_detection_data.zip";
constexpr const char* kGkdSha256 =
        "14da5f5296fcac60a248cde7623f8d93d9fa6b29b8cc9a94a0258f5da4189e23";

// Official LightGlue demo pair (qLightGlue): the day/night, wide-baseline
// sacre_coeur images shipped by the upstream repo and consumed by the AICore
// parity tests. The release asset carries no published digest yet, so the
// anchor stays empty on purpose (ecvAssetIntegrity "no content pin": the 1 MB
// ingestion gate plus a size-only ledger entry; pin a SHA-256 here once the
// release publishes one).
constexpr const char* kLightGlueZipName = "lightglue_test_images.zip";
constexpr const char* kLightGlueExtractDir = "lightglue_test_images";
constexpr const char* kLightGlueDownloadUrl =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/"
        "download/LightGlue/lightglue_test_images.zip";
constexpr const char* kLightGlueSha256 = "";

// PaddleOCR official demo/test images (qOCR + qAnyLabeling): pytest
// test_files, street scenes, release/2.7 scene shots, the ch/en single-word
// crops the official recognition demos use, the PP-Structure table pages
// (layout/table_page_1.png, layout/table_page_layout.jpg), two real-document
// table crops (tables/: PP-Structure docs table.jpg, PaddleX
// table_recognition_v2.jpg), and the labelled
// table-recognition GT (table_rec/val.txt + 100 images) behind the AICore
// table TEDS gate. test_files/table.jpg and table_rec/ are official but
// synthetic tables. MANIFEST.txt inside carries per-file SHA-256/source.
constexpr const char* kPpocrZipName = "ppocr_testdata.zip";
constexpr const char* kPpocrExtractDir = "ppocr_testdata";
constexpr const char* kPpocrDownloadUrl =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/"
        "download/PP-OCR/ppocr_testdata.zip";
constexpr const char* kPpocrSha256 =
        "9618131e12a52f9f25a793cc22d1a6773f233fea489bf946bc52353ab716b484";

// LingBot-Map streaming reconstruction scenes (qLingbotMap). Four official
// demo sequences (courthouse/loop/oxford/university, pre-cropped to the
// 518-wide processing grid) plus the cached native sky masks for the two
// outdoor scenes that ship them. SHA-256 pins come from the
// lingbot_map_data release (expanded_assets digest markers).
// ALL LingBot-Map test data lands UNDER the "lingbot_map_test_data" group
// directory instead of littering the extract root shared with the model
// catalogs (lingbot_models, da3_models, ...).
constexpr const char* kLingbotMapTestGroupDir = "lingbot_map_test_data";
constexpr const char* kLingbotMapDownloadBase =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "lingbot_map_data/";
constexpr const char* kLingbotMapCourthouseZipName = "courthouse.zip";
constexpr const char* kLingbotMapCourthouseExtractDir = "courthouse";
constexpr const char* kLingbotMapCourthouseSha256 =
        "fd0a19083600a5d588fdbb4740c808b5ca227ad99d48f2760f65e7bb6aa34ed4";
// Official long-model demo videos (single-file mp4 assets: the downloaded
// file is the content, no extraction). SHA-256 pinned from the release
// assets (locally verified against the upstream long_real campaign inputs).
constexpr const char* kLingbotMapDriveVideoName = "drive_frames.mp4";
constexpr const char* kLingbotMapDriveVideoExtractDir = "lingbot_map_long";
constexpr const char* kLingbotMapDriveVideoSha256 =
        "da814b6ca859d189c5e636ddbadfcc7e14ee532eed904d06b5f3e20de79497d9";
constexpr const char* kLingbotMapLingboWorldVideoName =
        "lingbo_world_frames.mp4";
constexpr const char* kLingbotMapLingboWorldVideoExtractDir =
        "lingbot_map_long";
constexpr const char* kLingbotMapLingboWorldVideoSha256 =
        "28bd8cbb7cf6b214865176c74083e89ed40adba50add05a768adae990913478e";

// MapAnything / VGGT official ETH3D benchmark scenes (qMapAnything). All 13
// scene zips come from the map_anything_vggt_test_data release; their
// SHA-256 pins live in aicore/asset_digests.h (resolved once here through
// ecvAssetIntegrity::PinnedDigest, the same single point the model
// downloads use).
// ALL MapAnything test data lands UNDER the "map_anything_test_data" group
// directory, mirroring the LingBot-Map grouped layout.
constexpr const char* kMapAnythingTestGroupDir = "map_anything_test_data";
constexpr const char* kMapAnythingDownloadBase =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "map_anything_vggt_test_data/";
struct MapAnythingScene {
    ecvTestDataRepository::Dataset kind;
    const char* scene;  // zip name without extension (= extract dir name)
    const char* label;  // display name
};
constexpr MapAnythingScene kMapAnythingScenes[] = {
        {ecvTestDataRepository::Dataset::MapAnythingCourtyard, "courtyard",
         "MapAnything courtyard (ETH3D)"},
        {ecvTestDataRepository::Dataset::MapAnythingDeliveryArea,
         "delivery_area", "MapAnything delivery_area (ETH3D)"},
        {ecvTestDataRepository::Dataset::MapAnythingElectro, "electro",
         "MapAnything electro (ETH3D)"},
        {ecvTestDataRepository::Dataset::MapAnythingFacade, "facade",
         "MapAnything facade (ETH3D)"},
        {ecvTestDataRepository::Dataset::MapAnythingKicker, "kicker",
         "MapAnything kicker (ETH3D)"},
        {ecvTestDataRepository::Dataset::MapAnythingMeadow, "meadow",
         "MapAnything meadow (ETH3D)"},
        {ecvTestDataRepository::Dataset::MapAnythingOffice, "office",
         "MapAnything office (ETH3D)"},
        {ecvTestDataRepository::Dataset::MapAnythingPipes, "pipes",
         "MapAnything pipes (ETH3D)"},
        {ecvTestDataRepository::Dataset::MapAnythingPlayground, "playground",
         "MapAnything playground (ETH3D)"},
        {ecvTestDataRepository::Dataset::MapAnythingRelief, "relief",
         "MapAnything relief (ETH3D)"},
        {ecvTestDataRepository::Dataset::MapAnythingRelief2, "relief_2",
         "MapAnything relief_2 (ETH3D)"},
        {ecvTestDataRepository::Dataset::MapAnythingTerrace, "terrace",
         "MapAnything terrace (ETH3D)"},
        {ecvTestDataRepository::Dataset::MapAnythingTerrains, "terrains",
         "MapAnything terrains (ETH3D)"},
};
constexpr const char* kLingbotMapLoopZipName = "loop.zip";
constexpr const char* kLingbotMapLoopExtractDir = "loop";
constexpr const char* kLingbotMapLoopSha256 =
        "67b49e8c4a56108b80700da3da6dc303f694ab4f028337625f57725eb573f3de";
constexpr const char* kLingbotMapOxfordZipName = "oxford.zip";
constexpr const char* kLingbotMapOxfordExtractDir = "oxford";
constexpr const char* kLingbotMapOxfordSha256 =
        "a81a975c80fc7f9e44224730c78dc3d3f3c52f1ca2200c13b64b57a130cbb032";
constexpr const char* kLingbotMapOxfordSkyMasksZipName = "oxford_sky_masks.zip";
constexpr const char* kLingbotMapOxfordSkyMasksExtractDir = "oxford_sky_masks";
constexpr const char* kLingbotMapOxfordSkyMasksSha256 =
        "8f8d7a6981ac8b56e9a2cd9665774a554901d0d0060caeb2fd9f873e21c02c27";
constexpr const char* kLingbotMapUniversityZipName = "university.zip";
constexpr const char* kLingbotMapUniversityExtractDir = "university";
constexpr const char* kLingbotMapUniversitySha256 =
        "bd745b5847ae8d2a5b1385e9e19a6f5de79dda80405864b42f2dee2e8492f653";
constexpr const char* kLingbotMapUniversitySkyMasksZipName =
        "university_sky_masks.zip";
constexpr const char* kLingbotMapUniversitySkyMasksExtractDir =
        "university_sky_masks";
constexpr const char* kLingbotMapUniversitySkyMasksSha256 =
        "fccfd832ea675d7b29708c80ed37504c1f5d1f5e6ab19bf8eb3cbf620498324e";

// OpenPCDet detection sample (qOpenPCDet): one raw KITTI LiDAR frame
// (kitti_000008.bin, 17238 x 4), its PointRCNN pre-sampled 16384 x 4
// variant, one nuScenes 5-feature sweep (n015 scene) and the two
// multi-modal scene bundles (scene_caddn single-camera, scene_bevfusion
// 6-camera). The zip keeps a top-level "openpcdet_test_data" group
// directory, so the dataset lands UNDER the extract group of the same
// name (mirroring the LingBot-Map / MapAnything grouped layout).
constexpr const char* kOpenPCDetZipName = "openpcdet_test_data.zip";
constexpr const char* kOpenPCDetExtractDir = "openpcdet_test_data";
constexpr const char* kOpenPCDetGroupDir = "openpcdet_test_data";
constexpr const char* kOpenPCDetDownloadUrl =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "OpenPCDet-GGUF/openpcdet_test_data.zip";
constexpr const char* kOpenPCDetSha256 =
        "f1f185a385ddd0b440ed5699be47169eb89b7a0107e29fd26e33ed81a07939dd";

}  // namespace

// ----------------------------------------------------------------------------
// Static path helpers
// ----------------------------------------------------------------------------

QString ecvTestDataRepository::dataRoot() {
    const QString configuredRoot =
            qEnvironmentVariable("CLOUDVIEWER_DATA_ROOT");
    if (!configuredRoot.isEmpty()) return configuredRoot;
    return QDir(QDir::homePath()).filePath(QStringLiteral("cloudViewer_data"));
}

QString ecvTestDataRepository::downloadDir() {
    return QDir(dataRoot()).filePath(QStringLiteral("download"));
}

QString ecvTestDataRepository::extractDir() {
    return QDir(dataRoot()).filePath(QStringLiteral("extract"));
}

QString ecvTestDataRepository::zipPath(Dataset kind) {
    const auto info = getDatasetInfo(kind);
    return QDir(downloadDir()).filePath(info.zipFileName);
}

QString ecvTestDataRepository::extractPath(Dataset kind) {
    const auto info = getDatasetInfo(kind);
    if (!info.extractGroupName.isEmpty()) {
        // Grouped layout: extract/<group>/<extractDirName> — keeps a plugin
        // family's test data out of the extract root shared with catalogs.
        return QDir(QDir(extractDir()).filePath(info.extractGroupName))
                .filePath(info.extractDirName);
    }
    return QDir(extractDir()).filePath(info.extractDirName);
}

QString ecvTestDataRepository::findDatasetFile(Dataset kind,
                                               const QString& fileName) {
    if (fileName.isEmpty() || QFileInfo(fileName).fileName() != fileName) {
        return {};
    }

    const QString root = extractPath(kind);
    if (!QDir(root).exists()) return {};

    if (kind == Dataset::RelateAnything) {
        struct Member {
            const char* path;
            const char* sha;
            qint64 size;
        };
        static const Member members[] = {
                {"demo_images/bicycle.jpg",
                 "1173a3bb48f4875064a73869bbeef7431c0c1c14a6ec751a75e8fbc7df4a8"
                 "e18",
                 211755},
                {"demo_images/catlaptop.jpg",
                 "55da50d82845ad990c7a4ea377ef10bc0db0dd6076895ca14eb527775674d"
                 "269",
                 207115},
                {"demo_images/frisbee.jpg",
                 "f52cbb5a26bda7c266e7e95fcda0b87f5b92f65829fd46896af2b30012c65"
                 "d88",
                 234814},
                {"demo_images/horse.jpg",
                 "549bd965a4ac9aa682f70e40757c5841d160c4e84eb86c2948d5bac5dedec"
                 "f29",
                 469827},
                {"demo_images/skateboard.jpg",
                 "6b5cb50eb4f13061528124f573ed9d433559f34bb8dae3e65cd970db78712"
                 "3e5",
                 233771},
                {"demo_images/tennis.jpg",
                 "5b5f10a29ba288bc792a676b3881e8d8f0b4e6bb3227bc6b8699cbeda66c5"
                 "649",
                 444472},
                {"demo_videos/r3c_adoption_54729571599.webm",
                 "6ec4160dfb683cfb880375a118064acc3caa67707244f040f7d0316017c7c"
                 "541",
                 32641680},
                {"demo_videos/"
                 "gof_a_gweithiwr_metrel_ray_burrell_metal_fabrica.webm",
                 "6f17d65d2012cba1a21513a436bfeababa7c043172aa0b615f804365872a9"
                 "807",
                 3736282},
                {"demo_videos/"
                 "technical_hardtail_riding_all_mountain_enduro.webm",
                 "7e5f17cd3b682e6d58a9367ea6d4441ec7b642aeba9e42314ee3feae0d03a"
                 "fb3",
                 17366970},
                {"demo_videos/jalebi_being_prepared_bangalore.webm",
                 "cec4667f1a037558bf8b02554629f829e0350f236e63e28be1118da8aa794"
                 "78c",
                 11665054},
                {"demo_videos/"
                 "david_farmer_guitarist_at_copped_hall_open_day_a.webm",
                 "e1dbd6cff0c7efc1529f3171f6db33646446c87e62c44f39ee7e7f6970eef"
                 "490",
                 37568246},
        };
        for (const auto& member : members) {
            if (QFileInfo(QString::fromUtf8(member.path)).fileName() !=
                fileName)
                continue;
            const QString path =
                    QDir(root).filePath(QString::fromUtf8(member.path));
            return ecvAssetIntegrity::isVerified(
                           path,
                           {QCryptographicHash::Sha256, QByteArray(member.sha)},
                           0, false, ecvAssetIntegrity::OnMiss::DeepVerify,
                           member.size)
                           ? QFileInfo(path).absoluteFilePath()
                           : QString();
        }
        return {};  // Only declared demo members participate in autofill.
    }

    if (kind == Dataset::RAM && fileName == QStringLiteral("demo1.jpg")) {
        const QString demo =
                QDir(root).filePath(QStringLiteral("demo/demo1.jpg"));
        const ecvAssetIntegrity::Anchor anchor{
                QCryptographicHash::Sha256,
                QByteArray("1b2906f4058a69936df49cb6156ec4cd117a286b420e1eb1476"
                           "4033bf8f3c05f")};
        return ecvAssetIntegrity::isVerified(
                       demo, anchor, 0, false,
                       ecvAssetIntegrity::OnMiss::DeepVerify, 5701095)
                       ? QFileInfo(demo).absoluteFilePath()
                       : QString();
    }

    const QString direct = QDir(root).filePath(fileName);
    if (QFileInfo(direct).isFile()) {
        return QFileInfo(direct).absoluteFilePath();
    }

    QString match;
    QDirIterator it(root, QStringList{fileName}, QDir::Files,
                    QDirIterator::Subdirectories);
    while (it.hasNext()) {
        const QString candidate = QFileInfo(it.next()).absoluteFilePath();
        if (!match.isEmpty()) {
            return {};  // Ambiguous archive contents must not pick arbitrarily.
        }
        match = candidate;
    }
    return match;
}

// ----------------------------------------------------------------------------
// Dataset metadata
// ----------------------------------------------------------------------------

ecvTestDataRepository::DatasetInfo ecvTestDataRepository::getDatasetInfo(
        Dataset kind) {
    switch (kind) {
        case Dataset::RelateAnything:
            return {kind,
                    QStringLiteral("RelateAnything"),
                    QStringLiteral("relate_datasets.zip"),
                    QStringLiteral("relate_datasets"),
                    QStringLiteral("https://huggingface.co/Asher-1/"
                                   "relateanything-ggml-full/resolve/main/"
                                   "assets/relate_datasets.zip"),
                    {QCryptographicHash::Sha256,
                     QByteArray("e0cf31c540dfa4ac8b122e7bc73c7d3070a5d385d17a70"
                                "17229344a2b7895568")}};
        case Dataset::RAM:
            return {kind,
                    QStringLiteral("Recognize Anything"),
                    QStringLiteral("ram_datasets.zip"),
                    QStringLiteral("ram_datasets"),
                    QStringLiteral("https://huggingface.co/Asher-1/RAM_GGUF/"
                                   "resolve/main/ram_datasets.zip"),
                    {QCryptographicHash::Sha256,
                     QByteArray("ee3060f41247895a7140162f8e217a51bf2dd46df8e166"
                                "79157b8086f7490177")}};
        case Dataset::Monstree:
            return {kind,
                    QStringLiteral("Monstree"),
                    QString::fromLatin1(kMonstreeZipName),
                    QString::fromLatin1(kMonstreeExtractDir),
                    QString::fromLatin1(kMonstreeDownloadUrl),
                    {QCryptographicHash::Sha256, QByteArray(kMonstreeSha256)}};
        case Dataset::FriendsFaces:
            return {kind,
                    QStringLiteral("FriendsFaces"),
                    QString::fromLatin1(kFriendsZipName),
                    QString::fromLatin1(kFriendsExtractDir),
                    QString::fromLatin1(kFriendsDownloadUrl),
                    {QCryptographicHash::Sha256, QByteArray(kFriendsSha256)}};
        case Dataset::ObjectsDetection:
            return {kind,
                    QStringLiteral("ObjectsDetection"),
                    QString::fromLatin1(kObjectsDetectionZipName),
                    QString::fromLatin1(kObjectsDetectionExtractDir),
                    QString::fromLatin1(kObjectsDetectionDownloadUrl),
                    {QCryptographicHash::Sha256,
                     QByteArray(kObjectsDetectionSha256)}};
        case Dataset::Image2Mesh:
            return {kind,
                    QStringLiteral("Image2Mesh"),
                    QString::fromLatin1(kImage2MeshZipName),
                    QString::fromLatin1(kImage2MeshExtractDir),
                    QString::fromLatin1(kImage2MeshDownloadUrl),
                    {QCryptographicHash::Sha256,
                     QByteArray(kImage2MeshSha256)}};
        case Dataset::SAM3:
            return {kind,
                    QStringLiteral("SAM3"),
                    QString::fromLatin1(kSam3ZipName),
                    QString::fromLatin1(kSam3ExtractDir),
                    QString::fromLatin1(kSam3DownloadUrl),
                    {QCryptographicHash::Sha256, QByteArray(kSam3Sha256)}};
        case Dataset::GeneralKeypointDetection:
            return {kind,
                    QStringLiteral("GeneralKeypointDetection"),
                    QString::fromLatin1(kGkdZipName),
                    QString::fromLatin1(kGkdExtractDir),
                    QString::fromLatin1(kGkdDownloadUrl),
                    {QCryptographicHash::Sha256, QByteArray(kGkdSha256)}};
        case Dataset::LightGlueTestImages:
            return {kind,
                    QStringLiteral("LightGlueTestImages"),
                    QString::fromLatin1(kLightGlueZipName),
                    QString::fromLatin1(kLightGlueExtractDir),
                    QString::fromLatin1(kLightGlueDownloadUrl),
                    {QCryptographicHash::Sha256, QByteArray(kLightGlueSha256)}};
        case Dataset::Ppocr:
            return {kind,
                    QStringLiteral("PaddleOCR demo images"),
                    QString::fromLatin1(kPpocrZipName),
                    QString::fromLatin1(kPpocrExtractDir),
                    QString::fromLatin1(kPpocrDownloadUrl),
                    {QCryptographicHash::Sha256, QByteArray(kPpocrSha256)}};
        case Dataset::LingbotMapCourthouse:
            return {kind,
                    QStringLiteral("LingBot-Map courthouse (outdoor)"),
                    QString::fromLatin1(kLingbotMapCourthouseZipName),
                    QString::fromLatin1(kLingbotMapCourthouseExtractDir),
                    QString::fromLatin1(kLingbotMapDownloadBase) +
                            QString::fromLatin1(kLingbotMapCourthouseZipName),
                    {QCryptographicHash::Sha256,
                     QByteArray(kLingbotMapCourthouseSha256)},
                    QString::fromLatin1(kLingbotMapTestGroupDir)};
        case Dataset::LingbotMapLoop:
            return {kind,
                    QStringLiteral("LingBot-Map loop (indoor loop closure)"),
                    QString::fromLatin1(kLingbotMapLoopZipName),
                    QString::fromLatin1(kLingbotMapLoopExtractDir),
                    QString::fromLatin1(kLingbotMapDownloadBase) +
                            QString::fromLatin1(kLingbotMapLoopZipName),
                    {QCryptographicHash::Sha256,
                     QByteArray(kLingbotMapLoopSha256)},
                    QString::fromLatin1(kLingbotMapTestGroupDir)};
        case Dataset::LingbotMapOxford:
            return {kind,
                    QStringLiteral("LingBot-Map Oxford Spires (outdoor)"),
                    QString::fromLatin1(kLingbotMapOxfordZipName),
                    QString::fromLatin1(kLingbotMapOxfordExtractDir),
                    QString::fromLatin1(kLingbotMapDownloadBase) +
                            QString::fromLatin1(kLingbotMapOxfordZipName),
                    {QCryptographicHash::Sha256,
                     QByteArray(kLingbotMapOxfordSha256)},
                    QString::fromLatin1(kLingbotMapTestGroupDir)};
        case Dataset::LingbotMapOxfordSkyMasks:
            return {kind,
                    QStringLiteral("LingBot-Map oxford sky masks"),
                    QString::fromLatin1(kLingbotMapOxfordSkyMasksZipName),
                    QString::fromLatin1(kLingbotMapOxfordSkyMasksExtractDir),
                    QString::fromLatin1(kLingbotMapDownloadBase) +
                            QString::fromLatin1(
                                    kLingbotMapOxfordSkyMasksZipName),
                    {QCryptographicHash::Sha256,
                     QByteArray(kLingbotMapOxfordSkyMasksSha256)},
                    QString::fromLatin1(kLingbotMapTestGroupDir)};
        case Dataset::LingbotMapUniversity:
            return {kind,
                    QStringLiteral("LingBot-Map university (outdoor)"),
                    QString::fromLatin1(kLingbotMapUniversityZipName),
                    QString::fromLatin1(kLingbotMapUniversityExtractDir),
                    QString::fromLatin1(kLingbotMapDownloadBase) +
                            QString::fromLatin1(kLingbotMapUniversityZipName),
                    {QCryptographicHash::Sha256,
                     QByteArray(kLingbotMapUniversitySha256)},
                    QString::fromLatin1(kLingbotMapTestGroupDir)};
        case Dataset::LingbotMapUniversitySkyMasks:
            return {kind,
                    QStringLiteral("LingBot-Map university sky masks"),
                    QString::fromLatin1(kLingbotMapUniversitySkyMasksZipName),
                    QString::fromLatin1(
                            kLingbotMapUniversitySkyMasksExtractDir),
                    QString::fromLatin1(kLingbotMapDownloadBase) +
                            QString::fromLatin1(
                                    kLingbotMapUniversitySkyMasksZipName),
                    {QCryptographicHash::Sha256,
                     QByteArray(kLingbotMapUniversitySkyMasksSha256)},
                    QString::fromLatin1(kLingbotMapTestGroupDir)};
        case Dataset::LingbotMapDriveVideo:
            // Single-file mp4 asset: zipFileName carries the download file,
            // extractDirName the directory it is materialized into.
            return {kind,
                    QStringLiteral("LingBot-Map drive (long model)"),
                    QString::fromLatin1(kLingbotMapDriveVideoName),
                    QString::fromLatin1(kLingbotMapDriveVideoExtractDir),
                    QString::fromLatin1(kLingbotMapDownloadBase) +
                            QString::fromLatin1(kLingbotMapDriveVideoName),
                    {QCryptographicHash::Sha256,
                     QByteArray(kLingbotMapDriveVideoSha256)},
                    QString::fromLatin1(kLingbotMapTestGroupDir)};
        case Dataset::LingbotMapLingboWorldVideo:
            return {kind,
                    QStringLiteral("LingBot-Map lingbo world (long model)"),
                    QString::fromLatin1(kLingbotMapLingboWorldVideoName),
                    QString::fromLatin1(kLingbotMapLingboWorldVideoExtractDir),
                    QString::fromLatin1(kLingbotMapDownloadBase) +
                            QString::fromLatin1(
                                    kLingbotMapLingboWorldVideoName),
                    {QCryptographicHash::Sha256,
                     QByteArray(kLingbotMapLingboWorldVideoSha256)},
                    QString::fromLatin1(kLingbotMapTestGroupDir)};
        case Dataset::MapAnythingCourtyard:
        case Dataset::MapAnythingDeliveryArea:
        case Dataset::MapAnythingElectro:
        case Dataset::MapAnythingFacade:
        case Dataset::MapAnythingKicker:
        case Dataset::MapAnythingMeadow:
        case Dataset::MapAnythingOffice:
        case Dataset::MapAnythingPipes:
        case Dataset::MapAnythingPlayground:
        case Dataset::MapAnythingRelief:
        case Dataset::MapAnythingRelief2:
        case Dataset::MapAnythingTerrace:
        case Dataset::MapAnythingTerrains: {
            const auto* row = std::find_if(std::begin(kMapAnythingScenes),
                                           std::end(kMapAnythingScenes),
                                           [kind](const MapAnythingScene& s) {
                                               return s.kind == kind;
                                           });
            if (row == std::end(kMapAnythingScenes)) Q_UNREACHABLE();
            const QString zipName = QStringLiteral("%1.zip").arg(row->scene);
            return {kind,
                    QString::fromLatin1(row->label),
                    zipName,
                    QString::fromLatin1(row->scene),
                    QString::fromLatin1(kMapAnythingDownloadBase) + zipName,
                    {QCryptographicHash::Sha256,
                     ecvAssetIntegrity::PinnedDigest(zipName)},
                    QString::fromLatin1(kMapAnythingTestGroupDir)};
        }
        case Dataset::OpenPCDetSample:
            return {kind,
                    QStringLiteral("OpenPCDet detection sample"),
                    QString::fromLatin1(kOpenPCDetZipName),
                    QString::fromLatin1(kOpenPCDetExtractDir),
                    QString::fromLatin1(kOpenPCDetDownloadUrl),
                    {QCryptographicHash::Sha256, QByteArray(kOpenPCDetSha256)},
                    QString::fromLatin1(kOpenPCDetGroupDir)};
    }
    Q_UNREACHABLE();
    return {};
}

// ----------------------------------------------------------------------------
// Singleton
// ----------------------------------------------------------------------------

ecvTestDataRepository& ecvTestDataRepository::instance() {
    static ecvTestDataRepository s_instance;
    return s_instance;
}

ecvTestDataRepository::ecvTestDataRepository(QObject* parent)
    : QObject(parent), m_downloader(new ecvModelDownloader(this)) {
    connect(m_downloader, &ecvModelDownloader::progress, this,
            &ecvTestDataRepository::onDownloaderProgress);
    connect(m_downloader, &ecvModelDownloader::finished, this,
            &ecvTestDataRepository::onDownloaderFinished);
    // Surface the transport-level detail (resume attempts, stall aborts,
    // the actual Qt error string) that the wrapper messages below omit.
    connect(m_downloader, &ecvModelDownloader::logMessage, this,
            [this](const QString& message) {
                emit downloadLogMessage(message);
            });
}

ecvTestDataRepository::~ecvTestDataRepository() = default;

// ----------------------------------------------------------------------------
// Availability query
// ----------------------------------------------------------------------------

bool ecvTestDataRepository::isExtractedComplete(Dataset kind) const {
    // A directory alone is not a valid cache marker: an interrupted extract
    // leaves the root behind. Validate the files that each consumer needs so
    // a partial cache can fall through to the intact zip or a fresh download.
    const QString extract = extractPath(kind);
    bool extractedComplete = false;
    switch (kind) {
        case Dataset::RelateAnything:
            extractedComplete =
                    !findDatasetFile(kind, QStringLiteral("skateboard.jpg"))
                             .isEmpty() &&
                    !findDatasetFile(
                             kind,
                             QStringLiteral("r3c_adoption_54729571599.webm"))
                             .isEmpty();
            break;
        case Dataset::RAM:
            extractedComplete =
                    !findDatasetFile(kind, QStringLiteral("demo1.jpg"))
                             .isEmpty();
            break;
        case Dataset::Monstree:
            extractedComplete = !getMonstreeImages(extract).isEmpty();
            break;
        case Dataset::FriendsFaces:
            extractedComplete = !findFriendsVideo(extract).isEmpty();
            break;
        case Dataset::ObjectsDetection: {
            const QStringList required = {
                    QStringLiteral("bus.jpg"),
                    QStringLiteral("000000397133.jpg"),
                    QStringLiteral("cat.jpg"),
                    QStringLiteral("aerial_airport.jpg"),
                    QStringLiteral("deeplsd_examples.jpg"),
                    QStringLiteral("people-walking.mp4"),
                    QStringLiteral("vehicles.mp4"),
                    QStringLiteral("supervision_demo.mp4"),
                    QStringLiteral("traffic.mp4")};
            extractedComplete = true;
            for (const QString& fileName : required) {
                if (findDatasetFile(kind, fileName).isEmpty()) {
                    extractedComplete = false;
                    break;
                }
            }
            break;
        }
        case Dataset::Image2Mesh: {
            // The main single-image-to-3D samples live in examples_images/.
            // Extensions do not prove valid contents; corrupt caches must
            // continue through verified ZIP/download recovery.
            const auto images = getImage2MeshImages(extract);
            extractedComplete = std::any_of(
                    images.cbegin(), images.cend(), [](const QString& path) {
                        return !QImageReader(path).read().isNull();
                    });
            break;
        }
        case Dataset::SAM3:
            // Both the segmentation images and the tracking videos must be
            // present for the qSAM3 sample-data flow to work.
            extractedComplete = !getSamImages(extract).isEmpty() &&
                                !getSamVideos(extract).isEmpty();
            break;
        case Dataset::GeneralKeypointDetection:
            // The official GKDT demo images must all be present.
            extractedComplete =
                    !getGeneralKeypointDetectionImages(extract).isEmpty();
            break;
        case Dataset::LightGlueTestImages:
            // The demo pair must both be present in the extraction tree.
            extractedComplete =
                    !findDatasetFile(kind, QStringLiteral("sacre_coeur1.jpg"))
                             .isEmpty() &&
                    !findDatasetFile(kind, QStringLiteral("sacre_coeur2.jpg"))
                             .isEmpty();
            break;
        case Dataset::LingbotMapCourthouse:
        case Dataset::LingbotMapLoop:
        case Dataset::LingbotMapOxford:
        case Dataset::LingbotMapUniversity:
        case Dataset::LingbotMapOxfordSkyMasks:
        case Dataset::LingbotMapUniversitySkyMasks:
            // A LingBot-Map bundle is complete when its ordered frame
            // sequence (or mask PNGs) is present.
            extractedComplete = !getLingbotMapImages(extract).isEmpty();
            break;
        case Dataset::LingbotMapDriveVideo:
        case Dataset::LingbotMapLingboWorldVideo:
            // Single-file mp4 datasets: complete when the video is
            // materialized under the extract directory.
            extractedComplete =
                    !findDatasetFile(kind, getDatasetInfo(kind).zipFileName)
                             .isEmpty();
            break;
        case Dataset::MapAnythingCourtyard:
        case Dataset::MapAnythingDeliveryArea:
        case Dataset::MapAnythingElectro:
        case Dataset::MapAnythingFacade:
        case Dataset::MapAnythingKicker:
        case Dataset::MapAnythingMeadow:
        case Dataset::MapAnythingOffice:
        case Dataset::MapAnythingPipes:
        case Dataset::MapAnythingPlayground:
        case Dataset::MapAnythingRelief:
        case Dataset::MapAnythingRelief2:
        case Dataset::MapAnythingTerrace:
        case Dataset::MapAnythingTerrains:
            // A MapAnything scene bundle is complete when its multi-view
            // image sequence is present.
            extractedComplete = !getMapAnythingSceneImages(extract).isEmpty();
            break;
        case Dataset::OpenPCDetSample:
            // Complete when the three LiDAR inputs the panels fill are all
            // present (the scene bundles ride along in the same archive).
            extractedComplete =
                    !findDatasetFile(kind, QStringLiteral("kitti_000008.bin"))
                             .isEmpty() &&
                    !findDatasetFile(
                             kind,
                             QStringLiteral("kitti_000008_sampled16384.bin"))
                             .isEmpty() &&
                    !findDatasetFile(kind,
                                     QStringLiteral("nuscenes_n015_lidar.bin"))
                             .isEmpty();
            break;
    }
    return extractedComplete;
}

bool ecvTestDataRepository::isDatasetAvailable(Dataset kind) const {
    if (isExtractedComplete(kind)) return true;

    // Check if a verified zip is cached. DeepVerify self-heals archives
    // downloaded before the integrity ledger existed (one hash pass, then
    // the state is recorded and later checks are stat-only).
    const auto info = getDatasetInfo(kind);
    const QString zip = zipPath(kind);
    return ecvAssetIntegrity::isVerified(zip, info.anchor, 0, false,
                                         ecvAssetIntegrity::OnMiss::DeepVerify);
}

void ecvTestDataRepository::finishExtraction(Dataset kind, bool success) {
    emit extractionFinished(success, kind);
    // ensureDatasetReady convergence: exactly one datasetReady per accepted
    // call, from this single choke point.
    if (m_ensurePending && kind == m_ensureKind) {
        m_ensurePending = false;
        emit datasetReady(success, kind);
    }
}

void ecvTestDataRepository::ensureDatasetReady(Dataset kind) {
    if (isExtractedComplete(kind)) {
        emit datasetReady(true, kind);
        return;
    }
    if (m_ensurePending) {
        if (kind == m_ensureKind) return;  // already preparing exactly this
        emit downloadLogMessage(QStringLiteral(
                "[Warning] Another dataset preparation is in progress"));
        emit datasetReady(false, kind);
        return;
    }
    if (m_downloadInProgress) {
        emit downloadLogMessage(QStringLiteral(
                "[Warning] A dataset download is already in progress"));
        emit datasetReady(false, kind);
        return;
    }
    m_ensurePending = true;
    m_ensureKind = kind;
    const auto info = getDatasetInfo(kind);
    if (ecvAssetIntegrity::isVerified(zipPath(kind), info.anchor, 0, false,
                                      ecvAssetIntegrity::OnMiss::DeepVerify)) {
        // Verified zip cached: extract now (synchronous; converges through
        // finishExtraction before this call returns).
        extractDataset(kind);
    } else {
        // Downloads converge through onDownloaderFinished.
        startDownload(kind);
    }
}

// ----------------------------------------------------------------------------
// Download
// ----------------------------------------------------------------------------

void ecvTestDataRepository::startDownload(Dataset kind) {
    if (m_downloadInProgress) {
        emit downloadLogMessage(
                QStringLiteral("[Warning] Download already in progress"));
        return;
    }

    const auto info = getDatasetInfo(kind);
    m_currentDataset = kind;

    // Ensure directories exist
    QDir().mkpath(downloadDir());
    QDir().mkpath(extractDir());

    const QString destPath = zipPath(kind);

    // Check if already downloaded and verified
    if (ecvAssetIntegrity::isVerified(destPath, info.anchor, 0, false,
                                      ecvAssetIntegrity::OnMiss::DeepVerify)) {
        emit downloadLogMessage(
                QStringLiteral("[Info] %1 dataset already downloaded")
                        .arg(info.displayName));
        m_downloadInProgress = false;
        emit downloadFinished(true, kind);
        // ensureDatasetReady chain (zip got verified between our check and
        // startDownload): converge here as well.
        if (m_ensurePending && kind == m_ensureKind) {
            extractDataset(kind);
        }
        return;
    }

    // Remove invalid cached file (and its ledger, so the artifact and the
    // recorded evidence never diverge).
    if (QFileInfo::exists(destPath)) {
        QFile::remove(destPath);
        ecvAssetIntegrity::invalidate(destPath);
    }

    m_downloadInProgress = true;
    emit downloadLogMessage(QStringLiteral("[Info] Downloading %1 dataset...")
                                    .arg(info.displayName));

    ecvModelDownloader::Request request;
    request.url = info.downloadUrl;
    request.destPath = destPath;
    request.minBytes = 1024 * 1024;       // At least 1 MB
    request.requireGgufMagic = false;     // Not a GGUF file
    request.contentAnchor = info.anchor;  // streamed digest check at ingestion

    m_downloader->download(request);
}

void ecvTestDataRepository::cancelDownload() {
    if (m_downloadInProgress && m_downloader) {
        m_downloader->cancel();
        m_downloadInProgress = false;
    }
}

void ecvTestDataRepository::onDownloaderProgress(qint64 received,
                                                 qint64 total) {
    if (total <= 0) return;
    const int percent = static_cast<int>((received * 100) / total);
    const QString status =
            QStringLiteral("%1 / %2 (%3%)")
                    .arg(ecvModelDownloader::formatFileSize(received),
                         ecvModelDownloader::formatFileSize(total))
                    .arg(percent);
    emit downloadProgress(percent, status);
}

void ecvTestDataRepository::onDownloaderFinished(bool ok,
                                                 const QString& destPath) {
    m_downloadInProgress = false;
    const auto info = getDatasetInfo(m_currentDataset);

    if (!ok) {
        emit downloadLogMessage(
                QStringLiteral("[Error] Failed to download %1 dataset")
                        .arg(info.displayName));
        emit downloadFinished(false, m_currentDataset);
        if (m_ensurePending && m_currentDataset == m_ensureKind) {
            m_ensurePending = false;
            emit datasetReady(false, m_currentDataset);
        }
        return;
    }

    // Content verification already happened inside the downloader (the
    // digest was streamed while writing) and the verified state is
    // recorded in the zip's integrity ledger — nothing to re-hash here.
    emit downloadLogMessage(
            QStringLiteral("[Info] Downloaded %1 dataset successfully")
                    .arg(info.displayName));
    emit downloadFinished(true, m_currentDataset);
    // ensureDatasetReady chain: the repository owns the download->extract
    // ordering (legacy downloadFinished consumers stay informational).
    if (m_ensurePending && m_currentDataset == m_ensureKind) {
        extractDataset(m_currentDataset);
    }
}

// ----------------------------------------------------------------------------
// Extraction (using minizip for cross-platform consistency)
// ----------------------------------------------------------------------------

namespace {

constexpr int kExtractBufferSize = 8192;
constexpr int kMaxZipEntryNameLen = 1024;

// RAII wrapper for minizip unzFile
struct UnzFileCloser {
    void operator()(unzFile uf) const {
        if (uf) unzClose(uf);
    }
};
using UniqueUnzFile = std::unique_ptr<void, UnzFileCloser>;

#ifdef Q_OS_WIN
// Windows: _wfopen supports Unicode paths (UTF-16), unlike fopen which uses
// the ANSI code page. minizip's unzOpen64 internally calls fopen, so we
// provide a custom file function table that uses _wfopen.
static voidpf ZCALLBACK wfopen64_file_func(voidpf /*opaque*/,
                                           const void* filename,
                                           int mode) {
    const wchar_t* wpath = static_cast<const wchar_t*>(filename);
    const wchar_t* wmode = nullptr;
    if ((mode & ZLIB_FILEFUNC_MODE_READWRITEFILTER) == ZLIB_FILEFUNC_MODE_READ)
        wmode = L"rb";
    else if (mode & ZLIB_FILEFUNC_MODE_EXISTING)
        wmode = L"r+b";
    else if (mode & ZLIB_FILEFUNC_MODE_CREATE)
        wmode = L"wb";
    if (wpath && wmode) return _wfopen(wpath, wmode);
    return nullptr;
}
#endif

UniqueUnzFile openUnzFile(const QString& path) {
#ifdef Q_OS_WIN
    // Pass wide-string path so our custom open func can use _wfopen
    zlib_filefunc64_def filefunc;
    fill_fopen64_filefunc(&filefunc);
    filefunc.zopen64_file = wfopen64_file_func;
    const std::wstring wpath = path.toStdWString();
    return UniqueUnzFile(unzOpen2_64(wpath.c_str(), &filefunc));
#else
    // macOS/Linux: fopen uses UTF-8 natively
    return UniqueUnzFile(unzOpen64(path.toUtf8().constData()));
#endif
}

int extractCurrentZipEntry(unzFile uf, const QString& extractDir) {
    char filename_inzip[kMaxZipEntryNameLen] = {};

    unz_file_info64 file_info;
    int err = unzGetCurrentFileInfo64(uf, &file_info, filename_inzip,
                                      sizeof(filename_inzip), nullptr, 0,
                                      nullptr, 0);
    if (err != UNZ_OK) return err;

    const QString entryName = QString::fromUtf8(filename_inzip);

    // Security: reject path traversal attempts
    if (entryName.contains(QStringLiteral("..")) ||
        entryName.startsWith(QLatin1Char('/')) ||
        entryName.startsWith(QLatin1Char('\\'))) {
        return UNZ_ERRNO;
    }

    // Directory entry — just create it
    if (entryName.endsWith(QLatin1Char('/'))) {
        QDir().mkpath(QDir(extractDir).filePath(entryName));
        return UNZ_OK;
    }

    // File entry — open, read, write
    err = unzOpenCurrentFilePassword(uf, nullptr);
    if (err != UNZ_OK) return err;

    const QString filePath = QDir(extractDir).filePath(entryName);
    QDir().mkpath(QFileInfo(filePath).path());

    QFile outFile(filePath);
    if (!outFile.open(QIODevice::WriteOnly)) {
        unzCloseCurrentFile(uf);
        return UNZ_ERRNO;
    }

    QByteArray buf(kExtractBufferSize, Qt::Uninitialized);
    do {
        err = unzReadCurrentFile(uf, buf.data(), buf.size());
        if (err < 0) break;
        if (err > 0) {
            if (outFile.write(buf.constData(), err) != err) {
                err = UNZ_ERRNO;
                break;
            }
        }
    } while (err > 0);

    outFile.close();

    if (err == UNZ_OK) {
        err = unzCloseCurrentFile(uf);
    } else {
        unzCloseCurrentFile(uf);
    }
    return err;
}

bool extractFromZipFile(const QString& zipPath,
                        const QString& extractDir,
                        const std::function<void(int, int)>& onProgress) {
    if (zipPath.isEmpty()) return false;

    auto uf = openUnzFile(zipPath);
    if (!uf) return false;

    unz_global_info64 gi;
    int err = unzGetGlobalInfo64(uf.get(), &gi);
    if (err != UNZ_OK) return false;

    const int totalEntries = static_cast<int>(gi.number_entry);
    for (uLong i = 0; i < gi.number_entry; ++i) {
        err = extractCurrentZipEntry(uf.get(), extractDir);
        if (err != UNZ_OK) return false;

        if (onProgress && totalEntries > 0) {
            onProgress(static_cast<int>(i + 1), totalEntries);
        }

        if ((i + 1) < gi.number_entry) {
            err = unzGoToNextFile(uf.get());
            if (err != UNZ_OK) return false;
        }
    }

    return true;
}

}  // namespace

// ----------------------------------------------------------------------------
// Static utility methods
// ----------------------------------------------------------------------------

int ecvTestDataRepository::zipEntryCount(const QString& zipPath) {
    if (zipPath.isEmpty() || !QFileInfo::exists(zipPath)) return 0;
    auto uf = openUnzFile(QFileInfo(zipPath).absoluteFilePath());
    if (!uf) return 0;
    unz_global_info64 gi;
    const int err = unzGetGlobalInfo64(uf.get(), &gi);
    return err == UNZ_OK ? static_cast<int>(gi.number_entry) : 0;
}

bool ecvTestDataRepository::extractZip(const QString& zipPath,
                                       const QString& extractDir,
                                       const ExtractProgressFn& onProgress) {
    if (zipPath.isEmpty() || !QFileInfo::exists(zipPath)) return false;
    QDir().mkpath(extractDir);
    return extractFromZipFile(QFileInfo(zipPath).absoluteFilePath(), extractDir,
                              onProgress);
}

bool ecvTestDataRepository::isSingleFileDataset(Dataset kind) {
    switch (kind) {
        case Dataset::LingbotMapDriveVideo:
        case Dataset::LingbotMapLingboWorldVideo:
            return true;
        default:
            return false;
    }
}

bool ecvTestDataRepository::extractDataset(Dataset kind) {
    const auto info = getDatasetInfo(kind);
    const QString zip = zipPath(kind);
    // Grouped datasets extract UNDER their group directory so the zip's
    // own top-level folder lands exactly at extractPath(kind)
    // (extract/<group>/<zip-root>); ungrouped datasets use the extract
    // root (legacy layout).
    const QString extract =
            info.extractGroupName.isEmpty()
                    ? extractDir()
                    : QDir(extractDir()).filePath(info.extractGroupName);

    if (!ecvAssetIntegrity::isVerified(zip, info.anchor, 0, false,
                                       ecvAssetIntegrity::OnMiss::DeepVerify)) {
        emit downloadLogMessage(
                QStringLiteral("[Error] Zip file is missing or invalid: %1")
                        .arg(zip));
        finishExtraction(kind, false);
        return false;
    }

    // Single-file datasets (e.g. the LingBot-Map long-model mp4s): the
    // downloaded file itself is the content. Materialize it AT
    // extractPath(kind) (group directory included) and skip the zip
    // machinery entirely.
    if (isSingleFileDataset(kind)) {
        const QString dest = extractPath(kind);
        if (!QDir().mkpath(QFileInfo(dest).absolutePath())) {
            emit downloadLogMessage(
                    QStringLiteral(
                            "[Error] Cannot create extract directory: %1")
                            .arg(QFileInfo(dest).absolutePath()));
            finishExtraction(kind, false);
            return false;
        }
        QFile::remove(dest);
        if (!QFile::copy(zip, dest)) {
            emit downloadLogMessage(
                    QStringLiteral("[Error] Cannot materialize %1 -> %2")
                            .arg(zip, dest));
            finishExtraction(kind, false);
            return false;
        }
        emit downloadLogMessage(
                QStringLiteral("[Test data] Materialized %1 -> %2")
                        .arg(QFileInfo(zip).fileName(), dest));
        emit extractionProgress(1, 1);
        finishExtraction(kind, true);
        return true;
    }

    // Ensure extract directory exists
    if (!QDir().mkpath(extract)) {
        emit downloadLogMessage(
                QStringLiteral("[Error] Cannot create extract directory: %1")
                        .arg(extract));
        finishExtraction(kind, false);
        return false;
    }

    // Use minizip for cross-platform consistency
    auto onProgress = [this](int current, int total) {
        emit extractionProgress(current, total);
    };

    const bool ok = extractZip(zip, extract, onProgress);

    if (!ok) {
        emit downloadLogMessage(
                QStringLiteral("[Error] Extraction failed: %1").arg(zip));
        finishExtraction(kind, false);
        return false;
    }

    // Verify extraction result
    const QString expectedDir = extractPath(kind);
    if (!QDir(expectedDir).exists()) {
        emit downloadLogMessage(
                QStringLiteral("[Error] Extraction completed but expected "
                               "directory not found: %1")
                        .arg(expectedDir));
        finishExtraction(kind, false);
        return false;
    }

    emit downloadLogMessage(
            QStringLiteral("[Info] Extracted %1 dataset successfully: %2 -> "
                           "%3")
                    .arg(info.displayName, zip, extract));
    finishExtraction(kind, true);
    return true;
}

// ----------------------------------------------------------------------------
// Dataset-specific helpers
// ----------------------------------------------------------------------------

QStringList ecvTestDataRepository::getMonstreeImages(
        const QString& bundleRoot) {
    if (bundleRoot.isEmpty()) return {};

    // Look specifically in mini3/ subdirectory
    const QString imageDir = QDir(bundleRoot).filePath(QStringLiteral("mini3"));
    if (!QDir(imageDir).exists()) return {};

    const QStringList patterns = {
            QStringLiteral("*.jpg"),  QStringLiteral("*.jpeg"),
            QStringLiteral("*.png"),  QStringLiteral("*.tif"),
            QStringLiteral("*.tiff"), QStringLiteral("*.webp")};

    QStringList images;
    QDirIterator it(imageDir, patterns, QDir::Files);
    while (it.hasNext()) {
        const QString path = it.next();
        const QString fileName = QFileInfo(path).fileName();
        if (fileName.startsWith(QLatin1Char('.'))) continue;
        images.append(QFileInfo(path).absoluteFilePath());
    }
    images.sort(Qt::CaseInsensitive);
    return images;
}

QString ecvTestDataRepository::findFriendsVideo(const QString& bundleRoot) {
    if (bundleRoot.isEmpty()) return {};

    // Prefer the known path first
    const QString knownRel =
            QDir(bundleRoot).filePath(QStringLiteral("query/friends_demo.mp4"));
    if (QFileInfo::exists(knownRel))
        return QFileInfo(knownRel).absoluteFilePath();

    // Fall back to recursive search in query/ subdirectory
    const QString videoDir = QDir(bundleRoot).filePath(QStringLiteral("query"));
    if (!QDir(videoDir).exists()) return {};

    const QStringList patterns = {
            QStringLiteral("*.mp4"), QStringLiteral("*.mov"),
            QStringLiteral("*.avi"), QStringLiteral("*.mkv"),
            QStringLiteral("*.webm")};

    QDirIterator it(videoDir, patterns, QDir::Files,
                    QDirIterator::Subdirectories);
    QString best;
    while (it.hasNext()) {
        const QString path = it.next();
        const QString fileName = QFileInfo(path).fileName();
        if (fileName.startsWith(QLatin1Char('.'))) continue;
        // Prefer files with "friend" in the name
        if (fileName.contains(QStringLiteral("friend"), Qt::CaseInsensitive))
            return QFileInfo(path).absoluteFilePath();
        if (best.isEmpty()) best = QFileInfo(path).absoluteFilePath();
    }
    return best;
}

QStringList ecvTestDataRepository::getImage2MeshImages(
        const QString& bundleRoot) {
    if (bundleRoot.isEmpty()) return {};

    // The curated single-image-to-3D samples live in examples_images/.
    const QString imageDir =
            QDir(bundleRoot).filePath(QStringLiteral("examples_images"));
    if (!QDir(imageDir).exists()) return {};

    const QStringList patterns = {
            QStringLiteral("*.jpg"), QStringLiteral("*.jpeg"),
            QStringLiteral("*.png"), QStringLiteral("*.webp")};

    QStringList images;
    QDirIterator it(imageDir, patterns, QDir::Files);
    while (it.hasNext()) {
        const QString path = it.next();
        const QString fileName = QFileInfo(path).fileName();
        if (fileName.startsWith(QLatin1Char('.'))) continue;
        images.append(QFileInfo(path).absoluteFilePath());
    }
    images.sort(Qt::CaseInsensitive);
    return images;
}

QStringList ecvTestDataRepository::getGeneralKeypointDetectionImages(
        const QString& bundleRoot) {
    if (bundleRoot.isEmpty()) return {};

    // The official GKDT demo images sit directly in the bundle root.
    if (!QDir(bundleRoot).exists()) return {};

    const QStringList patterns = {
            QStringLiteral("*.jpg"), QStringLiteral("*.jpeg"),
            QStringLiteral("*.png"), QStringLiteral("*.webp")};

    QStringList images;
    QDirIterator it(bundleRoot, patterns, QDir::Files);
    while (it.hasNext()) {
        const QString path = it.next();
        const QString fileName = QFileInfo(path).fileName();
        if (fileName.startsWith(QLatin1Char('.'))) continue;
        images.append(QFileInfo(path).absoluteFilePath());
    }
    images.sort(Qt::CaseInsensitive);
    return images;
}

QStringList ecvTestDataRepository::getLingbotMapImages(
        const QString& bundleRoot) {
    if (bundleRoot.isEmpty()) return {};
    if (!QDir(bundleRoot).exists()) return {};

    const QStringList patterns = {
            QStringLiteral("*.png"), QStringLiteral("*.jpg"),
            QStringLiteral("*.jpeg"), QStringLiteral("*.bmp")};

    // The lingbot_map_data archives keep a top-level scene directory
    // (<zip stem>/<frame>.png), so the scan is recursive.
    QStringList images;
    QDirIterator it(bundleRoot, patterns, QDir::Files,
                    QDirIterator::Subdirectories);
    while (it.hasNext()) {
        const QString path = it.next();
        const QString fileName = QFileInfo(path).fileName();
        if (fileName.startsWith(QLatin1Char('.'))) continue;
        images.append(QFileInfo(path).absoluteFilePath());
    }
    images.sort(Qt::CaseInsensitive);
    return images;
}

QStringList ecvTestDataRepository::getMapAnythingSceneImages(
        const QString& bundleRoot) {
    // The map_anything_vggt_test_data scene zips keep a top-level scene
    // directory (<zip stem>/<...>/frames), so the scan is recursive (same
    // contract as getLingbotMapImages).
    return getLingbotMapImages(bundleRoot);
}

QStringList ecvTestDataRepository::getSamImages(const QString& bundleRoot) {
    if (bundleRoot.isEmpty()) return {};

    const QString imageDir =
            QDir(bundleRoot).filePath(QStringLiteral("images"));
    if (!QDir(imageDir).exists()) return {};

    const QStringList patterns = {
            QStringLiteral("*.jpg"), QStringLiteral("*.jpeg"),
            QStringLiteral("*.png"), QStringLiteral("*.webp")};

    QStringList images;
    QDirIterator it(imageDir, patterns, QDir::Files);
    while (it.hasNext()) {
        const QString path = it.next();
        const QString fileName = QFileInfo(path).fileName();
        if (fileName.startsWith(QLatin1Char('.'))) continue;
        images.append(QFileInfo(path).absoluteFilePath());
    }
    images.sort(Qt::CaseInsensitive);
    return images;
}

QStringList ecvTestDataRepository::getSam3dObjectImages(
        const QString& bundleRoot) {
    if (bundleRoot.isEmpty()) return {};

    // The official scene images live in <root>/sam3d_images/ inside the
    // Image2Mesh bundle (image_to_mesh_data.zip).
    const QString imageDir =
            QDir(bundleRoot).filePath(QStringLiteral("sam3d_images"));
    if (!QDir(imageDir).exists()) return {};

    const QStringList patterns = {
            QStringLiteral("*.png"), QStringLiteral("*.jpg"),
            QStringLiteral("*.jpeg"), QStringLiteral("*.webp")};

    QStringList images;
    QDirIterator it(imageDir, patterns, QDir::Files,
                    QDirIterator::Subdirectories);
    while (it.hasNext()) {
        const QString path = it.next();
        const QString fileName = QFileInfo(path).fileName();
        if (fileName.startsWith(QLatin1Char('.'))) continue;
        images.append(QFileInfo(path).absoluteFilePath());
    }
    images.sort(Qt::CaseInsensitive);
    return images;
}

QStringList ecvTestDataRepository::getSamVideos(const QString& bundleRoot) {
    if (bundleRoot.isEmpty()) return {};

    const QString videoDir =
            QDir(bundleRoot).filePath(QStringLiteral("videos"));
    if (!QDir(videoDir).exists()) return {};

    const QStringList patterns = {
            QStringLiteral("*.mp4"), QStringLiteral("*.mov"),
            QStringLiteral("*.avi"), QStringLiteral("*.mkv"),
            QStringLiteral("*.webm")};

    QStringList videos;
    QDirIterator it(videoDir, patterns, QDir::Files);
    while (it.hasNext()) {
        const QString path = it.next();
        const QString fileName = QFileInfo(path).fileName();
        if (fileName.startsWith(QLatin1Char('.'))) continue;
        videos.append(QFileInfo(path).absoluteFilePath());
    }
    videos.sort(Qt::CaseInsensitive);
    return videos;
}

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "MapAnythingSceneMeta.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QVector>
#include <cmath>
#include <limits>

namespace MapAnythingSceneMeta {

namespace {

// WGS84 (COLMAP GPSTransform::Ellipsoid::WGS84).
constexpr double kA = 6378137.0;
constexpr double kF = 1.0 / 298.257223563;
constexpr double kE2 = kF * (2.0 - kF);
constexpr double kDegToRad = 3.14159265358979323846 / 180.0;

}  // namespace

void wgs84ToEcef(const double lla[3], double xyz[3]) {
    const double lat = lla[0] * kDegToRad;
    const double lon = lla[1] * kDegToRad;
    const double sinLat = std::sin(lat);
    const double cosLat = std::cos(lat);
    const double n = kA / std::sqrt(1.0 - kE2 * sinLat * sinLat);
    xyz[0] = (n + lla[2]) * cosLat * std::cos(lon);
    xyz[1] = (n + lla[2]) * cosLat * std::sin(lon);
    xyz[2] = (n * (1.0 - kE2) + lla[2]) * sinLat;
}

void wgs84ToEnu(const double lla[3], const double origin[3], double enu[3]) {
    double p[3], o[3];
    wgs84ToEcef(lla, p);
    wgs84ToEcef(origin, o);
    const double d[3] = {p[0] - o[0], p[1] - o[1], p[2] - o[2]};
    const double sinLat = std::sin(origin[0] * kDegToRad);
    const double cosLat = std::cos(origin[0] * kDegToRad);
    const double sinLon = std::sin(origin[1] * kDegToRad);
    const double cosLon = std::cos(origin[1] * kDegToRad);
    enu[0] = -sinLon * d[0] + cosLon * d[1];
    enu[1] = -sinLat * cosLon * d[0] - sinLat * sinLon * d[1] + cosLat * d[2];
    enu[2] = cosLat * cosLon * d[0] + cosLat * sinLon * d[1] + sinLat * d[2];
}

QString discover(const QStringList& images) {
    if (images.isEmpty()) return QString();
    const QString dir = QFileInfo(images.front()).absolutePath();
    if (QFileInfo(dir).fileName() != QStringLiteral("images")) return QString();
    for (const QString& image : images) {
        if (QFileInfo(image).absolutePath() != dir) return QString();
    }
    const QString meta =
            QDir(dir).absoluteFilePath(QStringLiteral("../scene_meta.json"));
    return QFileInfo::exists(meta) ? QFileInfo(meta).absoluteFilePath()
                                   : QString();
}

bool loadCenters(const QString& metaPath,
                 const QStringList& images,
                 std::vector<float>* centers,
                 QString* error) {
    QFile file(metaPath);
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) *error = QStringLiteral("cannot read %1").arg(metaPath);
        return false;
    }
    const QJsonValue framesValue =
            QJsonDocument::fromJson(file.readAll()).object().value("frames");
    if (!framesValue.isArray()) {
        if (error)
            *error = QStringLiteral("%1 has no frames array").arg(metaPath);
        return false;
    }
    QHash<QString, QVector<float>> byName;
    for (const QJsonValue& value : framesValue.toArray()) {
        const QJsonObject frame = value.toObject();
        const QJsonArray matrix = frame.value("transform_matrix").toArray();
        if (matrix.size() < 3) continue;
        QVector<float> center(3);
        bool valid = true;
        for (int r = 0; r < 3 && valid; ++r) {
            const QJsonArray row = matrix[r].toArray();
            valid = row.size() >= 4;
            if (valid) center[r] = static_cast<float>(row[3].toDouble());
        }
        if (valid) byName.insert(frame.value("frame_name").toString(), center);
    }
    centers->assign((size_t)images.size() * 3,
                    std::numeric_limits<float>::quiet_NaN());
    for (int v = 0; v < images.size(); ++v) {
        // completeBaseName drops only the last suffix (Python Path.stem).
        const QVector<float> c =
                byName.value(QFileInfo(images[v]).completeBaseName());
        if (c.size() != 3) continue;
        for (int r = 0; r < 3; ++r) (*centers)[(size_t)v * 3 + r] = c[r];
    }
    return true;
}

bool loadColmapReferences(const QString& path,
                          const QStringList& images,
                          bool isGps,
                          ReferenceCenters* out,
                          QString* error) {
    auto fail = [error](const QString& message) {
        if (error) *error = message;
        return false;
    };
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return fail(QStringLiteral("cannot read %1").arg(path));
    }
    // Image file name -> indices of the inputs carrying it, so a line is
    // resolved in O(1) plus a suffix check over its namesakes.
    QStringList normalized;
    QHash<QString, QVector<int>> byFileName;
    for (int v = 0; v < images.size(); ++v) {
        normalized << QDir::fromNativeSeparators(
                QFileInfo(images[v]).absoluteFilePath());
        byFileName[QFileInfo(images[v]).fileName()].append(v);
    }

    ReferenceCenters r;
    r.gps = isGps;
    r.centers.assign((size_t)images.size() * 3,
                     std::numeric_limits<float>::quiet_NaN());
    bool first = true;
    int lineNo = 0;
    while (!file.atEnd()) {
        ++lineNo;
        const QString line = QString::fromUtf8(file.readLine()).trimmed();
        if (line.isEmpty() || line.startsWith(QLatin1Char('#'))) continue;
        const QStringList tok = line.simplified().split(QLatin1Char(' '));
        double c[3];
        bool ok = tok.size() >= 4;
        for (int k = 0; ok && k < 3; ++k) {
            c[k] = tok[k + 1].toDouble(&ok);
            ok = ok && std::isfinite(c[k]);
        }
        if (!ok) {
            return fail(QStringLiteral("%1:%2: expected \"<image_name> X Y "
                                       "Z\", got \"%3\"")
                                .arg(path)
                                .arg(lineNo)
                                .arg(line));
        }
        if (isGps && (std::abs(c[0]) > 90.0 || std::abs(c[1]) > 180.0)) {
            return fail(QStringLiteral("%1:%2: latitude/longitude out of "
                                       "range (GPS mode expects \"<image_name> "
                                       "lat lon alt\" in degrees)")
                                .arg(path)
                                .arg(lineNo));
        }
        // The first line fixes the local frame for every later one, matched
        // or not (COLMAP: ENU origin = ref_locations[0]).
        if (first) {
            first = false;
            for (int k = 0; k < 3; ++k) {
                if (isGps) {
                    r.enuOrigin[k] = c[k];
                } else if (std::abs(c[k]) >= kShiftThreshold) {
                    r.offset[k] = std::trunc(c[k] / 100.0) * 100.0;
                }
            }
        }
        const QString name = QDir::fromNativeSeparators(tok[0]);
        int hit = -1;
        for (int v : byFileName.value(QFileInfo(name).fileName())) {
            if (normalized[v] != name &&
                !normalized[v].endsWith(QLatin1Char('/') + name))
                continue;
            if (hit >= 0) {
                return fail(QStringLiteral("%1:%2: \"%3\" matches both %4 "
                                           "and %5")
                                    .arg(path)
                                    .arg(lineNo)
                                    .arg(tok[0], images[hit], images[v]));
            }
            hit = v;
        }
        if (hit < 0) continue;
        float* dst = &r.centers[(size_t)hit * 3];
        if (!std::isnan(dst[0])) {
            return fail(QStringLiteral("%1:%2: %3 already has a reference "
                                       "line")
                                .arg(path)
                                .arg(lineNo)
                                .arg(images[hit]));
        }
        double local[3];
        if (isGps) {
            wgs84ToEnu(c, r.enuOrigin, local);
        } else {
            for (int k = 0; k < 3; ++k) local[k] = c[k] - r.offset[k];
        }
        for (int k = 0; k < 3; ++k) dst[k] = static_cast<float>(local[k]);
        ++r.matched;
    }
    if (r.matched < 3) {
        return fail(QStringLiteral("%1 names %2 of the %3 input images; the "
                                   "Sim3 needs at least 3")
                            .arg(path)
                            .arg(r.matched)
                            .arg(images.size()));
    }
    *out = std::move(r);
    return true;
}

}  // namespace MapAnythingSceneMeta

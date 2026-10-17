// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <ostream>
#include <utility>
#include <vector>

#include "util/types.h"

namespace colmap {

struct ImuCalibration {
    double gyro_noise_density = 0.00073088444;
    double accel_noise_density = 0.01883649;
    double bias_gyro_random_walk_sigma = 0.00038765;
    double bias_accel_random_walk_sigma = 0.012589254;
    double gyro_saturation_max = 7.8;
    double accel_saturation_max = 150;
    double gravity_magnitude = 9.81007;
    double imu_rate = 20.0;
};

struct ImuMeasurement {
    timestamp_t timestamp = kInvalidTimestamp;
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
    Eigen::Vector3d accel = Eigen::Vector3d::Zero();

    ImuMeasurement() = default;
    ImuMeasurement(timestamp_t timestamp,
                   const Eigen::Vector3d& gyro,
                   const Eigen::Vector3d& accel)
        : timestamp(timestamp), gyro(gyro), accel(accel) {}
};

std::ostream& operator<<(std::ostream& stream,
                         const ImuCalibration& calibration);
std::ostream& operator<<(std::ostream& stream,
                         const ImuMeasurement& measurement);

class ImuMeasurements {
public:
    ImuMeasurements() = default;
    explicit ImuMeasurements(std::vector<ImuMeasurement> measurements) {
        Insert(std::move(measurements));
    }

    void Insert(const ImuMeasurement& measurement);
    void Insert(std::vector<ImuMeasurement> measurements);
    void Insert(ImuMeasurements measurements);
    void InsertSorted(std::vector<ImuMeasurement> sorted_measurements);
    void Remove(const ImuMeasurement& measurement);

    void Clear() { measurements_.clear(); }
    bool Empty() const { return measurements_.empty(); }
    size_t Size() const { return measurements_.size(); }
    const ImuMeasurement& operator[](size_t index) const {
        return measurements_[index];
    }
    std::vector<ImuMeasurement>::const_iterator begin() const {
        return measurements_.begin();
    }
    std::vector<ImuMeasurement>::const_iterator end() const {
        return measurements_.end();
    }
    const ImuMeasurement& front() const { return measurements_.front(); }
    const ImuMeasurement& back() const { return measurements_.back(); }
    const std::vector<ImuMeasurement>& Data() const { return measurements_; }

    void ExtractMeasurementsInRange(timestamp_t t1,
                                    timestamp_t t2,
                                    ImuMeasurements* measurements) const;

private:
    std::vector<ImuMeasurement> measurements_;
};

}  // namespace colmap

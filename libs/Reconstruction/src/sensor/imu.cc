// SPDX-License-Identifier: BSD-3-Clause

#include "sensor/imu.h"

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <string>

#include "util/logging.h"

namespace colmap {
namespace {

void ThrowIfDuplicatesOrNotSorted(const std::vector<ImuMeasurement>& values) {
  for (size_t i = 1; i < values.size(); ++i) {
    if (values[i].timestamp < values[i - 1].timestamp) {
      throw std::invalid_argument("ImuMeasurements are not sorted by timestamp");
    }
    if (values[i].timestamp == values[i - 1].timestamp) {
      throw std::invalid_argument("Duplicate timestamp in ImuMeasurements");
    }
  }
}

bool Earlier(const ImuMeasurement& lhs, const ImuMeasurement& rhs) {
  return lhs.timestamp < rhs.timestamp;
}

}  // namespace

void ImuMeasurements::Insert(const ImuMeasurement& measurement) {
  if (Empty() || measurement.timestamp > back().timestamp) {
    measurements_.push_back(measurement);
    return;
  }
  const auto it = std::lower_bound(
      measurements_.begin(), measurements_.end(), measurement, Earlier);
  if (it != measurements_.end() && it->timestamp == measurement.timestamp) {
    throw std::invalid_argument("Duplicate timestamp in ImuMeasurements");
  }
  measurements_.insert(it, measurement);
}

void ImuMeasurements::Insert(std::vector<ImuMeasurement> measurements) {
  std::sort(measurements.begin(), measurements.end(), Earlier);
  InsertSorted(std::move(measurements));
}

void ImuMeasurements::Insert(ImuMeasurements measurements) {
  if (Empty()) {
    measurements_ = std::move(measurements.measurements_);
  } else {
    InsertSorted(std::move(measurements.measurements_));
  }
}

void ImuMeasurements::InsertSorted(std::vector<ImuMeasurement> measurements) {
  if (measurements.empty()) return;
  ThrowIfDuplicatesOrNotSorted(measurements);
  if (Empty()) {
    measurements_ = std::move(measurements);
    return;
  }
  if (measurements.front().timestamp > back().timestamp) {
    measurements_.insert(measurements_.end(), measurements.begin(),
                         measurements.end());
    return;
  }
  std::vector<ImuMeasurement> merged;
  merged.reserve(measurements_.size() + measurements.size());
  std::merge(measurements_.begin(), measurements_.end(), measurements.begin(),
             measurements.end(), std::back_inserter(merged), Earlier);
  ThrowIfDuplicatesOrNotSorted(merged);
  measurements_ = std::move(merged);
}

void ImuMeasurements::Remove(const ImuMeasurement& measurement) {
  const auto it = std::lower_bound(
      measurements_.begin(), measurements_.end(), measurement, Earlier);
  if (it == measurements_.end() || it->timestamp != measurement.timestamp) {
    throw std::invalid_argument("Element not found in the list");
  }
  measurements_.erase(it);
}

void ImuMeasurements::ExtractMeasurementsInRange(
    const timestamp_t t1,
    const timestamp_t t2,
    ImuMeasurements* measurements) const {
  THROW_CHECK_NOTNULL(measurements);
  THROW_CHECK(!Empty());
  THROW_CHECK_LT(t1, t2);
  measurements->Clear();
  if (t1 < front().timestamp || t2 > back().timestamp) return;

  ImuMeasurement lower;
  lower.timestamp = t1;
  ImuMeasurement upper;
  upper.timestamp = t2;
  const auto first = std::upper_bound(begin(), end(), lower, Earlier);
  const auto last = std::lower_bound(begin(), end(), upper, Earlier);
  measurements->measurements_.assign(first - 1, last + 1);
}

std::ostream& operator<<(std::ostream& stream,
                         const ImuCalibration& calibration) {
  return stream << "ImuCalibration(gyro_noise_density="
                << calibration.gyro_noise_density << ", accel_noise_density="
                << calibration.accel_noise_density << ", gravity_magnitude="
                << calibration.gravity_magnitude << ")";
}

std::ostream& operator<<(std::ostream& stream,
                         const ImuMeasurement& measurement) {
  return stream << "ImuMeasurement(t=" << measurement.timestamp << ", gyro=["
                << measurement.gyro.transpose() << "], accel=["
                << measurement.accel.transpose() << "])";
}

}  // namespace colmap

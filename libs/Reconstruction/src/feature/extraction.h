// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef COLMAP_SRC_FEATURE_EXTRACTION_H_
#define COLMAP_SRC_FEATURE_EXTRACTION_H_

#include "base/database.h"
#include "base/image_reader.h"
#include "feature/sift.h"
#include "util/opengl_utils.h"
#include "util/threading.h"

namespace colmap {

namespace internal {

struct ImageData;

}  // namespace internal

// Feature extraction class to extract features for all images in a directory.
class SiftFeatureExtractor : public Thread {
public:
    SiftFeatureExtractor(const ImageReaderOptions& reader_options,
                         const SiftExtractionOptions& sift_options);

private:
    void Run();

    const ImageReaderOptions reader_options_;
    const SiftExtractionOptions sift_options_;

    Database database_;
    ImageReader image_reader_;

    std::vector<std::unique_ptr<Thread>> resizers_;
    std::vector<std::unique_ptr<Thread>> extractors_;
    std::unique_ptr<Thread> writer_;

    std::unique_ptr<JobQueue<internal::ImageData>> resizer_queue_;
    std::unique_ptr<JobQueue<internal::ImageData>> extractor_queue_;
    std::unique_ptr<JobQueue<internal::ImageData>> writer_queue_;
};

// Import features from text files. Each image must have a corresponding text
// file with the same name and an additional ".txt" suffix.
class FeatureImporter : public Thread {
public:
    FeatureImporter(const ImageReaderOptions& reader_options,
                    const std::string& import_path);

private:
    void Run();

    const ImageReaderOptions reader_options_;
    const std::string import_path_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

namespace internal {

struct ImageData {
    ImageReader::Status status = ImageReader::Status::FAILURE;

    Camera camera;
    Image image;
    Bitmap bitmap;
    Bitmap mask;

    FeatureKeypoints keypoints;
    FeatureDescriptors descriptors;
};

class ImageResizerThread : public Thread {
public:
    ImageResizerThread(const int max_image_size,
                       JobQueue<ImageData>* input_queue,
                       JobQueue<ImageData>* output_queue);

private:
    void Run();

    const int max_image_size_;

    JobQueue<ImageData>* input_queue_;
    JobQueue<ImageData>* output_queue_;
};

class SiftFeatureExtractorThread : public Thread {
public:
    SiftFeatureExtractorThread(const SiftExtractionOptions& sift_options,
                               const std::shared_ptr<Bitmap>& camera_mask,
                               JobQueue<ImageData>* input_queue,
                               JobQueue<ImageData>* output_queue);

private:
    void Run();

    const SiftExtractionOptions sift_options_;
    std::shared_ptr<Bitmap> camera_mask_;

    std::unique_ptr<OpenGLContextManager> opengl_context_;

    JobQueue<ImageData>* input_queue_;
    JobQueue<ImageData>* output_queue_;
};

class FeatureWriterThread : public Thread {
public:
    FeatureWriterThread(const size_t num_images,
                        Database* database,
                        JobQueue<ImageData>* input_queue);

private:
    void Run();

    const size_t num_images_;
    Database* database_;
    JobQueue<ImageData>* input_queue_;
};

}  // namespace internal

}  // namespace colmap

#endif  // COLMAP_SRC_FEATURE_EXTRACTION_H_

#define FORK
// TEMP-AUDIT: same-seed fork-vs-upstream scan of the pipeline contract cases.
// Compiled into both trees (FORK defined for the fork). One (case, seed) per
// process via SCAN_CASE / SCAN_SEED so a crash cannot mask other rows.

#ifdef FORK
#include "controllers/hierarchical_pipeline.h"
#include "controllers/incremental_pipeline.h"
#include "estimators/alignment.h"
#include "math/random.h"
#include "scene/database.h"
#include "scene/reconstruction_matchers.h"
#include "scene/synthetic.h"
#include "util/testing.h"
#define REC(mgr, i) ((mgr)->Get(i))
#define P3XYZ(p) ((p).XYZ())
#else
#include "colmap/controllers/hierarchical_pipeline.h"
#include "colmap/controllers/incremental_pipeline.h"
#include "colmap/estimators/alignment.h"
#include "colmap/math/random.h"
#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"
#define REC(mgr, i) (*(mgr)->Get(i))
#define P3XYZ(p) ((p).xyz)
#endif

#include <cstdio>
#include <cstdlib>
#include <string>

#include <gtest/gtest.h>

namespace colmap {
namespace {

std::string GtDigest(const Reconstruction& gt) {
  double s_xyz = 0, s_c = 0;
  for (const auto& [_, p] : gt.Points3D()) s_xyz += P3XYZ(p).sum();
  for (const auto& [_, im] : gt.Images()) s_c += im.ProjectionCenter().sum();
  char buf[128];
  std::snprintf(buf, sizeof(buf), "%.12g/%.12g/%zu", s_xyz, s_c,
                static_cast<size_t>(gt.ComputeNumObservations()));
  return buf;
}

bool HierEqual(const Reconstruction& gt, const Reconstruction& c) {
  if (c.NumCameras() != gt.NumCameras() || c.NumImages() != gt.NumImages() ||
      c.NumRegImages() != gt.NumRegImages() ||
      c.ComputeNumObservations() < gt.ComputeNumObservations()) {
    return false;
  }
  Sim3d gt_from_c;
  if (!AlignReconstructionsViaProjCenters(c, gt, 0.1, &gt_from_c)) return false;
  const auto errors = ComputeImageAlignmentError(c, gt, gt_from_c);
  if (errors.size() != gt.NumImages()) return false;
  for (const auto& e : errors) {
    if (e.rotation_error_deg >= 1e-2 || e.proj_center_error >= 1e-4) {
      return false;
    }
  }
  return true;
}

template <typename Mgr>
void Report(const std::string& name, unsigned seed, const std::string& digest,
            const Mgr& mgr, int pass) {
  std::printf("[seed-scan] case=%s seed=%u gt=%s models=%zu", name.c_str(),
              seed, digest.c_str(), static_cast<size_t>(mgr->Size()));
  for (size_t i = 0; i < mgr->Size(); ++i) {
    const Reconstruction& r = REC(mgr, i);
    std::printf(" m%zu=[frames=%zu imgs=%zu pts=%zu obs=%zu mre=%.6f]", i,
                static_cast<size_t>(r.NumRegFrames()),
                static_cast<size_t>(r.NumRegImages()),
                static_cast<size_t>(r.NumPoints3D()),
                static_cast<size_t>(r.ComputeNumObservations()),
                r.ComputeMeanReprojectionError());
  }
  std::printf(" pass=%d\n", pass);
  std::fflush(stdout);
}

void RunIncremental(const std::filesystem::path& db_path,
                    const std::shared_ptr<IncrementalPipelineOptions>& options,
                    const std::shared_ptr<ReconstructionManager>& mgr) {
#ifdef FORK
  IncrementalPipeline mapper(options.get(), "", db_path, mgr.get());
  mapper.Start();
  mapper.Wait();
#else
  IncrementalPipeline mapper(options, Database::Open(db_path), mgr);
  mapper.Run();
#endif
}

TEST(PipelineSeedScan, DISABLED_Run) {
  const char* case_env = std::getenv("SCAN_CASE");
  const char* seed_env = std::getenv("SCAN_SEED");
  ASSERT_NE(case_env, nullptr);
  ASSERT_NE(seed_env, nullptr);
  const std::string name = case_env;
  const unsigned seed = static_cast<unsigned>(std::stoul(seed_env));
  SetPRNGSeed(seed);

  const auto db_path = CreateTestDir() / "database.db";
  auto database = Database::Open(db_path);
  auto mgr = std::make_shared<ReconstructionManager>();
  auto options = std::make_shared<IncrementalPipelineOptions>();
  const char* threads_env = std::getenv("SCAN_THREADS");
  const int scan_threads = threads_env ? std::stoi(threads_env) : -1;
  options->num_threads = scan_threads;
  // SCAN_RSEED=1 pins the mapper seed so every RANSAC re-seeds from the
  // options instead of the ambient (thread-dependent) PRNG stream.
  if (std::getenv("SCAN_RSEED")) options->random_seed = static_cast<int>(seed);
  SyntheticDatasetOptions so;
  Reconstruction gt, gt2;

  if (name == "inc_prior_nontrivial") {
    so.num_rigs = 2;
    so.num_cameras_per_rig = 2;
    so.num_frames_per_rig = 7;
    so.num_points3D = 100;
    so.camera_has_prior_focal_length = false;
    so.prior_position = true;
    SynthesizeDataset(so, &gt, database.get());
    FlatHashSet<sensor_t> ref_sensor_ids;
    for (const auto& [_, rig] : gt.Rigs()) ref_sensor_ids.insert(rig.RefSensorId());
    const std::vector<PosePrior> priors = database->ReadAllPosePriors();
    database->ClearPosePriors();
    for (const PosePrior& p : priors) {
      if (ref_sensor_ids.count(p.corr_data_id.sensor_id)) database->WritePosePrior(p);
    }
    database.reset();
    options->use_prior_position = true;
    options->use_robust_loss_on_prior_position = true;
    RunIncremental(db_path, options, mgr);
    const int pass = mgr->Size() == 1 &&
        testing::Matches(ReconstructionNear(REC(mgr, 0), 1e-1, 1e-1,
                                            std::nullopt, 0.02, false))(gt);
    Report(name, seed, GtDigest(gt), mgr, pass);
  } else if (name == "inc_prior_nonoise") {
    so.num_rigs = 2;
    so.num_cameras_per_rig = 1;
    so.num_frames_per_rig = 10;
    so.num_points3D = 100;
    so.prior_position = true;
    SynthesizeDataset(so, &gt, database.get());
    SyntheticNoiseOptions no;
    no.point2D_stddev = 0.5;
    no.prior_position_stddev = 0.0;
    SynthesizeNoise(no, &gt, database.get());
    database.reset();
    options->use_prior_position = true;
    RunIncremental(db_path, options, mgr);
    const int pass = mgr->Size() == 1 &&
        testing::Matches(ReconstructionNear(REC(mgr, 0), 1e-1, 1e-1,
                                            std::nullopt, 0.02, false))(gt);
    Report(name, seed, GtDigest(gt), mgr, pass);
  } else if (name == "inc_nontrivial") {
    so.num_rigs = 2;
    so.num_cameras_per_rig = 2;
    so.num_frames_per_rig = 7;
    so.num_points3D = 100;
    so.camera_has_prior_focal_length = false;
    so.sensor_from_rig_translation_stddev = 0.05;
    so.sensor_from_rig_rotation_stddev = 30;
    SynthesizeDataset(so, &gt, database.get());
    database.reset();
    RunIncremental(db_path, options, mgr);
    const int pass = mgr->Size() == 1 &&
        testing::Matches(ReconstructionNear(REC(mgr, 0), 1e-2, 1e-4))(gt);
    Report(name, seed, GtDigest(gt), mgr, pass);
  } else if (name == "inc_multi") {
    so.num_rigs = 1;
    so.num_cameras_per_rig = 1;
    so.num_frames_per_rig = 5;
    so.num_points3D = 50;
    SynthesizeDataset(so, &gt, database.get());
    so.num_frames_per_rig = 4;
    SynthesizeDataset(so, &gt2, database.get());
    database.reset();
    options->min_model_size = 4;
    RunIncremental(db_path, options, mgr);
    int pass = 0;
    if (mgr->Size() == 2) {
      const bool first5 = REC(mgr, 0).NumRegImages() == 5;
      const Reconstruction& c1 = REC(mgr, first5 ? 0 : 1);
      const Reconstruction& c2 = REC(mgr, first5 ? 1 : 0);
      pass = testing::Matches(ReconstructionNear(c1, 1e-2, 1e-4))(gt) &&
             testing::Matches(ReconstructionNear(c2, 1e-2, 1e-4))(gt2);
    }
    Report(name, seed, GtDigest(gt) + "+" + GtDigest(gt2), mgr, pass);
  } else if (name == "hier_multi") {
    so.num_rigs = 1;
    so.num_cameras_per_rig = 1;
    so.num_frames_per_rig = 5;
    so.num_points3D = 50;
    SynthesizeDataset(so, &gt, database.get());
    so.num_frames_per_rig = 4;
    SynthesizeDataset(so, &gt2, database.get());
    database.reset();
    HierarchicalPipelineOptions ho;
    ho.clustering_options.leaf_max_num_images = 5;
    ho.clustering_options.image_overlap = 3;
    ho.num_threads = scan_threads;
    if (std::getenv("SCAN_RSEED")) ho.incremental_options.random_seed = static_cast<int>(seed);
#ifdef FORK
    HierarchicalPipeline mapper(ho, db_path, mgr.get());
    mapper.Start();
    mapper.Wait();
#else
    HierarchicalPipeline mapper(ho, Database::Open(db_path), mgr);
    mapper.Run();
#endif
    int pass = 0;
    if (mgr->Size() == 2) {
      const bool first5 = REC(mgr, 0).NumRegImages() == 5;
      pass = HierEqual(gt, REC(mgr, first5 ? 0 : 1)) &&
             HierEqual(gt2, REC(mgr, first5 ? 1 : 0));
    }
    Report(name, seed, GtDigest(gt) + "+" + GtDigest(gt2), mgr, pass);
  } else {
    FAIL() << "unknown case " << name;
  }
}

}  // namespace
}  // namespace colmap

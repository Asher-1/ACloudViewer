// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Cross-catalog default-row contract test — fast, no GGUF assets required.
//
// Every AICore plugin dialog selects its model-combo default through
// ecvAICoreUi::selectModelRow, which prefers the index the task's catalog
// declares via aicore_<task>_model_default_index(). These tests lock the
// invariant that makes that mechanism trustworthy:
//
//   1. the declared default row is inside the catalog and carries the
//      visible "(recommended)" marker (declaration can never drift from
//      what the user sees), and
//   2. catalogs without a default query (SAM3, FaceDetect) keep their
//      long-standing "first row of every family is the recommended pack"
//      property, which the UI relies on via the guard fallback.
//

#include <gtest/gtest.h>

#include <cstring>

#include "aicore/facedetect_capi.h"
#include "aicore/gkd_capi.h"
#include "aicore/rfdetr_capi.h"
#include "aicore/rmbg_capi.h"
#include "aicore/sam3_capi.h"
#include "aicore/yolo_capi.h"

static int noteHasRecommended(const char* note) {
    return note && std::strstr(note, "(recommended)") != nullptr;
}

TEST(ModelCatalogDefaults, RmbgDefaultIsRecommended) {
    const int n = aicore_rmbg_model_count();
    ASSERT_TRUE(n > 0);
    const int d = aicore_rmbg_model_default_index();
    ASSERT_TRUE(d >= 0 && d < n);
    const aicore_rmbg_model_entry* e = aicore_rmbg_model_at(d);
    ASSERT_TRUE(e && e->quant_note);
    ASSERT_TRUE(noteHasRecommended(e->quant_note));
    ASSERT_TRUE(std::strcmp(e->filename, "rmbg_f16.gguf") == 0);
}

TEST(ModelCatalogDefaults, GkdDefaultIsRecommended) {
    const int n = aicore_gkd_model_count();
    ASSERT_TRUE(n > 0);
    const int d = aicore_gkd_model_default_index();
    ASSERT_TRUE(d >= 0 && d < n);
    const aicore_gkd_model_entry* e = aicore_gkd_model_at(d);
    ASSERT_TRUE(e && e->quant_note);
    ASSERT_TRUE(noteHasRecommended(e->quant_note));
    ASSERT_TRUE(std::strcmp(e->filename, "gkd_fullset-q4_K.gguf") == 0);
}

TEST(ModelCatalogDefaults, RfDetrDefaultIsRecommended) {
    const int n = aicore_rfdetr_model_count();
    ASSERT_TRUE(n > 0);
    const int d = aicore_rfdetr_model_default_index();
    ASSERT_TRUE(d >= 0 && d < n);
    const aicore_rfdetr_model_entry* e = aicore_rfdetr_model_at(d);
    ASSERT_TRUE(e && e->quant_note);
    ASSERT_TRUE(noteHasRecommended(e->quant_note));
    ASSERT_TRUE(std::strcmp(e->filename, "rfdetr-nano-f16.gguf") == 0);
}

TEST(ModelCatalogDefaults, YoloRoleDefaultsAreRecommended) {
    const aicore_yolo_model_role roles[] = {
            AICORE_YOLO_ROLE_ANY,      AICORE_YOLO_ROLE_DETECTION,
            AICORE_YOLO_ROLE_DEPTH,    AICORE_YOLO_ROLE_SEGMENT,
            AICORE_YOLO_ROLE_POSE,     AICORE_YOLO_ROLE_OBB,
            AICORE_YOLO_ROLE_CLASSIFY, AICORE_YOLO_ROLE_SEMANTIC,
            AICORE_YOLO_ROLE_WORLD,    AICORE_YOLO_ROLE_YOLOE,
            AICORE_YOLO_ROLE_TEXT,
    };
    for (const aicore_yolo_model_role role : roles) {
        const int n = aicore_yolo_model_count(role);
        if (n <= 0) continue;  // empty views have no default
        const int d = aicore_yolo_model_default_index(role);
        ASSERT_TRUE(d >= 0 && d < n);
        const aicore_yolo_model_entry* e = aicore_yolo_model_at(d, role);
        ASSERT_TRUE(e && e->quant_note);
        ASSERT_TRUE(noteHasRecommended(e->quant_note));
    }
}

TEST(ModelCatalogDefaults, Sam3FamiliesStartWithRecommendedPack) {
    const int n = aicore_sam3_model_count();
    ASSERT_TRUE(n > 0);
    const char* firstFamily = nullptr;
    int firstIndex = -1;
    for (int i = 0; i <= n; ++i) {
        const aicore_sam3_model_entry* e =
                i < n ? aicore_sam3_model_at(i) : nullptr;
        const char* family = e ? e->model_family : nullptr;
        if (firstFamily &&
            (family == nullptr || std::strcmp(family, firstFamily) != 0)) {
            // Family run ended: its first row must be the marked one.
            const aicore_sam3_model_entry* f = aicore_sam3_model_at(firstIndex);
            ASSERT_TRUE(f && f->quant_note);
            ASSERT_TRUE(noteHasRecommended(f->quant_note));
        }
        if (family &&
            (firstFamily == nullptr || std::strcmp(family, firstFamily) != 0)) {
            firstFamily = family;
            firstIndex = i;
        }
    }
}

TEST(ModelCatalogDefaults, FaceDetectorStartsWithRecommendedPack) {
    const int n = aicore_facedetect_detector_model_count();
    ASSERT_TRUE(n > 0);
    const aicore_facedetect_model_entry* e =
            aicore_facedetect_detector_model_at(0);
    ASSERT_TRUE(e && e->display_name);
    ASSERT_TRUE(noteHasRecommended(e->display_name));
}

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Unit test of the `table` reconstruction post-process (no model, no
// backend): structure decode, span grid, OCR reading order and cell
// assignment, HTML (PaddleX get_html_result, plus the optional
// stacked-line-aware cell join) and RFC 4180 CSV.

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "tasks/table/table_postprocess.hpp"

using namespace aicore::table;

namespace {

const std::vector<std::string> kSpanTokens = {
        "<tr>",      "<td",   " colspan=\"2\"", ">",
        "</td>",     "<td",   " rowspan=\"2\"", ">",
        "</td>",     "</tr>", "<tr>",           "<td></td>",
        "<td></td>", "</tr>"};

TEST(TablePostprocess, BuildsSpanGridAndClipsRowSpans) {
    const TableGrid grid = buildGrid(kSpanTokens);
    EXPECT_EQ(grid.rows, 2);
    EXPECT_EQ(grid.cols, 3);
    ASSERT_EQ(grid.cells.size(), 4u);
    EXPECT_TRUE(grid.cells[0].row == 0 && grid.cells[0].col == 0);
    EXPECT_TRUE(grid.cells[0].colSpan == 2 && grid.cells[0].rowSpan == 1);
    EXPECT_TRUE(grid.cells[1].row == 0 && grid.cells[1].col == 2);
    EXPECT_EQ(grid.cells[1].rowSpan, 2);
    // Row 1 skips the slot reserved by the row span above.
    EXPECT_TRUE(grid.cells[2].row == 1 && grid.cells[2].col == 0);
    EXPECT_TRUE(grid.cells[3].row == 1 && grid.cells[3].col == 1);
    // A row span past the last <tr> is clipped.
    const TableGrid clipped =
            buildGrid({"<tr>", "<td", " rowspan=\"5\"", ">", "</td>", "</tr>"});
    EXPECT_EQ(clipped.rows, 1);
    ASSERT_EQ(clipped.cells.size(), 1u);
    EXPECT_EQ(clipped.cells[0].rowSpan, 1);
}

TEST(TablePostprocess, ExportsRfc4180Csv) {
    const TableGrid grid = buildGrid(kSpanTokens);
    EXPECT_EQ(buildCsv(grid, {"a", "b", "c", "d"}), "a,,b\r\nc,d,\r\n");
    EXPECT_EQ(buildCsv(grid, {"x,y", "say \"hi\"", "", "line\nbreak"}),
              "\"x,y\",,\"say \"\"hi\"\"\"\r\n,\"line\nbreak\",\r\n");
    EXPECT_TRUE(buildCsv(TableGrid{}, {}).empty());
}

TEST(TablePostprocess, BuildsEscapedHtmlAndCellText) {
    // cell 0: two boxes joined by a space; cell 1: escaped text;
    // cells 2/3: empty.
    std::vector<std::vector<int>> matched = {{0, 1}, {2}, {}, {}};
    std::vector<std::string> texts = {"Total", " 12", "a<b & c"};
    std::vector<std::string> cellTexts;
    const std::string html = buildHtml(kSpanTokens, matched, texts, &cellTexts);
    EXPECT_EQ(html,
              "<html><body><table><tr><td colspan=\"2\">Total 12</td>"
              "<td rowspan=\"2\">a&lt;b &amp; c</td></tr><tr><td></td>"
              "<td></td></tr></table></body></html>");
    EXPECT_EQ(cellTexts.size(), 4);
    if (cellTexts.size() == 4) {
        EXPECT_EQ(cellTexts[0], "Total 12");
        EXPECT_EQ(cellTexts[1], "a<b & c");
        EXPECT_TRUE(cellTexts[2].empty());
    }
    // A single box keeps its leading space (get_html_result only strips
    // when several boxes share a cell).
    std::vector<std::string> single;
    buildHtml({"<tr>", "<td></td>", "</tr>"}, {{0}}, {" x"}, &single);
    ASSERT_EQ(single.size(), 1u);
    EXPECT_EQ(single[0], " x");
}

TEST(TablePostprocess, MatchesOcrByOverlapDistanceAndFirstTie) {
    const std::vector<std::array<float, 4>> cells = {{0, 0, 100, 50},
                                                     {100, 0, 200, 50}};
    // Inside cell 1; straddling (IoU decides); outside both (distance).
    const std::vector<std::array<float, 4>> ocr = {
            {120, 10, 180, 40}, {60, 10, 160, 40}, {400, 0, 450, 50}};
    const auto matched = matchOcrToCells(cells, ocr);
    ASSERT_EQ(matched.size(), 2u);
    EXPECT_EQ(matched[1].size(), 3);
    EXPECT_TRUE(matched[0].empty());
    // Equal IoU and distance: the first cell wins.
    const auto tie =
            matchOcrToCells({{0, 0, 10, 10}, {0, 0, 10, 10}}, {{0, 0, 10, 10}});
    ASSERT_EQ(tie.size(), 2u);
    EXPECT_TRUE(tie[0].size() == 1 && tie[1].empty());
}

TEST(TablePostprocess, JoinsStackedLinesWithOptionalGeometry) {
    // zh_val_22.jpg case: a wrapped email is two det boxes in one cell
    // ('gftiec@gftiec.c' over 'om'); the bilingual City label is two tight
    // stacked lines; unrelated rows keep a clear vertical gap.
    const std::vector<std::array<float, 4>> boxes = {
            {1410, 1812, 1723, 1864},  // 'gftiec@gftiec.c'
            {1410, 1871, 1487, 1915},  // 'om' (7 px below, x overlap)
            {229, 892, 314, 946},      // 'City'
            {232, 943, 321, 998},    // City in Chinese (overlaps City's bottom)
            {232, 1038, 432, 1089},  // 'Postal Code' (40 px below)
    };
    EXPECT_TRUE(stackedLineJoin(boxes[0], boxes[1]));
    EXPECT_TRUE(stackedLineJoin(boxes[2], boxes[3]));
    EXPECT_FALSE(stackedLineJoin(boxes[3], boxes[4]));
    // Same-line neighbours are not "stacked".
    EXPECT_FALSE(stackedLineJoin({0, 0, 100, 50}, {110, 2, 200, 48}));
    // Degenerate boxes never glue.
    EXPECT_FALSE(stackedLineJoin({0, 0, 0, 0}, {0, 0, 10, 10}));

    const std::vector<std::vector<int>> matched = {{0, 1}, {2, 3, 4}};
    const std::vector<std::string> texts = {"gftiec@gftiec.c", "om", "City",
                                            "\u57ce\u5e02", "Postal Code"};
    std::vector<std::string> cellTexts;
    // Mode 0 (upstream get_html_result): always space-joined.
    buildHtml({"<tr>", "<td></td>", "<td></td>", "</tr>"}, matched, texts,
              &cellTexts, &boxes, false);
    EXPECT_EQ(cellTexts.size(), 2);
    if (cellTexts.size() == 2) {
        EXPECT_EQ(cellTexts[0], "gftiec@gftiec.c om");
        EXPECT_EQ(cellTexts[1], "City \u57ce\u5e02 Postal Code");
    }
    // Mode 1 (stacked-line-aware): wrapped lines lose the separator.
    buildHtml({"<tr>", "<td></td>", "<td></td>", "</tr>"}, matched, texts,
              &cellTexts, &boxes, true);
    EXPECT_EQ(cellTexts.size(), 2);
    if (cellTexts.size() == 2) {
        EXPECT_EQ(cellTexts[0], "gftiec@gftiec.com");
        EXPECT_EQ(cellTexts[1], "City\u57ce\u5e02 Postal Code");
    }
    // Without box geometry the stacked mode degrades to the upstream join.
    buildHtml({"<tr>", "<td></td>", "</tr>"}, {{0, 1}}, texts, &cellTexts,
              nullptr, true);
    ASSERT_EQ(cellTexts.size(), 1u);
    EXPECT_EQ(cellTexts[0], "gftiec@gftiec.c om");
}

TEST(TablePostprocess, SortsOcrInPaddleReadingOrder) {
    // (100, 10) and (0, 12) share a 10 px line -> x order; (0, 30) is the
    // next line.
    const auto order = paddleReadingOrder({{100, 10}, {0, 30}, {0, 12}});
    ASSERT_EQ(order.size(), 3u);
    EXPECT_EQ(order[0], 2);
    EXPECT_EQ(order[1], 0);
    EXPECT_EQ(order[2], 1);
}

TEST(TablePostprocess, DecodesStructureAndExcludesEosScore) {
    const std::vector<std::string> vocab = {"sos", "<tr>", "<td></td>", "</tr>",
                                            "eos"};
    TableStructureOutput out;
    out.tokens = {1, 2, 3, 4};
    out.scores = {1.0f, 0.5f, 0.75f, 0.1f};
    out.loc.assign(4 * 8, 0.0f);
    for (int k = 0; k < 8; ++k) out.loc[8 + k] = 0.2501f;
    // 976 x 488 crop -> ratio 0.5 -> polygon scale 976.
    const DecodedTable t = decodeStructure(out, vocab, 8, 488, 976, 488);
    EXPECT_EQ(t.tokens.size(), 3);
    ASSERT_EQ(t.cells.size(), 1u);
    EXPECT_EQ(t.cells[0][0], 244.0f);              // trunc(0.2501 * 976)
    EXPECT_LT(std::fabs(t.score - 0.75f), 1e-6f);  // eos excluded
    // A leading eos is skipped (reference: idx > 0 breaks), sos ignored.
    out.tokens = {4, 0, 2, 4};
    const DecodedTable lead = decodeStructure(out, vocab, 8, 488, 976, 488);
    ASSERT_EQ(lead.tokens.size(), 1u);
    EXPECT_EQ(lead.tokens[0], "<td></td>");
}

}  // namespace

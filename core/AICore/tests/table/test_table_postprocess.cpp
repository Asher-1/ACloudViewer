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

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "tasks/table/table_postprocess.hpp"
#include "tests/common/test_macros.hpp"

static int failures = 0;

using namespace aicore::table;

namespace {

const std::vector<std::string> kSpanTokens = {
        "<tr>",      "<td",   " colspan=\"2\"", ">",
        "</td>",     "<td",   " rowspan=\"2\"", ">",
        "</td>",     "</tr>", "<tr>",           "<td></td>",
        "<td></td>", "</tr>"};

void testGrid() {
    const TableGrid grid = buildGrid(kSpanTokens);
    AICORE_CHECK(grid.rows == 2);
    AICORE_CHECK(grid.cols == 3);
    AICORE_CHECK(grid.cells.size() == 4);
    if (grid.cells.size() != 4) return;
    AICORE_CHECK(grid.cells[0].row == 0 && grid.cells[0].col == 0);
    AICORE_CHECK(grid.cells[0].colSpan == 2 && grid.cells[0].rowSpan == 1);
    AICORE_CHECK(grid.cells[1].row == 0 && grid.cells[1].col == 2);
    AICORE_CHECK(grid.cells[1].rowSpan == 2);
    // Row 1 skips the slot reserved by the row span above.
    AICORE_CHECK(grid.cells[2].row == 1 && grid.cells[2].col == 0);
    AICORE_CHECK(grid.cells[3].row == 1 && grid.cells[3].col == 1);
    // A row span past the last <tr> is clipped.
    const TableGrid clipped =
            buildGrid({"<tr>", "<td", " rowspan=\"5\"", ">", "</td>", "</tr>"});
    AICORE_CHECK(clipped.rows == 1 && clipped.cells.size() == 1 &&
                 clipped.cells[0].rowSpan == 1);
}

void testCsv() {
    const TableGrid grid = buildGrid(kSpanTokens);
    AICORE_CHECK(buildCsv(grid, {"a", "b", "c", "d"}) == "a,,b\r\nc,d,\r\n");
    AICORE_CHECK(buildCsv(grid, {"x,y", "say \"hi\"", "", "line\nbreak"}) ==
                 "\"x,y\",,\"say \"\"hi\"\"\"\r\n,\"line\nbreak\",\r\n");
    AICORE_CHECK(buildCsv(TableGrid{}, {}).empty());
}

void testHtml() {
    // cell 0: two boxes joined by a space; cell 1: escaped text;
    // cells 2/3: empty.
    std::vector<std::vector<int>> matched = {{0, 1}, {2}, {}, {}};
    std::vector<std::string> texts = {"Total", " 12", "a<b & c"};
    std::vector<std::string> cellTexts;
    const std::string html = buildHtml(kSpanTokens, matched, texts, &cellTexts);
    AICORE_CHECK(html ==
                 "<html><body><table><tr><td colspan=\"2\">Total 12</td>"
                 "<td rowspan=\"2\">a&lt;b &amp; c</td></tr><tr><td></td>"
                 "<td></td></tr></table></body></html>");
    AICORE_CHECK(cellTexts.size() == 4);
    if (cellTexts.size() == 4) {
        AICORE_CHECK(cellTexts[0] == "Total 12");
        AICORE_CHECK(cellTexts[1] == "a<b & c");
        AICORE_CHECK(cellTexts[2].empty());
    }
    // A single box keeps its leading space (get_html_result only strips
    // when several boxes share a cell).
    std::vector<std::string> single;
    buildHtml({"<tr>", "<td></td>", "</tr>"}, {{0}}, {" x"}, &single);
    AICORE_CHECK(single.size() == 1 && single[0] == " x");
}

void testMatching() {
    const std::vector<std::array<float, 4>> cells = {{0, 0, 100, 50},
                                                     {100, 0, 200, 50}};
    // Inside cell 1; straddling (IoU decides); outside both (distance).
    const std::vector<std::array<float, 4>> ocr = {
            {120, 10, 180, 40}, {60, 10, 160, 40}, {400, 0, 450, 50}};
    const auto matched = matchOcrToCells(cells, ocr);
    AICORE_CHECK(matched.size() == 2);
    if (matched.size() != 2) return;
    AICORE_CHECK(matched[1].size() == 3);
    AICORE_CHECK(matched[0].empty());
    // Equal IoU and distance: the first cell wins.
    const auto tie =
            matchOcrToCells({{0, 0, 10, 10}, {0, 0, 10, 10}}, {{0, 0, 10, 10}});
    AICORE_CHECK(tie[0].size() == 1 && tie[1].empty());
}

void testStackedJoin() {
    // zh_val_22.jpg case: a wrapped email is two det boxes in one cell
    // ('gftiec@gftiec.c' over 'om'); the bilingual City label is two tight
    // stacked lines; unrelated rows keep a clear vertical gap.
    const std::vector<std::array<float, 4>> boxes = {
            {1410, 1812, 1723, 1864},  // 'gftiec@gftiec.c'
            {1410, 1871, 1487, 1915},  // 'om' (7 px below, x overlap)
            {229, 892, 314, 946},      // 'City'
            {232, 943, 321, 998},      // '城市' (overlaps City's bottom)
            {232, 1038, 432, 1089},    // 'Postal Code' (40 px below)
    };
    AICORE_CHECK(stackedLineJoin(boxes[0], boxes[1]));
    AICORE_CHECK(stackedLineJoin(boxes[2], boxes[3]));
    AICORE_CHECK(!stackedLineJoin(boxes[3], boxes[4]));
    // Same-line neighbours are not "stacked".
    AICORE_CHECK(!stackedLineJoin({0, 0, 100, 50}, {110, 2, 200, 48}));
    // Degenerate boxes never glue.
    AICORE_CHECK(!stackedLineJoin({0, 0, 0, 0}, {0, 0, 10, 10}));

    const std::vector<std::vector<int>> matched = {{0, 1}, {2, 3, 4}};
    const std::vector<std::string> texts = {"gftiec@gftiec.c", "om", "City",
                                            "城市", "Postal Code"};
    std::vector<std::string> cellTexts;
    // Mode 0 (upstream get_html_result): always space-joined.
    buildHtml({"<tr>", "<td></td>", "<td></td>", "</tr>"}, matched, texts,
              &cellTexts, &boxes, false);
    AICORE_CHECK(cellTexts.size() == 2);
    if (cellTexts.size() == 2) {
        AICORE_CHECK(cellTexts[0] == "gftiec@gftiec.c om");
        AICORE_CHECK(cellTexts[1] == "City 城市 Postal Code");
    }
    // Mode 1 (stacked-line-aware): wrapped lines lose the separator.
    buildHtml({"<tr>", "<td></td>", "<td></td>", "</tr>"}, matched, texts,
              &cellTexts, &boxes, true);
    AICORE_CHECK(cellTexts.size() == 2);
    if (cellTexts.size() == 2) {
        AICORE_CHECK(cellTexts[0] == "gftiec@gftiec.com");
        AICORE_CHECK(cellTexts[1] == "City城市 Postal Code");
    }
    // Without box geometry the stacked mode degrades to the upstream join.
    buildHtml({"<tr>", "<td></td>", "</tr>"}, {{0, 1}}, texts, &cellTexts,
              nullptr, true);
    AICORE_CHECK(cellTexts.size() == 1 && cellTexts[0] == "gftiec@gftiec.c om");
}

void testReadingOrder() {
    // (100, 10) and (0, 12) share a 10 px line -> x order; (0, 30) is the
    // next line.
    const auto order = paddleReadingOrder({{100, 10}, {0, 30}, {0, 12}});
    AICORE_CHECK(order.size() == 3);
    if (order.size() == 3)
        AICORE_CHECK(order[0] == 2 && order[1] == 0 && order[2] == 1);
}

void testDecode() {
    const std::vector<std::string> vocab = {"sos", "<tr>", "<td></td>", "</tr>",
                                            "eos"};
    TableStructureOutput out;
    out.tokens = {1, 2, 3, 4};
    out.scores = {1.0f, 0.5f, 0.75f, 0.1f};
    out.loc.assign(4 * 8, 0.0f);
    for (int k = 0; k < 8; ++k) out.loc[8 + k] = 0.2501f;
    // 976 x 488 crop -> ratio 0.5 -> polygon scale 976.
    const DecodedTable t = decodeStructure(out, vocab, 8, 488, 976, 488);
    AICORE_CHECK(t.tokens.size() == 3);
    AICORE_CHECK(t.cells.size() == 1);
    if (!t.cells.empty())
        AICORE_CHECK(t.cells[0][0] == 244.0f);         // trunc(0.2501 * 976)
    AICORE_CHECK(std::fabs(t.score - 0.75f) < 1e-6f);  // eos excluded
    // A leading eos is skipped (reference: idx > 0 breaks), sos ignored.
    out.tokens = {4, 0, 2, 4};
    const DecodedTable lead = decodeStructure(out, vocab, 8, 488, 976, 488);
    AICORE_CHECK(lead.tokens.size() == 1 && lead.tokens[0] == "<td></td>");
}

}  // namespace

int main() {
    testGrid();
    testCsv();
    testHtml();
    testStackedJoin();
    testMatching();
    testReadingOrder();
    testDecode();
    if (failures) {
        std::fprintf(stderr, "test_table_postprocess: %d failure(s)\n",
                     failures);
        return 1;
    }
    std::printf("test_table_postprocess: OK\n");
    return 0;
}

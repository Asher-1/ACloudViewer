// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Table reconstruction post-process (pure functions, no ggml/Qt):
//
//   decoder steps -> structure tokens + one polygon per td token
//                    (PaddleX TableLabelDecode)
//   tokens        -> grid row/col/row_span/col_span per cell
//   OCR boxes     -> cell assignment by (1 - IoU, L1 distance)
//                    (PaddleX match_table_and_ocr)
//   tokens + text -> HTML (PaddleX get_html_result) and RFC 4180 CSV
//
// The Paddle routines are reproduced operation by operation because the
// GT matrix compares the composed HTML against both the dataset truth and
// the official pipeline; any deviation would be a TEDS difference that is
// not the model's.

#include <array>
#include <string>
#include <vector>

#include "table_runtime.hpp"

namespace aicore {
namespace table {

struct DecodedTable {
    /** Structure tokens without the html/body/table wrapper. */
    std::vector<std::string> tokens;
    /** One 8-value polygon (x0,y0,...,x3,y3) per td token, crop pixels,
     *  truncated to integers like the reference (astype(int)). */
    std::vector<std::array<float, 8>> cells;
    /** Mean max-probability of the kept tokens. */
    float score = 0.0f;
};

/** Decode one table crop of `cropW` x `cropH` pixels that was resized by
 *  its long side to `maxLen` and padded to `padSize`. */
DecodedTable decodeStructure(const TableStructureOutput& out,
                             const std::vector<std::string>& vocab,
                             int locReg,
                             int padSize,
                             int cropW,
                             int cropH);

struct GridCell {
    int row = 0;
    int col = 0;
    int rowSpan = 1;
    int colSpan = 1;
};

struct TableGrid {
    int rows = 0;
    int cols = 0;
    /** One entry per td token, in token order (= DecodedTable::cells). */
    std::vector<GridCell> cells;
};

/** Slot every td of the token stream into the row/column lattice with the
 *  HTML table model: a cell takes the first free slot of its row, and its
 *  row/col span reserves the slots below/right of it. O(tokens + slots). */
TableGrid buildGrid(const std::vector<std::string>& tokens);

/** PaddleX sorted_boxes reading order of OCR quads (top-left corner sorted
 *  by (y, x), then neighbours within 10 px vertically swapped into x
 *  order). Returns the permutation. */
std::vector<int> paddleReadingOrder(const std::vector<std::array<float, 2>>& topLeft);

/** For every OCR box (xyxy), the index of its cell (xyxy): minimum of
 *  (1 - IoU, distance) lexicographically, first index on ties. Returns
 *  per-cell OCR index lists in OCR order. */
std::vector<std::vector<int>> matchOcrToCells(
        const std::vector<std::array<float, 4>>& cellBoxes,
        const std::vector<std::array<float, 4>>& ocrBoxes);

/** Compose the table HTML from the tokens and the matched OCR texts, and
 *  return each cell's assembled text in `cellTexts` (td order). Text is
 *  HTML-escaped in the markup and raw in `cellTexts`. */
std::string buildHtml(const std::vector<std::string>& tokens,
                      const std::vector<std::vector<int>>& matched,
                      const std::vector<std::string>& ocrTexts,
                      std::vector<std::string>* cellTexts);

/** RFC 4180 CSV (CRLF rows) of the grid; a spanned cell's text sits in
 *  its top-left slot and the covered slots are empty. */
std::string buildCsv(const TableGrid& grid,
                     const std::vector<std::string>& cellTexts);

}  // namespace table
}  // namespace aicore

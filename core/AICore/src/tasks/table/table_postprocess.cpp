// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "table_postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>

namespace aicore {
namespace table {

namespace {

bool isTdToken(const std::string& t) {
    return t == "<td>" || t == "<td" || t == "<td></td>";
}

int spanValue(const std::string& token, const char* attr) {
    // ` colspan="3"` -> 3; anything malformed counts as 1.
    const auto pos = token.find(attr);
    if (pos == std::string::npos) return 0;
    const auto quote = token.find('"', pos);
    if (quote == std::string::npos) return 1;
    const int v = std::atoi(token.c_str() + quote + 1);
    return v >= 1 ? v : 1;
}

void appendEscaped(std::string& out, const std::string& text) {
    for (char c : text) {
        switch (c) {
            case '&':
                out += "&amp;";
                break;
            case '<':
                out += "&lt;";
                break;
            case '>':
                out += "&gt;";
                break;
            default:
                out += c;
        }
    }
}

double iou(const std::array<float, 4>& a, const std::array<float, 4>& b) {
    const double areaA = (static_cast<double>(a[2]) - a[0]) * (a[3] - a[1]);
    const double areaB = (static_cast<double>(b[2]) - b[0]) * (b[3] - b[1]);
    const double left = std::max(a[0], b[0]);
    const double right = std::min(a[2], b[2]);
    const double top = std::max(a[1], b[1]);
    const double bottom = std::min(a[3], b[3]);
    if (left >= right || top >= bottom) return 0.0;
    const double inter = (right - left) * (bottom - top);
    return inter / (areaA + areaB - inter);
}

double l1Distance(const std::array<float, 4>& a,
                  const std::array<float, 4>& b) {
    const double d1 = std::fabs(static_cast<double>(b[0]) - a[0]) +
                      std::fabs(static_cast<double>(b[1]) - a[1]);
    const double d2 = std::fabs(static_cast<double>(b[2]) - a[2]) +
                      std::fabs(static_cast<double>(b[3]) - a[3]);
    return d1 + d2 + std::min(d1, d2);
}

}  // namespace

DecodedTable decodeStructure(const TableStructureOutput& out,
                             const std::vector<std::string>& vocab,
                             int locReg,
                             int padSize,
                             int cropW,
                             int cropH) {
    DecodedTable table;
    if (vocab.size() < 2 || cropW <= 0 || cropH <= 0 || locReg != 8)
        return table;
    const int sos = 0;
    const int eos = static_cast<int>(vocab.size()) - 1;
    // TableLabelDecode._get_bbox_scales: polygons are in the padded input
    // space; x and y both scale by pad / ratio with ratio = min(pad / w,
    // pad / h), i.e. the inverse of the long-side resize.
    const double ratio = std::min(static_cast<double>(padSize) / cropW,
                                  static_cast<double>(padSize) / cropH);
    const double scale = padSize / ratio;
    double scoreSum = 0.0;
    for (size_t s = 0; s < out.tokens.size(); ++s) {
        const int id = out.tokens[s];
        if (s > 0 && id == eos) break;
        if (id == sos || id == eos || id < 0 ||
            id >= static_cast<int>(vocab.size()))
            continue;
        const std::string& text = vocab[static_cast<size_t>(id)];
        if (isTdToken(text)) {
            std::array<float, 8> poly{};
            for (int k = 0; k < 8; ++k) {
                const double v =
                        out.loc[s * 8 + static_cast<size_t>(k)] * scale;
                // numpy astype(int) truncates toward zero.
                poly[static_cast<size_t>(k)] =
                        static_cast<float>(std::trunc(v));
            }
            table.cells.push_back(poly);
        }
        table.tokens.push_back(text);
        scoreSum += out.scores[s];
    }
    if (!table.tokens.empty())
        table.score = static_cast<float>(scoreSum / table.tokens.size());
    return table;
}

TableGrid buildGrid(const std::vector<std::string>& tokens) {
    TableGrid grid;
    std::vector<std::vector<char>> occupied;  // [row][col]
    const auto taken = [&](int r, int c) {
        return r < static_cast<int>(occupied.size()) &&
               c < static_cast<int>(occupied[static_cast<size_t>(r)].size()) &&
               occupied[static_cast<size_t>(r)][static_cast<size_t>(c)];
    };
    const auto mark = [&](int r, int c) {
        if (r >= static_cast<int>(occupied.size()))
            occupied.resize(static_cast<size_t>(r) + 1);
        auto& line = occupied[static_cast<size_t>(r)];
        if (c >= static_cast<int>(line.size()))
            line.resize(static_cast<size_t>(c) + 1, 0);
        line[static_cast<size_t>(c)] = 1;
    };
    int row = -1;
    int col = 0;
    for (size_t i = 0; i < tokens.size(); ++i) {
        const std::string& t = tokens[i];
        if (t == "<tr>") {
            ++row;
            col = 0;
            continue;
        }
        if (!isTdToken(t)) continue;
        if (row < 0) row = 0;  // a td before any <tr> opens the first row
        GridCell cell;
        if (t == "<td") {
            // Attribute tokens up to the closing '>'.
            for (size_t j = i + 1; j < tokens.size() && tokens[j] != ">"; ++j) {
                if (int v = spanValue(tokens[j], "colspan")) cell.colSpan = v;
                if (int v = spanValue(tokens[j], "rowspan")) cell.rowSpan = v;
            }
        }
        while (taken(row, col)) ++col;
        cell.row = row;
        cell.col = col;
        for (int r = 0; r < cell.rowSpan; ++r)
            for (int c = 0; c < cell.colSpan; ++c) mark(row + r, col + c);
        col += cell.colSpan;
        grid.cols = std::max(grid.cols, col);
        grid.cells.push_back(cell);
    }
    // A row span past the last <tr> is clipped by the HTML table model.
    grid.rows = row + 1;
    for (GridCell& cell : grid.cells)
        cell.rowSpan =
                std::max(1, std::min(cell.rowSpan, grid.rows - cell.row));
    return grid;
}

std::vector<int> paddleReadingOrder(
        const std::vector<std::array<float, 2>>& topLeft) {
    std::vector<int> order(topLeft.size());
    std::iota(order.begin(), order.end(), 0);
    // sorted(key=(y, x)) is stable.
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        const auto& pa = topLeft[static_cast<size_t>(a)];
        const auto& pb = topLeft[static_cast<size_t>(b)];
        if (pa[1] != pb[1]) return pa[1] < pb[1];
        return pa[0] < pb[0];
    });
    // The reference bubble pass: O(n * band) — each box only moves left
    // past neighbours on the same 10 px line.
    for (size_t i = 0; i + 1 < order.size(); ++i) {
        for (size_t j = i + 1; j-- > 0;) {
            const auto& next = topLeft[static_cast<size_t>(order[j + 1])];
            const auto& cur = topLeft[static_cast<size_t>(order[j])];
            if (std::fabs(next[1] - cur[1]) < 10.0f && next[0] < cur[0])
                std::swap(order[j], order[j + 1]);
            else
                break;
        }
    }
    return order;
}

std::vector<std::vector<int>> matchOcrToCells(
        const std::vector<std::array<float, 4>>& cellBoxes,
        const std::vector<std::array<float, 4>>& ocrBoxes) {
    std::vector<std::vector<int>> matched(cellBoxes.size());
    if (cellBoxes.empty()) return matched;
    // O(ocr * cells): a table has at most a few hundred of each.
    for (size_t i = 0; i < ocrBoxes.size(); ++i) {
        size_t best = 0;
        double bestIou = 2.0, bestDist = 0.0;
        for (size_t j = 0; j < cellBoxes.size(); ++j) {
            const double cost = 1.0 - iou(cellBoxes[j], ocrBoxes[i]);
            const double dist = l1Distance(cellBoxes[j], ocrBoxes[i]);
            if (cost < bestIou || (cost == bestIou && dist < bestDist)) {
                best = j;
                bestIou = cost;
                bestDist = dist;
            }
        }
        matched[best].push_back(static_cast<int>(i));
    }
    return matched;
}

std::string buildHtml(const std::vector<std::string>& tokens,
                      const std::vector<std::vector<int>>& matched,
                      const std::vector<std::string>& ocrTexts,
                      std::vector<std::string>* cellTexts) {
    std::string html = "<html><body><table>";
    if (cellTexts) cellTexts->clear();
    size_t td = 0;
    for (const std::string& tag : tokens) {
        if (tag.find("</td>") == std::string::npos) {
            html += tag;
            continue;
        }
        const bool merged = tag == "<td></td>";
        if (merged) html += "<td>";
        std::string text;
        const std::vector<int>* boxes =
                td < matched.size() && !matched[td].empty() ? &matched[td]
                                                            : nullptr;
        bool bold = false;
        if (boxes) {
            // get_html_result: several boxes in one cell are stripped of a
            // leading space and <b> markers, then space-joined; a leading
            // <b> on the first box makes the whole cell bold.
            const size_t n = boxes->size();
            bold = n > 1 && ocrTexts[static_cast<size_t>((*boxes)[0])].find(
                                    "<b>") != std::string::npos;
            for (size_t k = 0; k < n; ++k) {
                std::string content =
                        ocrTexts[static_cast<size_t>((*boxes)[k])];
                if (n > 1) {
                    if (content.empty()) continue;
                    if (content[0] == ' ') content.erase(0, 1);
                    if (content.find("<b>") != std::string::npos)
                        content.erase(0, std::min<size_t>(3, content.size()));
                    if (content.find("</b>") != std::string::npos)
                        content.resize(content.size() >= 4 ? content.size() - 4
                                                           : 0);
                    if (content.empty()) continue;
                    if (k != n - 1 && content.back() != ' ') content += ' ';
                }
                text += content;
            }
        }
        if (bold) html += "<b>";
        appendEscaped(html, text);
        if (bold) html += "</b>";
        html += merged ? std::string("</td>") : tag;
        if (cellTexts) cellTexts->push_back(text);
        ++td;
    }
    html += "</table></body></html>";
    return html;
}

std::string buildCsv(const TableGrid& grid,
                     const std::vector<std::string>& cellTexts) {
    if (grid.rows <= 0 || grid.cols <= 0) return {};
    std::vector<std::string> slots(static_cast<size_t>(grid.rows) * grid.cols);
    for (size_t i = 0; i < grid.cells.size() && i < cellTexts.size(); ++i) {
        const GridCell& c = grid.cells[i];
        if (c.row < grid.rows && c.col < grid.cols)
            slots[static_cast<size_t>(c.row) * grid.cols + c.col] =
                    cellTexts[i];
    }
    std::string csv;
    for (int r = 0; r < grid.rows; ++r) {
        for (int c = 0; c < grid.cols; ++c) {
            if (c) csv += ',';
            const std::string& v =
                    slots[static_cast<size_t>(r) * grid.cols + c];
            if (v.find_first_of(",\"\r\n") == std::string::npos) {
                csv += v;
                continue;
            }
            csv += '"';
            for (char ch : v) {
                if (ch == '"') csv += '"';
                csv += ch;
            }
            csv += '"';
        }
        csv += "\r\n";
    }
    return csv;
}

}  // namespace table
}  // namespace aicore

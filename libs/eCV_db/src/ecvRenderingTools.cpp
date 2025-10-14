// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvRenderingTools.h"

// ECV_DB_LIB
#include "ecvColorScalesManager.h"
#include "ecvDisplayTools.h"
#include "ecvGBLSensor.h"
#include "ecvGenericPointCloud.h"
#include "ecvScalarField.h"

// Qt
#include <QDialog>
#include <QLabel>
#include <QVBoxLayout>

void ccRenderingTools::ShowDepthBuffer(ccGBLSensor* sensor,
                                       QWidget* parent /*=0*/,
                                       unsigned maxDim /*=1024*/) {
    if (!sensor) return;

    const ccDepthBuffer& depthBuffer = sensor->getDepthBuffer();
    if (depthBuffer.zBuff.empty()) {
        return;
    }

    // determine min and max depths
    ScalarType minDist = 0.0f;
    ScalarType maxDist = 0.0f;
    {
        const PointCoordinateType* _zBuff = depthBuffer.zBuff.data();
        double sumDist = 0.0;
        double sumDist2 = 0.0;
        unsigned count = 0;
        for (unsigned x = 0; x < depthBuffer.height * depthBuffer.width;
             ++x, ++_zBuff) {
            if (x == 0) {
                maxDist = minDist = *_zBuff;
            } else if (*_zBuff > 0) {
                maxDist = std::max(maxDist, *_zBuff);
                minDist = std::min(minDist, *_zBuff);
            }

            if (*_zBuff > 0) {
                sumDist += *_zBuff;
                sumDist2 += *_zBuff * *_zBuff;
                ++count;
            }
        }

        if (count) {
            double avg = sumDist / count;
            double stdDev = sqrt(fabs(sumDist2 / count - avg * avg));
            // for better dynamics
            maxDist = std::min(maxDist,
                               static_cast<ScalarType>(avg + 1.0 * stdDev));
        }
    }

    QImage bufferImage(depthBuffer.width, depthBuffer.height,
                       QImage::Format_RGB32);
    {
        ccColorScale::Shared colorScale =
                ccColorScalesManager::GetDefaultScale();
        assert(colorScale);
        ScalarType coef =
                cloudViewer::LessThanEpsilon(maxDist - minDist)
                        ? 0
                        : static_cast<ScalarType>(ccColorScale::MAX_STEPS - 1) /
                                  (maxDist - minDist);

        const PointCoordinateType* _zBuff = depthBuffer.zBuff.data();
        for (unsigned y = 0; y < depthBuffer.height; ++y) {
            for (unsigned x = 0; x < depthBuffer.width; ++x, ++_zBuff) {
                const ecvColor::Rgb& col =
                        (*_zBuff >= minDist
                                 ? colorScale->getColorByIndex(
                                           static_cast<unsigned>(
                                                   (std::min(maxDist, *_zBuff) -
                                                    minDist) *
                                                   coef))
                                 : ecvColor::black);
                bufferImage.setPixel(x, depthBuffer.height - 1 - y,
                                     qRgb(col.r, col.g, col.b));
            }
        }
    }

    QDialog* dlg = new QDialog(parent);
    dlg->setWindowTitle(QString("%0 depth buffer [%1 x %2]")
                                .arg(sensor->getParent()->getName())
                                .arg(depthBuffer.width)
                                .arg(depthBuffer.height));

    unsigned maxDBDim =
            std::max<unsigned>(depthBuffer.width, depthBuffer.height);
    unsigned scale = 1;
    while (maxDBDim > maxDim) {
        maxDBDim >>= 1;
        scale <<= 1;
    }
    dlg->setFixedSize(bufferImage.size() / scale);

    QVBoxLayout* vboxLayout = new QVBoxLayout(dlg);
    vboxLayout->setContentsMargins(0, 0, 0, 0);
    QLabel* label = new QLabel(dlg);
    label->setScaledContents(true);
    vboxLayout->addWidget(label);

    label->setPixmap(QPixmap::fromImage(bufferImage));
    dlg->show();
}

//! Graphical scale atomical element
struct ScaleElement {
    //! Starting value
    double value;
    //! Specifies whether the value should be displayed
    bool textDisplayed;
    //! Specifies whether the cube is condensed or not
    bool condensed;

    //! Default constructor
    ScaleElement(double val, bool dispText = true, bool isCondensed = false)
        : value(val), textDisplayed(dispText), condensed(isCondensed) {}
};

// structure for recursive display of labels
struct vlabel {
    int yPos;   /**< label center pos **/
    int yMin;   /**< label 'ROI' min **/
    int yMax;   /**< label 'ROI' max **/
    double val; /**< label value **/

    // default constructor
    vlabel(int y, int y1, int y2, double v)
        : yPos(y), yMin(y1), yMax(y2), val(v) {
        assert(y2 >= y1);
    }
};

//! A set of 'vlabel' structures
using vlabelSet = std::list<vlabel>;

// helper: returns the neighbouring labels at a given position
//(first: above label, second: below label)
// Warning: set must be already sorted!
using vlabelPair = std::pair<vlabelSet::iterator, vlabelSet::iterator>;

static vlabelPair GetVLabelsAround(int y, vlabelSet& set) {
    if (set.empty()) {
        return vlabelPair(set.end(), set.end());
    } else {
        vlabelSet::iterator it1 = set.begin();
        if (y < it1->yPos) {
            return vlabelPair(set.end(), it1);
        }
        vlabelSet::iterator it2 = it1;
        ++it2;
        for (; it2 != set.end(); ++it2, ++it1) {
            if (y <= it2->yPos)  // '<=' to make sure the last label stays at
                                 // the top!
                return vlabelPair(it1, it2);
        }
        return vlabelPair(it1, set.end());
    }
}

// For log scale inversion
const double c_log10 = log(10.0);

// Convert standard range to log scale
void ConvertToLogScale(ScalarType& dispMin, ScalarType& dispMax) {
    ScalarType absDispMin = (dispMax < 0 ? std::min(-dispMax, -dispMin)
                                         : std::max<ScalarType>(dispMin, 0));
    ScalarType absDispMax = std::max(std::abs(dispMin), std::abs(dispMax));
    dispMin = std::log10(std::max(absDispMin, FLT_EPSILON));
    dispMax = std::log10(std::max(absDispMax, FLT_EPSILON));
}

void ccRenderingTools::DrawColorRamp(const CC_DRAW_CONTEXT& context) {
    const ccScalarField* sf = context.sfColorScaleToDisplay;
    QWidget* display =
            static_cast<QWidget*>(ecvDisplayTools::GetCurrentScreen());

    DrawColorRamp(context, sf, display, context.glW, context.glH,
                  context.renderZoom);
}

void ccRenderingTools::DrawColorRamp(const CC_DRAW_CONTEXT& context,
                                     const ccScalarField* sf,
                                     QWidget* win,
                                     int glW,
                                     int glH,
                                     float renderZoom /*=1.0f*/) {
    WIDGETS_PARAMETER params(WIDGETS_TYPE::WIDGET_SCALAR_BAR, "vtkBlockColors");
    params.context = context;
    params.context.viewID = "vtkBlockColors";

    if (!sf || !sf->getColorScale() || !win) {
        ecvDisplayTools::RemoveWidgets(params);
        return;
    }

    ecvDisplayTools::DrawWidgets(params);
}

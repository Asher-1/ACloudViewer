// ##########################################################################
// #                                                                        #
// #                              CLOUDVIEWER                               #
// #                                                                        #
// #  This program is free software; you can redistribute it and/or modify  #
// #  it under the terms of the GNU General Public License as published by  #
// #  the Free Software Foundation; version 2 or later of the License.      #
// #                                                                        #
// #  This program is distributed in the hope that it will be useful,       #
// #  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
// #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
// #  GNU General Public License for more details.                          #
// #                                                                        #
// #                  COPYRIGHT: Daniel Girardeau-Montaut                   #
// #                                                                        #
// ##########################################################################

#ifndef ECV_FILTER_BY_LABEL_DIALOG_HEADER
#define ECV_FILTER_BY_LABEL_DIALOG_HEADER

// Local
#include <ecvOverlayDialog.h>
#include <ui_filterByLabelDlg.h>

// CV_CORE_LIB
#include <CVConst.h>

class ccHObject;
class ccPointCloud;

//! Dialog to sepcify a range of SF values and how the corresponding points
//! should be extracted
class ecvFilterByLabelDlg : public ccOverlayDialog,
                            public Ui::FilterByLabelDialog {
    Q_OBJECT

public:
    //! Default constructor
    ecvFilterByLabelDlg(QWidget* parent = nullptr);
    virtual ~ecvFilterByLabelDlg();

    // inherited from ccOverlayDialog
    virtual bool start() override;
    virtual void stop(bool state) override;
    virtual bool linkWith(QWidget* win) override;

    //! Mode
    enum Mode { EXPORT_SELECTED, EXPORT_UNSELECTED, SPLIT, CANCEL };

    //! Returns the selected mode
    Mode mode() const { return m_mode; }

    //! Adds an entity to the 'selected' entities set
    /** Only the 'selected' entities are moved.
            \return success, if the entity is eligible for graphical
    transformation
    **/
    bool setInputEntity(ccHObject* entity);
    void clear();

protected:
    void apply();
    void clearLayoutWidgets(QLayout* layout);
    void createCheckboxesWithLabels();
    void getSelectedFilterClasses(std::vector<ScalarType>& filteredClasses);

protected slots:
    void cancel();
    void selectAllClasses();
    void toggleSelectedVisibility();
    void onSplit() {
        m_mode = SPLIT;
        apply();
    }
    void onExportSelected() {
        m_mode = EXPORT_SELECTED;
        apply();
    }
    void onExportUnSelected() {
        m_mode = EXPORT_UNSELECTED;
        apply();
    }

protected:
    Mode m_mode;
    typedef std::pair<ccHObject*, ccPointCloud*> EntityAndVerticesType;
    EntityAndVerticesType m_toFilter;

    double m_minVald;
    double m_maxVald;

    std::vector<size_t> m_labels;
};

#endif  // ECV_FILTER_BY_LABEL_DIALOG_HEADER

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Qt
#include <QIcon>
#include <QObject>
#include <QString>

// ECV_DB_LIB
#include <ecvHObject.h>

// ECV_PLUGINS
#include <ecvPluginInterface.h>

class QAction;
class ccPointCloud;
class ecvMainAppInterface;

//! PCL filter description
struct PclModuleDescription {
public:
    QString m_module_name;
    QString m_entry_name;
    QString m_status_tip;
    QIcon m_icon;

    PclModuleDescription()
        : m_module_name("PCLModule"),
          m_entry_name("module"),
          m_status_tip("Compute something with PCL"),
          m_icon(QIcon(QString::fromUtf8(":/toolbar/pcl.png"))) {}

    PclModuleDescription(QString filterName,
                         QString entryName,
                         QString statusTip,
                         QString icon)
        : m_module_name(filterName),
          m_entry_name(entryName),
          m_status_tip(statusTip),
          m_icon(QIcon(icon)) {}
};

//! Base abstract class for each implemented PCL filter
/** \note Remember to override compute(), at least
        \author DAHAI LU
**/
class BasePclModule : public QObject {
    Q_OBJECT

public:
    /** \brief Default constructor
     *	\param[in] desc a PclModuleDescription structure containing filter infos
     *	\param[in] parent_plugin parent plugin (optional)
     */
    BasePclModule(PclModuleDescription desc = PclModuleDescription(),
                  ccPluginInterface* parent_plugin = 0);

    /** \brief Get the action associated with the button
     * used in menu and toolbar creation
     */
    QAction* getAction();

    //! Returns the error message corresponding to a given error code
    /** Each filter have a set of possible error message to be used given bt
    getFilterErrorMessage() Baseclass implementation provides some generic
    messages. \note These messages can be replaced by re-implementing this
    method and handling the same codes BEFORE calling the baseclass method
    **/
    virtual QString getErrorMessage(int errorCode);

    //! Returns the status tip
    /** Status tip is visualized in status bar when button is hovered.
            used in QAction creation
    **/
    QString getStatusTip() const;

    //! Returns the name of the filter
    QString getModuleName() const;

    //! Returns the entry name
    /** Entry name is used when creating the corresponding QAction by initAction
     **/
    QString getEntryName() const;

    //! Returns the icon associated with the button
    QIcon getIcon() const;

    //! Sets whether to show a progressbar while computing or not
    void setShowProgressBar(bool status) { m_show_progress = status; }

    //! Returns a vector of strings representing the names of the available
    //! scalar fields
    /** For the first selected entity.
     **/
    std::vector<std::string> getSelectedAvailableScalarFields();

    //! Returns the first selected entity as a ccPointCloud
    /** \return nullptr if no cloud is selected or if is not a ccPointCloud
     **/
    ccPointCloud* getSelectedEntityAsCCPointCloud() const;

    //! Returns the first selected entity as a ccHObject
    /** \return nullptr if no object is selected
     **/
    ccHObject* getSelectedEntityAsCCHObject() const;

    //! Get selected object that also have the provided metadata key
    ccHObject::Container getSelectedThatHaveMetaData(const QString key) const;

    //! Returns all the objects in db tree of type "type"
    void getAllEntitiesOfType(CV_CLASS_ENUM type,
                              ccHObject::Container& entities);

    //! Returns all the existent hierarchical objects which have a specific
    //! metadata
    /** May return an empty container if none found.
     **/
    void getAllEntitiesThatHaveMetaData(QString key,
                                        ccHObject::Container& entities);

    //! get all entities that are selected and that also are cc_point_cloud
    void getSelectedEntitiesThatAreCCPointCloud(ccHObject::Container& entities);

    //! get all entities that are selected and that also are of the specified
    //! type
    void getSelectedEntitiesThatAre(CV_CLASS_ENUM kind,
                                    ccHObject::Container& entities);

    //! Returns 1 if the first selected entity has RGB info
    int hasSelectedRGB();

    //! Returns 1 if the first selected entity has at least one scalar field
    int hasSelectedScalarField();

    //! Returns 1 if the first selected entity has a scalar field with name
    //! field_name
    int hasSelectedScalarField(std::string field_name);

    //! Returns 1 if the first selected object is a ccPointCloud
    int isFirstSelectedCcPointCloud();

    //! Updates the internal copy of selected entities
    /** 'selectedEntities' is a vector of pointers to selected entities.
     **/
    virtual void updateSelectedEntities(
            const ccHObject::Container& selectedEntities);

    //! Performs the actual filter job
    /** This method MUST be re-implemented by derived filter
            \return 1 if successful (error code otherwise)
    **/
    virtual int compute() = 0;

    //! Sets associated CC application interface (to access DB)
    void setMainAppInterface(ecvMainAppInterface* app) { m_app = app; }

    //! Returns associated CC application interface for accessing to some of
    //! mainWindow methods
    ecvMainAppInterface* getMainAppInterface() { return m_app; }

    //! Returns the associated parent plugin interface
    ccPluginInterface* getParentPlugin() const { return m_parent_plugin; }

protected slots:

    //! Returns is called when the dialog window is accepted.
    /** it can be overridden but normally should not be necessary
            the parameters will be retrieved from the dialog
            via the getParametersFromDialog() method
            this always need to be overridden.
    **/
    // DGM: useless as dialogs are always modal
    // virtual int dialogAccepted();

    //! Called when action is triggered
    /** \note performAction calls start() in base class
     **/
    int performAction();

signals:

    //! Signal emitted when an entity is (visually) modified
    void entityHasChanged(ccHObject*);

    //! Signal emitted when a new entity is created by the filter
    void newEntity(ccHObject*);

    //! Signal emitted when a new error message is produced
    void newErrorMessage(QString);

protected:
    //! Checks if current selection is compliant with the filter
    /** If not, an error is returned and computation stops.
            By default, baseclass method simply checks that selection
            is composed of one and only one ccPointCloud.
            This method should be overridden if necessary.
            \return 1 if selection is compliant (error code otherwise)
    **/
    virtual int checkSelected();

    //! Opens the input dialog window. Where the user can supply parameters for
    //! the computation
    /** Automatically called by performAction.
            Does nothing by default. Must be overridden if a dialog
            must be displayed.
            \return 1 if dialog has been successfully executed, 0 if canceled,
    negative error code otherwise
    **/
    virtual int openInputDialog() { return 1; }

    //! Opens the output dialog window. To be used when the computations have
    //! output to be shown in a dedicated dialog (as plots, histograms, etc)
    /** Automatically called by performAction.
            Does nothing by default. Must be overridden if a output dialog
            must be displayed.
            \return 1 if dialog has been successfully executed, 0 if canceled,
    negative error code otherwise
    **/
    virtual int openOutputDialog() { return 1; }

    //! Collects parameters from the filter dialog (if openDialog is successful)
    /** Automatically called by performAction.
            Does nothing by default. Must be overridden if necessary.
    **/
    virtual void getParametersFromDialog() {}

    //! Checks that the parameters retrieved by getParametersFromDialog are
    //! valid
    /** Automatically called by performAction.
            Does nothing by default. Must be overridden if necessary.
            \return 1 if parameters are valid (error code otherwise)
    */
    virtual int checkParameters() { return 1; }

    //! Starts computation
    /** Automatically called by performAction.
            By default, baseclass method simply calls compute
            Can be overridden if needed (e.g. a pre-processing step before
    compute()) \return 1 if whole process is successful (error code otherwise)
    **/
    virtual int start();

    //! Initializes the corresponding action
    /** Action can be retrieved with getAction.
     **/
    virtual void initAction();

    //! Emits the error corresponding to a given error code (see
    //! newErrorMessage)
    /** Error messages are retrieved from getErrorMessage() and
    getFilterErrorMessage() \param errCode Error code (identifies a given error
    message)
    **/
    void throwError(int errCode);

    //! Forces the Ui to be updated
    /** Simply calls m_q_parent->UpdateUI();
     **/
    // DGM: dirty! library has no access to GUI (see MVC architecture)
    // virtual void updateUi();

    //! The filter action (created by initAction)
    QAction* m_action;

    //! Pointer to the main window
    /** Used for accessing qCC functionalities from filters
     **/
    // DGM: dirty! library has no access to GUI (see MVC architecture)
    // MainWindow * m_q_parent;

    //! Currently selected entities
    /** Updated using updateSelectedEntities()
            \note DGM: now called by qPCL! (see MVC architecture)
    **/
    ccHObject::Container m_selected;

    //! Associated dialog
    /** Created inside the derived class constructor
     **/
    // DGM: dirty! some siblings must define the same variable with a different
    // type! QFileDialog * m_dialog;

    //! Filter information
    /** Contains all informations about the given filter, as name etc..
            Passed to the BasePclModule class constructor.
    **/
    PclModuleDescription m_desc;

    //! Associated application interface
    ecvMainAppInterface* m_app;

    //! associated parent plugin of the filter
    ccPluginInterface* m_parent_plugin;

    //! Do we want to show a progress bar when the filter works?
    bool m_show_progress;
};

#ifndef CC_COMMAND_LINE_PARSER_HEADER
#define CC_COMMAND_LINE_PARSER_HEADER

//interface
#include "ecvCommandLineInterface.h"

//Local
#include "ecvPluginManager.h"

class ecvProgressDialog;
class QDialog;

//! Command line parser
class ccCommandLineParser : public ccCommandLineInterface
{
public:

	//! Parses the input command
	static int Parse(int nargs, char** args, ccPluginInterfaceList& plugins);

	//! Destructor
	virtual ~ccCommandLineParser();

	//inherited from ccCommandLineInterface
	virtual QString getExportFilename(	const CLEntityDesc& entityDesc,
										QString extension = QString(),
										QString suffix = QString(),
										QString* baseOutputFilename = nullptr,
										bool forceNoTimestamp = false) const override;
	virtual QString exportEntity(	CLEntityDesc& entityDesc,
									const QString& suffix = QString(),
									QString* baseOutputFilename = nullptr,
									ccCommandLineInterface::ExportOptions options = ExportOption::NoOptions) override;
	virtual void removeClouds(bool onlyLast = false) override;
	virtual void removeMeshes(bool onlyLast = false) override;
	virtual QStringList& arguments() override { return m_arguments; }
	virtual const QStringList& arguments() const override { return m_arguments; }
	virtual bool registerCommand(Command::Shared command) override;
	virtual QDialog* widgetParent() override { return m_parentWidget; }
	virtual void print(const QString& message) const override;
	virtual void warning(const QString& message) const override;
	virtual bool error(const QString& message) const override; //must always return false!
	virtual bool saveClouds(QString suffix = QString(), bool allAtOnce = false, const QString* allAtOnceFileName = 0) override;
	virtual bool saveMeshes(QString suffix = QString(), bool allAtOnce = false, const QString* allAtOnceFileName = 0) override;
	virtual bool importFile(QString filename, FileIOFilter::Shared filter = FileIOFilter::Shared(0)) override;
	virtual QString cloudExportFormat() const override { return m_cloudExportFormat; }
	virtual QString cloudExportExt() const override { return m_cloudExportExt; }
	virtual QString meshExportFormat() const override { return m_meshExportFormat; }
	virtual QString meshExportExt() const override { return m_meshExportExt; }
	virtual QString hierarchyExportFormat() const override { return m_hierarchyExportFormat; }
	virtual QString hierarchyExportExt() const override { return m_hierarchyExportExt; }
	virtual void setCloudExportFormat(QString format, QString ext) override { m_cloudExportFormat = format; m_cloudExportExt = ext; }
	virtual void setMeshExportFormat(QString format, QString ext) override { m_meshExportFormat = format; m_meshExportExt = ext; }
	virtual void setHierarchyExportFormat(QString format, QString ext) override { m_hierarchyExportFormat = format; m_hierarchyExportExt = ext; }

protected: //other methods

	//! Default constructor
	/** Shouldn't be called by user.
	**/
	ccCommandLineParser();

	void  registerBuiltInCommands();

	void  cleanup();

	//! Parses the command line
	int start(QDialog* parent = 0);

private: //members

	//! Current cloud(s) export format (can be modified with the 'COMMAND_CLOUD_EXPORT_FORMAT' option)
	QString m_cloudExportFormat;
	//! Current cloud(s) export extension (warning: can be anything)
	QString m_cloudExportExt;
	//! Current mesh(es) export format (can be modified with the 'COMMAND_MESH_EXPORT_FORMAT' option)
	QString m_meshExportFormat;
	//! Current mesh(es) export extension (warning: can be anything)
	QString m_meshExportExt;
	//! Current hierarchy(ies) export format (can be modified with the 'COMMAND_HIERARCHY_EXPORT_FORMAT' option)
	QString m_hierarchyExportFormat;
	//! Current hierarchy(ies) export extension (warning: can be anything)
	QString m_hierarchyExportExt;

	//! Mesh filename
	QString m_meshFilename;

	//! Arguments
	QStringList m_arguments;

	//! Registered commands
	QMap< QString, Command::Shared > m_commands;

	//! Oprhan entities
	ccHObject m_orphans;

	//! Shared progress dialog
	ecvProgressDialog* m_progressDialog;

	//! Widget parent
	QDialog* m_parentWidget;
};

#endif

// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "PythonConfig.h"
#include "Utilities.h"

#include <QApplication>
#include <QDebug>
#include <QDir>
#include <QMessageBox>
#include <QProcess>
#include <QVector>
#include <QtGlobal>

// Qt5/Qt6 Compatibility
#include <QtCompat.h>

#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
#include <QTextCodec>
#endif

#if defined(USE_EMBEDDED_MODULES)
#if defined(Q_OS_WINDOWS)
static QString BundledSitePackagesPath()
{
    return QDir::listSeparator() + QApplication::applicationDirPath() +
           "/plugins/Python/Lib/site-packages";
}
#elif defined(Q_OS_MACOS)
static QString BundledSitePackagesPath()
{
    return QDir::listSeparator() + QApplication::applicationDirPath() +
           "/../Resources/python/lib/site-packages";
}
#else
static QString BundledSitePackagesPath()
{
    return QDir::listSeparator() + QApplication::applicationDirPath() +
           "/plugins/Python/lib/site-packages";
}
#endif
#endif

//================================================================================

Version::Version(const QtCompatStringRef &versionStr) : Version()
{
    QString str = qtCompatStringRefToString(versionStr);
    auto parts = qtCompatSplitRefChar(str, '.');
    if (parts.size() == 3)
    {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
        versionMajor = parts[0].toString().toUInt();
        versionMinor = parts[1].toString().toUInt();
        versionPatch = parts[2].toString().toUInt();
#else
        versionMajor = parts[0].toUInt();
        versionMinor = parts[1].toUInt();
        versionPatch = parts[2].toUInt();
#endif
    }
}

bool Version::isCompatibleWithCompiledVersion() const
{
    return versionMajor == PythonVersion.versionMajor && versionMinor == PythonVersion.versionMinor;
}

bool Version::operator==(const Version &other) const
{
    return versionMajor == other.versionMajor && versionMinor == other.versionMinor &&
           versionPatch == other.versionPatch;
}

static Version GetPythonExeVersion(QProcess &pythonProcess)
{
    pythonProcess.setArguments({"--version"});
    pythonProcess.start(QIODevice::ReadOnly);
    pythonProcess.waitForFinished();

    const QString versionStr =
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
        QString::fromUtf8(pythonProcess.readAllStandardOutput());
#else
        QTextCodec::codecForName("utf-8")->toUnicode(pythonProcess.readAllStandardOutput());
#endif

    auto splits = qtCompatSplitRefChar(versionStr, ' ');
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    if (splits.size() == 2 && splits[0].toString().contains("Python"))
    {
        return Version(qtCompatStringRef(splits[1].toString()));
#else
    if (splits.size() == 2 && splits[0].contains("Python"))
    {
        return Version(qtCompatStringRef(splits[1].toString()));
#endif
    }
    return Version{};
}
//================================================================================

struct PyVenvCfg
{
    PyVenvCfg() = default;

    static PyVenvCfg FromFile(const QString &path);

    QString home{};
    bool includeSystemSitesPackages{};
    Version version;
};

PyVenvCfg PyVenvCfg::FromFile(const QString &path)
{
    PyVenvCfg cfg{};

    QFile cfgFile(path);
    if (cfgFile.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        while (!cfgFile.atEnd())
        {
            QString line = cfgFile.readLine();
            QStringList v = line.split("=");

            if (v.size() == 2)
            {
                QString name = v[0].simplified();
                QString value = v[1].simplified();

                if (name == "home")
                {
                    cfg.home = value;
                }
                else if (name == "include-system-site-packages")
                {
                    cfg.includeSystemSitesPackages = (value == "true");
                }
                else if (name == "version")
                {
                    cfg.version = Version(qtCompatStringRef(value));
                }
            }
        }
    }

    return cfg;
}

//================================================================================

bool PythonConfigPaths::isSet() const
{
    return m_pythonHome != nullptr && m_pythonPath != nullptr;
}

const wchar_t *PythonConfigPaths::pythonHome() const
{
    return m_pythonHome.get();
}

const wchar_t *PythonConfigPaths::pythonPath() const
{
    return m_pythonPath.get();
}

//================================================================================

static QString PathToPythonExecutableInEnv(PythonConfig::Type envType, const QString &envRoot)
{
#if defined(Q_OS_WINDOWS)
    switch (envType)
    {
    case PythonConfig::Type::Conda:
        return envRoot + "/python.exe";
    case PythonConfig::Type::Venv:
        return envRoot + "/Scripts/python.exe";
    case PythonConfig::Type::Bundled:
        return envRoot + "/python.exe";
    case PythonConfig::Type::System:
        return "python.exe";
    }
#else
    switch (envType)
    {
    case PythonConfig::Type::Conda:
    case PythonConfig::Type::Venv:
    case PythonConfig::Type::Bundled:
        return envRoot + "/bin/python";
    case PythonConfig::Type::System:
        return "python";
    }
#endif
    return {};
}

void PythonConfig::initDefault()
{
#if defined(Q_OS_LINUX) || defined(Q_OS_MACOS)
    m_type = Type::System;
#else
    m_type = Type::Bundled;
#endif
}

void PythonConfig::initBundled()
{
#if defined(Q_OS_MACOS)
    const QString pythonEnvDirPath(QApplication::applicationDirPath() + "/../Resources/python");
#else
    const QString pythonEnvDirPath(QApplication::applicationDirPath() + "/plugins/Python");
#endif
    initFromLocation(pythonEnvDirPath);
}

void PythonConfig::initFromLocation(const QString &prefix)
{
    QDir envRoot(prefix);

    if (!envRoot.exists())
    {
        m_pythonHome = QString();
        m_pythonPath = QString();
        m_type = Type::Bundled;
        return;
    }

    if (envRoot.exists("pyvenv.cfg"))
    {
        QString pythonExePath = PathToPythonExecutableInEnv(Type::Venv, prefix);
        initFromPythonExecutable(pythonExePath);
        if (m_pythonHome.isEmpty() && m_pythonPath.isEmpty())
        {
            qDebug() << "Failed to get paths info from python executable at (venv)"
                     << pythonExePath;
            initVenv(envRoot.path());
        }
        else
        {
            m_type = Type::Venv;
        }
    }
    else if (envRoot.exists("conda-meta"))
    {
        QString pythonExePath = PathToPythonExecutableInEnv(Type::Conda, prefix);
        initFromPythonExecutable(pythonExePath);
        if (m_pythonHome.isEmpty() && m_pythonPath.isEmpty())
        {
            qDebug() << "Failed to get paths info from python executable at (conda)"
                     << pythonExePath;
            initCondaEnv(envRoot.path());
        }
        else
        {
            m_type = Type::Conda;
        }
    }
    else
#if defined(Q_OS_WIN32) || defined(Q_OS_MACOS)
    {
        QString pythonExePath = PathToPythonExecutableInEnv(Type::Bundled, prefix);
        initFromPythonExecutable(pythonExePath);
        if (m_pythonHome.isEmpty() && m_pythonPath.isEmpty())
        {
            qDebug() << "Failed to get paths info from python executable at (bundled)"
                     << pythonExePath;
            initVenv(envRoot.path());
        }
        else
        {
            m_type = Type::Bundled;
        }
    }
#else
    {
        m_pythonHome = envRoot.path();
        m_pythonPath = QString("%1/DLLs;%1/lib;%1/Lib;%1/Lib/site-packages;").arg(m_pythonHome);
        m_type = Type::Bundled;

#if defined(USE_EMBEDDED_MODULES)
        m_pythonPath.append(BundledSitePackagesPath());
#endif
    }
#endif
}

void PythonConfig::initCondaEnv(const QString &condaPrefix)
{
    m_type = Type::Conda;
    m_pythonHome = condaPrefix;
    m_pythonPath = QString("%1/DLLs;%1/lib;%1/Lib;%1/Lib/site-packages;").arg(condaPrefix);

#if defined(USE_EMBEDDED_MODULES)
    m_pythonPath.append(BundledSitePackagesPath());
#endif
}

void PythonConfig::initVenv(const QString &venvPrefix)
{
    PyVenvCfg cfg = PyVenvCfg::FromFile(QString("%1/pyvenv.cfg").arg(venvPrefix));

    m_type = Type::Venv;
    m_pythonHome = venvPrefix;
    m_pythonPath = QString("%1/Lib;%1/Lib/site-packages;%3/DLLs;%3/lib;").arg(venvPrefix, cfg.home);
    if (cfg.includeSystemSitesPackages)
    {
        m_pythonPath.append(QString("%1/Lib/site-packages;").arg(cfg.home));
    }

#if defined(USE_EMBEDDED_MODULES)
    m_pythonPath.append(BundledSitePackagesPath());
#endif
}

void PythonConfig::preparePythonProcess(QProcess &pythonProcess) const
{
    const QString pythonExePath = PathToPythonExecutableInEnv(type(), m_pythonHome);
    pythonProcess.setProgram(pythonExePath);

    // Conda env have SSL related libraries stored in a part that is not
    // in the path of the python exe, we have to add it ourselves.
    if (m_type == Type::Conda)
    {
#if defined(Q_OS_WINDOWS)
        const QString additionalPath = QString("%1/Library/bin").arg(m_pythonHome);
#else
        const QString additionalPath = QString("%1/lib/bin").arg(m_pythonHome);
#endif

        QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
        QString path = env.value("PATH").append(QDir::listSeparator()).append(additionalPath);
        env.insert("PATH", path);
        pythonProcess.setProcessEnvironment(env);
    }
}

PythonConfigPaths PythonConfig::pythonCompatiblePaths() const
{
    PythonConfigPaths paths;
    paths.m_pythonHome.reset(QStringToWcharArray(m_pythonHome));
    paths.m_pythonPath.reset(QStringToWcharArray(m_pythonPath));
    return paths;
}

Version PythonConfig::getVersion() const
{
    QProcess pythonProcess;
    preparePythonProcess(pythonProcess);
    return GetPythonExeVersion(pythonProcess);
}

bool PythonConfig::validateAndDisplayErrors(QWidget *parent) const
{
    Version envVersion = getVersion();
    if (envVersion.isNull())
    {
        // This hints that the selected directory is likely not valid.
        QMessageBox::warning(
            parent,
            "Invalid Python Environment",
            "The selected directory does not seems to be a valid python environment");
        return false;
    }

    if (!envVersion.isCompatibleWithCompiledVersion())
    {
        QMessageBox::warning(
            parent,
            "Incompatible Python Environment",
            QString("The selected directory does not contain a Python Environment that is "
                    "compatible. Expected a python version like %1.%2.x, selected environment "
                    "has version %3.%4.%5")
                .arg(QString::number(PythonVersion.versionMajor),
                     QString::number(PythonVersion.versionMinor),
                     QString::number(envVersion.versionMajor),
                     QString::number(envVersion.versionMinor),
                     QString::number(envVersion.versionPatch)));
        return false;
    }

    return true;
}

bool PythonConfig::IsInsideEnvironment()
{
    return qEnvironmentVariableIsSet("CONDA_PREFIX") || qEnvironmentVariableIsSet("VIRTUAL_ENV");
}

PythonConfig PythonConfig::fromContainingEnvironment()
{
    PythonConfig config;

    QString root = qEnvironmentVariable("CONDA_PREFIX");
    if (!root.isEmpty())
    {
        const QString pythonExePath = PathToPythonExecutableInEnv(Type::Conda, root);
        config.initFromPythonExecutable(pythonExePath);
        config.m_type = Type::Conda;
        return config;
    }

    root = qEnvironmentVariable("VIRTUAL_ENV");
    if (!root.isEmpty())
    {
        const QString pythonExePath = PathToPythonExecutableInEnv(Type::Venv, root);
        config.initFromPythonExecutable(pythonExePath);
        config.m_type = Type::Venv;
        return config;
    }

    return config;
}

void PythonConfig::initFromPythonExecutable(const QString &pythonExecutable)
{
    m_type = Type::Bundled;

    const QString pythonPathScript = QStringLiteral(
        "import os;import sys;print(os.pathsep.join(sys.path[1:]));print(sys.prefix, end='')");

    QProcess pythonProcess;
    pythonProcess.setProgram(pythonExecutable);
    pythonProcess.setArguments({"-c", pythonPathScript});
    pythonProcess.start(QIODevice::ReadOnly);
    pythonProcess.waitForFinished();

    const QString result =
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
        QString::fromUtf8(pythonProcess.readAllStandardOutput());
#else
        QTextCodec::codecForName("utf-8")->toUnicode(pythonProcess.readAllStandardOutput());
#endif

    QStringList pathsAndHome = result.split('\n');

    if (pathsAndHome.size() != 2)
    {
        plgPrint() << "pythonExecutable: " << pythonExecutable;
        plgWarning() << "'" << result << "' could not be parsed as a list if paths and a home path."
                     << "Expected 2 strings found " << pathsAndHome.size();
        return;
    }

    m_pythonPath = pathsAndHome.takeFirst();
    m_pythonHome = pathsAndHome.takeFirst();

#if defined(USE_EMBEDDED_MODULES)
    m_pythonPath.append(BundledSitePackagesPath());
#endif
}

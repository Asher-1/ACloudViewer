// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "PythonInterpreter.h"
#include "PythonStdErrOutRedirect.h"
#include "Runtime/Consoles.h"
#include "Utilities.h"

#include <pybind11/embed.h>

#include <QApplication>
#include <QCoreApplication>
#include <QDir>
#include <QListWidgetItem>
#include <QMessageBox>
// Qt5/Qt6 Compatibility
#include <QtCompat.h>

#include <CVLog.h>
#include <CVTools.h>

#ifdef Q_OS_LINUX
#include <cstdio>
#include <dlfcn.h>
#endif

#include <vector>

// seems like gcc defines macro with these names
#undef major
#undef minor

namespace py = pybind11;

static py::dict CreateGlobals()
{
    py::dict globals;
    globals["__name__"] = "__main__";
    // Someday we should require pybind11 > 2.6 and use py::detail::ensure_builtins_in_globals ?
    globals["__builtins__"] = PyEval_GetBuiltins();
    return globals;
}

PythonInterpreter::State::State() : globals(CreateGlobals()), locals() {};

//================================================================================

PythonInterpreter::PythonInterpreter(QObject *parent) : QObject(parent) {}

bool PythonInterpreter::executeFile(const std::string &filepath)
{
    if (m_isExecuting)
    {
        return false;
    }
    Q_EMIT executionStarted();

    bool success{true};
    try
    {
        const auto movePolicy = py::return_value_policy::move;
        py::object newStdout = py::cast(ccConsoleOutput(), movePolicy);
        py::object newStderr = py::cast(ccConsoleOutput(), movePolicy);
        PyStdErrOutStreamRedirect r{newStdout, newStderr};

        py::dict globals = CreateGlobals();
        globals["__file__"] = filepath;
        py::eval_file(filepath, globals);
    }
    catch (const std::exception &e)
    {
        CVLog::Warning(e.what());
        success = false;
    }

    Q_EMIT executionFinished();
    return success;
}

template <pybind11::eval_mode mode>
void PythonInterpreter::executeCodeString(const std::string &code,
                                          QListWidget *output,
                                          State &state)
{
    if (m_isExecuting)
    {
        return;
    }

    Q_EMIT executionStarted();
    m_isExecuting = true;
    const QColor orange(255, 100, 0);

    try
    {
        if (output != nullptr)
        {
            const auto movePolicy = py::return_value_policy::move;
            py::object newStdout = py::cast(ListWidgetConsole(output), movePolicy);
            py::object newStderr = py::cast(ListWidgetConsole(output, orange), movePolicy);
            PyStdErrOutStreamRedirect redirect{newStdout, newStderr};
            py::eval<mode>(code, state.globals, state.locals);
        }
        else
        {
            PyStdErrOutStreamRedirect redirect;
            py::eval<mode>(code, state.globals, state.locals);
        }
    }
    catch (const std::exception &e)
    {
        if (output)
        {
            auto message = new QListWidgetItem(e.what());
            message->setForeground(Qt::red);
            output->addItem(message);
        }
        else
        {
            CVLog::Error(e.what());
        }
    }

    m_isExecuting = false;
    Q_EMIT executionFinished();
}

void PythonInterpreter::executeCodeWithState(const std::string &code,
                                             QListWidget *output,
                                             State &state)
{
    executeCodeString<py::eval_mode::eval_statements>(code, output, state);
}

void PythonInterpreter::executeStatementWithState(const std::string &code,
                                                  QListWidget *output,
                                                  State &state)
{
    executeCodeString<py::eval_mode::eval_single_statement>(code, output, state);
}

void PythonInterpreter::executeCode(const std::string &code, QListWidget *output)
{
    State tmpState;
    executeCodeWithState(code, output, tmpState);
}

void PythonInterpreter::executeFunction(const pybind11::object &function)
{
    if (m_isExecuting)
    {
        return;
    }

    m_isExecuting = true;
    Q_EMIT executionStarted();
    try
    {
        py::gil_scoped_acquire scopedGil;
        PyStdErrOutStreamRedirect scopedRedirect;
        function();
    }
    catch (const std::exception &e)
    {
        CVLog::Error("Failed to start Python actions: %s", e.what());
    }
    m_isExecuting = false;
    Q_EMIT executionFinished();
}

void PythonInterpreter::initialize(const PythonConfig &config)
{
    plgPrint() << "Initializing the interpreter with: " << config;

#ifdef Q_OS_LINUX
    // Work-around issue: undefined symbol: PyExc_RecursionError
    // when trying to import numpy in the intepreter
    // e.g: https://github.com/numpy/numpy/issues/14946
    // https://stackoverflow.com/questions/49784583/numpy-import-fails-on-multiarray-extension-library-when-called-from-embedded-pyt
    // This workaround is weak

    const auto displaydlopenError = []()
    {
        char *error = dlerror();
        if (error)
        {
            plgWarning() << "dlopen error: " << error;
        }
    };

    QDir appDir = QCoreApplication::applicationDirPath();
    QString libDirPath = QString("%1/lib").arg(appDir.absolutePath());
    QDir libDir(libDirPath);

    char soName[25];
    snprintf(
        soName, 24, "libpython%d.%d.so*", PythonVersion.versionMajor, PythonVersion.versionMinor);

    QStringList filters;
    filters << soName;
    libDir.setNameFilters(filters);
    QStringList fileList = libDir.entryList(QDir::Files);
    if (!fileList.isEmpty())
    {
        QString soFilePath = QString("%1/%2").arg(libDirPath, fileList.first());
        m_libPythonHandle =
            dlopen(CVTools::FromQString(soFilePath).c_str(), RTLD_LAZY | RTLD_GLOBAL);
    }

    if (!m_libPythonHandle)
    {
        displaydlopenError();
        snprintf(soName,
                 24,
                 "libpython%d.%dm.so*",
                 PythonVersion.versionMajor,
                 PythonVersion.versionMinor);
        filters.clear();
        filters << soName;
        libDir.setNameFilters(filters);
        fileList = libDir.entryList(QDir::Files);
        if (!fileList.isEmpty())
        {
            QString soFilePath = QString("%1/%2").arg(libDirPath, fileList.first());
            m_libPythonHandle =
                dlopen(CVTools::FromQString(soFilePath).c_str(), RTLD_LAZY | RTLD_GLOBAL);
        }

        if (!m_libPythonHandle)
        {
            displaydlopenError();
        }
    }
#endif
    if (config.type() != PythonConfig::Type::System)
    {
        // We use PEP 0587 to init the interpreter.
        // The changes introduced in this PEP allows to handle the error
        // when the interpreter could not be initialized.
        //
        // Before that the python interpreter would simply exit the program
        // and that could be a bad user experience
        //
        // https://www.python.org/dev/peps/pep-0587/
        // https://docs.python.org/3/c-api/init_config.html#init-python-config
        m_config = config.pythonCompatiblePaths();
        PyStatus status;

        // Force UTF-8 mode (PEP 540) before the interpreter is initialized.
        // utf8_mode lives on PyPreConfig (not PyConfig), so it must be set via
        // pre-initialization to avoid filesystem-encoding failures in isolated
        // mode when the locale is not set (e.g. init_fs_encoding codec errors).
        PyPreConfig preConfig;
        PyPreConfig_InitIsolatedConfig(&preConfig);
        preConfig.utf8_mode = 1;
        status = Py_PreInitialize(&preConfig);
        if (PyStatus_Exception(status))
        {
            throw std::runtime_error(status.err_msg);
        }

        PyConfig pyConfig;
        PyConfig_InitPythonConfig(&pyConfig);
        pyConfig.isolated = 1;
        // Explicitly ignore the host environment (pyenv, PYTHONHOME, PYTHONPATH,
        // ...) so "Bundled" mode always uses the install-path python.
        pyConfig.use_environment = 0;

        // Anchor stdlib discovery to the bundled/home python executable. Without
        // this, PyConfig_Read derives the module search path from the host
        // executable/environment, which can leak the machine's (pyenv) python
        // into the interpreter and break init_fs_encoding (encodings codec).
        // Only needed on POSIX bundled layouts (bin/python + lib/pythonX.Y);
        // Windows bundled uses Lib/DLLs and its own program_name handling.
#ifndef Q_OS_WINDOWS
        if (const wchar_t *pythonHome = m_config.pythonHome();
            pythonHome != nullptr && *pythonHome != L'\0')
        {
            const std::wstring programName = std::wstring(pythonHome) + L"/bin/python";
            status = PyConfig_SetString(&pyConfig, &pyConfig.program_name, programName.c_str());
            if (PyStatus_Exception(status))
            {
                PyConfig_Clear(&pyConfig);
                throw std::runtime_error(status.err_msg);
            }
        }
#endif

        status = PyConfig_SetString(&pyConfig, &pyConfig.home, m_config.pythonHome());
        if (PyStatus_Exception(status))
        {
            PyConfig_Clear(&pyConfig);
            throw std::runtime_error(status.err_msg);
        }

        // Isolated boots IGNORE pythonpath_env: getpath derives the module
        // search path from home alone, and the site module then appends the
        // scheme of whichever libpython actually loaded — on hosts with a
        // same-ABI distro python (noble's /usr/lib/python3.12 vs the pyenv
        // build python) that scheme is dist-packages, so the build
        // interpreter's site-packages (numpy, ...) never enter sys.path
        // (CI cpu-noble 'ModuleNotFoundError: No module named numpy' in
        // test_qpythonruntime_plugin_loading; reproduced and diagnosed in a
        // noble container: home-anchored boots still miss site-packages).
        // Set the search path explicitly from the config's path string
        // (split on both ';' — this file's historical join separator — and
        // ':' — the POSIX env separator) so the boot is host-independent.
        {
            const std::wstring joined(m_config.pythonPath() ? m_config.pythonPath() : L"");
            std::vector<std::wstring> pathItems;
            size_t start = 0;
            while (start <= joined.size())
            {
                size_t sep = joined.find_first_of(L";:", start);
                if (sep == std::wstring::npos)
                    sep = joined.size();
                std::wstring item = joined.substr(start, sep - start);
                const size_t first = item.find_first_not_of(L" \t");
                if (first != std::wstring::npos)
                {
                    const size_t last = item.find_last_not_of(L" \t");
                    pathItems.push_back(item.substr(first, last - first + 1));
                }
                if (sep == joined.size())
                    break;
                start = sep + 1;
            }
            if (!pathItems.empty())
            {
                std::vector<wchar_t *> items;
                items.reserve(pathItems.size());
                for (std::wstring &value : pathItems)
                    items.push_back(value.data());
                status = PyConfig_SetWideStringList(&pyConfig,
                                                    &pyConfig.module_search_paths,
                                                    static_cast<Py_ssize_t>(items.size()),
                                                    items.data());
                if (PyStatus_Exception(status))
                {
                    PyConfig_Clear(&pyConfig);
                    throw std::runtime_error(status.err_msg);
                }
                // Without the _set flag getpath recomputes (and discards)
                // the list during initialization.
                pyConfig.module_search_paths_set = 1;
            }
        }

        status = PyConfig_Read(&pyConfig);
        if (PyStatus_Exception(status))
        {
            PyConfig_Clear(&pyConfig);
            throw std::runtime_error(status.err_msg);
        }

        status = Py_InitializeFromConfig(&pyConfig);
        if (PyStatus_Exception(status))
        {
            PyConfig_Clear(&pyConfig);
            throw std::runtime_error(status.err_msg);
        }
    }
    else
    {
        Py_Initialize();
    }

#if defined(Q_OS_LINUX) && defined(RTLD_DEEPBIND)
    // Mesa/LLVM may already expose libedit's incompatible readline symbols.
    // Bind this optional C extension to its own dependency before user imports.
    // Restore the flags even on failure; C++ extensions must retain normal loading.
    py::dict readlineGlobals;
    readlineGlobals["deepbind"] = RTLD_DEEPBIND;
    py::exec(R"(
import importlib.util
import sys
if importlib.util.find_spec("readline") is not None:
    flags = sys.getdlopenflags()
    try:
        sys.setdlopenflags(flags | deepbind)
        import readline
    finally:
        sys.setdlopenflags(flags)
)", readlineGlobals);
#endif

    // Make sure this module is imported
    // so that we can later easily construct our consoles.
    py::module::import("ccinternals");
}

bool PythonInterpreter::IsInitialized()
{
    return Py_IsInitialized();
}

void PythonInterpreter::finalize()
{
    if (Py_IsInitialized())
    {
        py::finalize_interpreter();
#ifdef Q_OS_LINUX
        if (m_libPythonHandle)
        {
            dlclose(m_libPythonHandle);
            m_libPythonHandle = nullptr;
        }
#else
        Q_UNUSED(this);
#endif
    }
}

bool PythonInterpreter::isExecuting() const
{
    return m_isExecuting;
}

const PythonConfigPaths &PythonInterpreter::config() const
{
    return m_config;
}

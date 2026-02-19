// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QMenu>
#include <QPair>
#include <QVector>

#include "CVAppCommon.h"

/**
 * @class ccTranslationManager
 * @brief Manager for application translations/localization
 * 
 * Singleton class that handles loading and switching between language
 * translations. Supports multiple translation file prefixes (for plugins
 * and core application) and automatic language detection.
 * 
 * Translation file naming convention:
 * - Format: `<prefix>_<lang>.qm` (compiled) or `<prefix>_<lang>.ts` (source)
 * - Language: 2-letter ISO 639 code (lowercase)
 * - Example: `CloudViewer_fr.qm` for French, `CloudViewer_de.qm` for German
 * 
 * Features:
 * - Multiple translation file registration (core + plugins)
 * - Automatic language detection from system locale
 * - Runtime language switching
 * - Menu population with available languages
 * - Persistent language preference storage
 * 
 * @see QTranslator
 */
class CVAPPCOMMON_LIB_API ccTranslationManager : public QObject {
    Q_OBJECT

public:
    /**
     * @brief Get singleton instance
     * @return Reference to translation manager
     */
    static ccTranslationManager &get();

    /**
     * @brief Destructor
     */
    ~ccTranslationManager() override = default;

    /**
     * @brief Register translation file prefix
     * 
     * Registers a translation file prefix for loading. Files should
     * follow the naming convention: `<prefix>_<lang>.qm`
     * 
     * Example: For prefix "CloudViewer" and path "/app/translations",
     * the manager will look for:
     * - CloudViewer_en.qm
     * - CloudViewer_fr.qm
     * - CloudViewer_de.qm
     * - etc.
     * 
     * @param prefix File prefix (e.g., "CloudViewer")
     * @param path Directory containing translation files
     */
    void registerTranslatorFile(const QString &prefix, const QString &path);

    /**
     * @brief Load translations for current language
     * 
     * Loads all registered translation files for the currently
     * selected or system-default language.
     */
    void loadTranslations();

    /**
     * @brief Load translations for specific language
     * 
     * Loads all registered translation files for the specified language.
     * Language code should be 2-letter ISO 639 lowercase (e.g., "en", "fr").
     * 
     * @param language ISO 639 language code
     */
    void loadTranslation(QString language);

    /**
     * @brief Populate menu with language choices
     * 
     * Scans translation directory and adds menu items for each
     * available language. User can select language from menu to
     * switch at runtime.
     * 
     * @param menu Menu to populate with language items
     * @param pathToTranslationFiles Path to scan for translation files
     */
    void populateMenu(QMenu *menu, const QString &pathToTranslationFiles);

protected:
    /**
     * @brief Protected constructor (singleton)
     */
    explicit ccTranslationManager() = default;

private:
    /**
     * @struct TranslatorFile
     * @brief Information about a translation file
     */
    struct TranslatorFile {
        QString prefix;  ///< File prefix
        QString path;    ///< Directory path
    };
    using TranslatorFileList = QVector<TranslatorFile>;

    /// Translation info: language code + display name
    using TranslationInfo = QPair<QString, QString>;
    using LanguageList = QVector<TranslationInfo>;

    /**
     * @brief Get saved language preference
     * @return Preferred language code
     */
    const QString languagePref();

    /**
     * @brief Get list of available languages
     * 
     * Scans translation directory for available language files.
     * @param appName Application name
     * @param pathToTranslationFiles Path to scan
     * @return List of available languages
     */
    LanguageList availableLanguages(const QString &appName,
                                    const QString &pathToTranslationFiles);

    /**
     * @brief Save language preference
     * @param languageCode Language code to save
     */
    void setLanguagePref(const QString &languageCode);

    TranslatorFileList mTranslatorFileInfo;  ///< Registered translation files
};

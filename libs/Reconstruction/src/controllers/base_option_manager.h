// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <boost/program_options.hpp>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "util/hash_containers.h"
#include "util/logging.h"
#include "util/string.h"
#include "util/types.h"

namespace config = boost::program_options;

namespace colmap {

// Helper for the RegisterOption static_assert below.
template <typename T>
struct always_false : std::false_type {};

// Upstream parity (d3ccaf35 controllers/base_option_manager.h): the generic
// option-management surface shared by every CLI/GUI entry point. It owns the
// boost description, the typed option registries used by Write()/Reset(), the
// project/database/image paths, idempotent option-group registration, enum
// options, and the Parse/Read/ReRead/Write pipeline. The CLI-only
// OptionManager derives from it and adds the task-specific option groups.
//
// Fork deviations from upstream (documented where they occur):
//  - The constructor does not invoke AddRandomOptions()/AddLogOptions(); a
//    derived manager overrides these with its legacy flag spellings
//    (--log_to_stderr/--random_seed), and calling virtuals from the base
//    constructor would bind the base versions and poison the group guards.
//    Groups are instead added through AddAllOptions()/explicit calls.
//  - The four Add*Options entry points are virtual for the same reason.
//  - The log options omit --log_severity/--log_color: the fork's glog build
//    declares neither FLAGS_minloglevel nor a stdout color flag.
class BaseOptionManager {
public:
    NON_COPYABLE(BaseOptionManager)

    explicit BaseOptionManager(bool add_project_options = true);
    BaseOptionManager(BaseOptionManager&&) = default;
    BaseOptionManager& operator=(BaseOptionManager&&) = default;
    virtual ~BaseOptionManager() = default;

    virtual void AddRandomOptions();
    virtual void AddLogOptions();
    virtual void AddDatabaseOptions();
    virtual void AddImageOptions();

    template <typename T>
    void AddRequiredOption(const std::string& name,
                           T* option,
                           const std::string& help_text = "");
    template <typename T>
    void AddDefaultOption(const std::string& name,
                          T* option,
                          const std::string& help_text = "");
    // An alias must not install a default value: boost::program_options would
    // otherwise overwrite an explicitly supplied sibling option during notify.
    template <typename T>
    void AddAliasOption(const std::string& name,
                        T* option,
                        const std::string& help_text = "");
    template <typename EnumT>
    void AddDefaultEnumOption(const std::string& name,
                              EnumT* option,
                              std::string_view (*to_string_fn)(EnumT),
                              EnumT (*from_string_fn)(std::string_view),
                              const std::string& help_text = "");

    virtual void Reset(bool reset_logging = true);
    virtual void ResetOptions(bool reset_paths);
    virtual bool Check();
    bool Parse(int argc, char** argv);
    virtual bool Read(const std::filesystem::path& path,
                      bool allow_unregistered = true);
    bool ReRead(const std::filesystem::path& path,
                bool reset_logging = true,
                bool allow_unregistered = true);
    void Write(const std::filesystem::path& path) const;

    std::shared_ptr<std::filesystem::path> project_path;
    std::shared_ptr<std::filesystem::path> database_path;
    std::shared_ptr<std::filesystem::path> image_path;

protected:
    template <typename T>
    void RegisterOption(const std::string& name, const T* option);

    bool RegisterOptionGroupOnce(const std::string& option_group);
    bool HasOptionGroup(const std::string& option_group) const;

    // Shared reset helpers: the derived manager composes them inside its own
    // Reset()/ResetOptions() overrides without recursing through them.
    void ResetPaths();
    void ClearRegisteredOptions();

    virtual void PostParse() {}
    virtual void ApplyLogFlags();
    virtual void PrintHelp() const;
    virtual void AddAllOptions();

    std::shared_ptr<boost::program_options::options_description> desc_;
    std::string log_target_ = "stderr_and_file";
    std::vector<std::pair<std::string, const bool*>> options_bool_;
    std::vector<std::pair<std::string, const int*>> options_int_;
    std::vector<std::pair<std::string, const double*>> options_double_;
    std::vector<std::pair<std::string, const std::string*>> options_string_;
    std::vector<std::pair<std::string, const std::filesystem::path*>>
            options_path_;
    struct EnumOptionInfo {
        std::string value;            // String value for parsing and Write()
        std::function<void()> apply;  // String -> enum conversion callback
    };
    std::vector<std::unique_ptr<EnumOptionInfo>> enum_options_;
    FlatHashSet<std::string> added_option_groups_;

private:
    void ApplyEnumConversions();
};

template <typename T>
void BaseOptionManager::AddRequiredOption(const std::string& name,
                                          T* option,
                                          const std::string& help_text) {
    if constexpr (std::is_same<T, std::filesystem::path>::value) {
        desc_->add_options()(
                name.c_str(),
                boost::program_options::value<std::string>()
                        ->required()
                        ->notifier([option](const std::string& val) {
                            *option = val;
                        }),
                help_text.c_str());
    } else {
        desc_->add_options()(
                name.c_str(),
                boost::program_options::value<T>(option)->required(),
                help_text.c_str());
    }
    RegisterOption(name, option);
}

template <typename T>
void BaseOptionManager::AddDefaultOption(const std::string& name,
                                         T* option,
                                         const std::string& help_text) {
    if constexpr (std::is_floating_point<T>::value) {
        desc_->add_options()(
                name.c_str(),
                boost::program_options::value<T>(option)->default_value(
                        *option, StringPrintf("%.3g", *option)),
                help_text.c_str());
    } else if constexpr (std::is_same<T, std::filesystem::path>::value) {
        desc_->add_options()(
                name.c_str(),
                boost::program_options::value<std::string>()
                        ->default_value(option->string())
                        ->notifier([option](const std::string& val) {
                            *option = val;
                        }),
                help_text.c_str());
    } else {
        desc_->add_options()(
                name.c_str(),
                boost::program_options::value<T>(option)->default_value(
                        *option),
                help_text.c_str());
    }
    RegisterOption(name, option);
}

template <typename T>
void BaseOptionManager::AddAliasOption(const std::string& name,
                                       T* option,
                                       const std::string& help_text) {
    desc_->add_options()(name.c_str(), boost::program_options::value<T>(option),
                         help_text.c_str());
}

template <typename T>
void BaseOptionManager::RegisterOption(const std::string& name,
                                       const T* option) {
    if constexpr (std::is_same<T, bool>::value) {
        options_bool_.emplace_back(name, reinterpret_cast<const bool*>(option));
    } else if constexpr (std::is_same<T, int>::value) {
        options_int_.emplace_back(name, reinterpret_cast<const int*>(option));
    } else if constexpr (std::is_same<T, double>::value) {
        options_double_.emplace_back(name,
                                     reinterpret_cast<const double*>(option));
    } else if constexpr (std::is_same<T, std::string>::value) {
        options_string_.emplace_back(
                name, reinterpret_cast<const std::string*>(option));
    } else if constexpr (std::is_same<T, std::filesystem::path>::value) {
        options_path_.emplace_back(
                name, reinterpret_cast<const std::filesystem::path*>(option));
    } else {
        static_assert(always_false<T>::value, "Unsupported option type");
    }
}

template <typename EnumT>
void BaseOptionManager::AddDefaultEnumOption(
        const std::string& name,
        EnumT* option,
        std::string_view (*to_string_fn)(EnumT),
        EnumT (*from_string_fn)(std::string_view),
        const std::string& help_text) {
    // Create storage for this enum option (unique_ptr for pointer stability).
    auto info = std::make_unique<EnumOptionInfo>();
    info->value = std::string(to_string_fn(*option));

    EnumOptionInfo* info_ptr = info.get();
    info->apply = [info_ptr, option, from_string_fn]() {
        *option = from_string_fn(info_ptr->value);
    };

    // Register as a string option pointing at the storage: Write() emits the
    // registration-time value through the shared options_string_ registry and
    // Read()/Parse() round-trip it back through ApplyEnumConversions().
    AddDefaultOption(name, &info->value, help_text);

    enum_options_.push_back(std::move(info));
}

}  // namespace colmap

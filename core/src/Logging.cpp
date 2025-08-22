// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "Logging.h"

#include <fmt/core.h>
#include <fmt/printf.h>
#include <fmt/ranges.h>

namespace cloudViewer {
namespace utility {

std::function<void(const std::string &)> Logger::Impl::console_print_fcn_ =
        [](const std::string &msg) { std::cout << msg << std::endl; };

Logger::Logger() : impl_(new Logger::Impl()) {
    impl_->print_fcn_ = Logger::Impl::console_print_fcn_;
    impl_->verbosity_level_ = VerbosityLevel::Info;
}

Logger &Logger::GetInstance() {
    static Logger instance;
    return instance;
}

void Logger::VError [[noreturn]] (const char *file,
                                  int line,
                                  const char *function,
                                  const std::string &message) const {
    std::string err_msg = fmt::format("[ Error] ({}) {}:{}: {}\n", function,
                                      file, line, message);
    err_msg = impl_->ColorString(err_msg, TextColor::Red, 1);
#ifdef _MSC_VER  // Uncaught exception error messages not shown in Windows
    std::cerr << err_msg << std::endl;
#endif
    throw std::runtime_error(err_msg);
}

void Logger::VWarning(const char *file,
                      int line,
                      const char *function,
                      const std::string &message) const {
    std::string err_msg = fmt::format("[CloudViewer WARNING] {}", message);
    err_msg = impl_->ColorString(err_msg, TextColor::Yellow, 1);
    impl_->print_fcn_(err_msg);
}

void Logger::VInfo(const char *file,
                   int line,
                   const char *function,
                   const std::string &message) const {
    std::string err_msg = fmt::format("[CloudViewer INFO] {}", message);
    impl_->print_fcn_(err_msg);
}

void Logger::VDebug(const char *file,
                    int line,
                    const char *function,
                    const std::string &message) const {
    std::string err_msg = fmt::format("[CloudViewer DEBUG] {}", message);
    impl_->print_fcn_(err_msg);
}

void Logger::SetPrintFunction(
        std::function<void(const std::string &)> print_fcn) {
    impl_->print_fcn_ = print_fcn;
}

const std::function<void(const std::string &)> Logger::GetPrintFunction() {
    return impl_->print_fcn_;
}

void Logger::ResetPrintFunction() {
    impl_->print_fcn_ = impl_->console_print_fcn_;
}

void Logger::SetVerbosityLevel(VerbosityLevel verbosity_level) {
    impl_->verbosity_level_ = verbosity_level;
}

VerbosityLevel Logger::GetVerbosityLevel() const {
    return impl_->verbosity_level_;
}

void SetVerbosityLevel(VerbosityLevel level) {
    Logger::GetInstance().SetVerbosityLevel(level);
}

VerbosityLevel GetVerbosityLevel() {
    return Logger::GetInstance().GetVerbosityLevel();
}

ConsoleProgressBar::ConsoleProgressBar(size_t expected_count,
                                       const std::string &progress_info,
                                       bool active) {
    reset(expected_count, progress_info, active);
}

void ConsoleProgressBar::reset(size_t expected_count,
                               const std::string &progress_info,
                               bool active) {
    expected_count_ = expected_count;
    current_count_ = static_cast<size_t>(-1);  // Guaranteed to wraparound
    progress_info_ = progress_info;
    progress_pixel_ = 0;
    active_ = active;
    operator++();
}

ConsoleProgressBar &ConsoleProgressBar::operator++() {
    setCurrentCount(current_count_ + 1);
    return *this;
}

void ConsoleProgressBar::setCurrentCount(size_t n) {
    current_count_ = n;
    if (!active_) {
        return;
    }
    if (current_count_ >= expected_count_) {
        fmt::print("{}[{}] 100%\n", progress_info_,
                   std::string(resolution_, '='));
    } else {
        size_t new_progress_pixel =
                int(current_count_ * resolution_ / expected_count_);
        if (new_progress_pixel > progress_pixel_) {
            progress_pixel_ = new_progress_pixel;
            int percent = int(current_count_ * 100 / expected_count_);
            fmt::print("{}[{}>{}] {:d}%\r", progress_info_,
                       std::string(progress_pixel_, '='),
                       std::string(resolution_ - 1 - progress_pixel_, ' '),
                       percent);
            fflush(stdout);
        }
    }
}

}  // namespace utility
}  // namespace cloudViewer

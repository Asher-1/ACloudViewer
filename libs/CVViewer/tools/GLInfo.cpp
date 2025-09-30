// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "CloudViewer.h"

void GLFWErrorCallback(int error, const char *description) {
    cloudViewer::utility::LogWarning("GLFW Error: {}", description);
}

void TryGLVersion(int major,
                  int minor,
                  bool forwardCompat,
                  bool setProfile,
                  int profileId) {
    using namespace cloudViewer;
    using namespace visualization;

    std::string forwardCompatStr =
            (forwardCompat ? "GLFW_OPENGL_FORWARD_COMPAT " : "");
    std::string profileStr = "UnknownProfile";
#define CLOUDVIEWER_CHECK_PROFILESTR(p) \
    if (profileId == p) {               \
        profileStr = #p;                \
    }
    CLOUDVIEWER_CHECK_PROFILESTR(GLFW_OPENGL_CORE_PROFILE);
    CLOUDVIEWER_CHECK_PROFILESTR(GLFW_OPENGL_COMPAT_PROFILE);
    CLOUDVIEWER_CHECK_PROFILESTR(GLFW_OPENGL_ANY_PROFILE);
#undef CLOUDVIEWER_CHECK_PROFILESTR

    cloudViewer::utility::LogInfo("TryGLVersion: {:d}.{:d} {}{}", major, minor,
                                  forwardCompatStr, profileStr);

    cloudViewer::utility::SetVerbosityLevel(
            cloudViewer::utility::VerbosityLevel::Debug);

    glfwSetErrorCallback(GLFWErrorCallback);
#ifdef HEADLESS_RENDERING
    glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_NULL);
#endif
    if (!glfwInit()) {
        cloudViewer::utility::LogError("Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, major);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, minor);
    if (forwardCompat) glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    if (setProfile) glfwWindowHint(GLFW_OPENGL_PROFILE, profileId);
    glfwWindowHint(GLFW_VISIBLE, 0);

    GLFWwindow *window_ = glfwCreateWindow(640, 480, "GLInfo", NULL, NULL);
    if (!window_) {
        cloudViewer::utility::LogDebug("Failed to create window");
        glfwTerminate();
        return;
    } else {
        glfwMakeContextCurrent(window_);
    }

    auto reportGlStringFunc = [](GLenum id, std::string name) {
// Note: with GLFW 3.3.9 it appears that OpenGL entry points are no longer auto
// loaded? The else part crashes on Apple with a null pointer.
#ifdef __APPLE__
        PFNGLGETSTRINGIPROC _glGetString =
                (PFNGLGETSTRINGIPROC)glfwGetProcAddress("glGetString");
        const auto r = _glGetString(id, 0);
#else
        const auto r = glGetString(id);
#endif
        if (!r) {
            cloudViewer::utility::LogWarning("Unable to get info on {} id {:d}",
                                             name, id);
        } else {
            cloudViewer::utility::LogDebug("{}:\t{}", name,
                                           reinterpret_cast<const char *>(r));
        }
    };
#define CLOUDVIEWER_REPORT_GL_STRING(n) reportGlStringFunc(n, #n)
    CLOUDVIEWER_REPORT_GL_STRING(GL_VERSION);
    CLOUDVIEWER_REPORT_GL_STRING(GL_RENDERER);
    CLOUDVIEWER_REPORT_GL_STRING(GL_VENDOR);
    CLOUDVIEWER_REPORT_GL_STRING(GL_SHADING_LANGUAGE_VERSION);
    // CLOUDVIEWER_REPORT_GL_STRING(GL_EXTENSIONS);
#undef CLOUDVIEWER_REPORT_GL_STRING

    if (window_) glfwDestroyWindow(window_);
    glfwTerminate();
}

int main(int argc, char **argv) {
    TryGLVersion(1, 0, false, false, GLFW_OPENGL_ANY_PROFILE);
    TryGLVersion(3, 2, true, true, GLFW_OPENGL_CORE_PROFILE);
    TryGLVersion(4, 1, false, false, GLFW_OPENGL_ANY_PROFILE);
    TryGLVersion(3, 3, false, true, GLFW_OPENGL_CORE_PROFILE);
    TryGLVersion(3, 3, true, true, GLFW_OPENGL_CORE_PROFILE);
    TryGLVersion(3, 3, false, true, GLFW_OPENGL_COMPAT_PROFILE);
    TryGLVersion(3, 3, false, true, GLFW_OPENGL_ANY_PROFILE);
    TryGLVersion(1, 0, false, true, GLFW_OPENGL_ANY_PROFILE);
    return 0;
}

// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                          -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 asher-1.github.io
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
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
    if (profileId == p) {          \
        profileStr = #p;           \
    }
    CLOUDVIEWER_CHECK_PROFILESTR(GLFW_OPENGL_CORE_PROFILE);
    CLOUDVIEWER_CHECK_PROFILESTR(GLFW_OPENGL_COMPAT_PROFILE);
    CLOUDVIEWER_CHECK_PROFILESTR(GLFW_OPENGL_ANY_PROFILE);
#undef CLOUDVIEWER_CHECK_PROFILESTR

    cloudViewer::utility::LogInfo("TryGLVersion: {:d}.{:d} {}{}", major, minor,
                     forwardCompatStr, profileStr);

    cloudViewer::utility::SetVerbosityLevel(cloudViewer::utility::VerbosityLevel::Debug);

    glfwSetErrorCallback(GLFWErrorCallback);
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
        const auto r = glGetString(id);
        if (!r) {
            cloudViewer::utility::LogWarning("Unable to get info on {} id {:d}", name, id);
        } else {
            cloudViewer::utility::LogDebug("{}:\t{}", name, r);
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

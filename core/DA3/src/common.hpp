#pragma once
#include <cstdio>

#ifdef DA3_HAS_CVLOG
#include <CVLog.h>
#define DA_LOG(...)  CVLog::Print("[DA3] " __VA_ARGS__)
#define DA_WARN(...) CVLog::Warning("[DA3] " __VA_ARGS__)
#define DA_ERR(...)  CVLog::Error("[DA3] " __VA_ARGS__)
#else
#define DA_LOG(...)  do { std::fprintf(stderr, "[da3] " __VA_ARGS__); std::fprintf(stderr, "\n"); } while (0)
#define DA_WARN(...) DA_LOG(__VA_ARGS__)
#define DA_ERR(...)  DA_LOG(__VA_ARGS__)
#endif

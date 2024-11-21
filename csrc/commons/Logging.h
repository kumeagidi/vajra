#pragma once

#include "StdCommon.h"
//==============================================================================
enum LogLevel {
  DEBUG,
  INFO,
  WARNING,
  ERROR,
  CRITICAL
};
//==============================================================================
#define LOG_LEVEL INFO
//==============================================================================
#define LOG_DEBUG(fmt, ...)                                                 \
    if (LOG_LEVEL <= DEBUG)                                       \
        fprintf(stderr, "[%d][DEBUG][%s:%d:%s] " fmt "\n", (int)time(NULL), __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__);
#define LOG_INFO(fmt, ...)                                                  \
    if (LOG_LEVEL <= INFO)                                        \
        fprintf(stderr, "[%d][INFO][%s:%d:%s] " fmt "\n", (int)time(NULL), __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__);
#define LOG_WARNING(fmt, ...)                                               \
    if (LOG_LEVEL <= WARNING)                                     \
        fprintf(stderr, "[%d][WARNING][%s:%d:%s] " fmt "\n", (int)time(NULL), __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__);
#define LOG_ERROR(fmt, ...)                                                 \
    if (LOG_LEVEL <= ERROR)                                       \
        fprintf(stderr, "[%d][ERROR][%s:%d:%s] " fmt "\n", (int)time(NULL), __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__);
#define LOG_CRITICAL(fmt, ...)                                              \
    if (LOG_LEVEL <= CRITICAL)                                    \
        fprintf(stderr, "[%d][CRITICAL][%s:%d:%s] " fmt "\n", (int)time(NULL), __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__);
//==============================================================================
#define ASSERT(x) \
    if (!(x)) { \
        LOG_CRITICAL("ASSERTION FAILED: %s", #x); \
        exit(1); \
    }
//==============================================================================
#define TRACE_CRITICAL_AND_EXIT(fmt, ...) \
    LOG_CRITICAL(fmt, ##__VA_ARGS__); \
    exit(1);
//==============================================================================
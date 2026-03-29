#pragma once

#if defined(_WIN32)
# ifdef MRF_STATIC_DEFINE
#   define MRF_EXPORT
# else
#   ifndef MRF_EXPORT
#     ifdef MRF_EXPORTS
/* We are building this library */
#       define MRF_EXPORT __declspec(dllexport)
#     else
/* We are using this library */
#       define MRF_EXPORT __declspec(dllimport)
#     endif
#   endif
# endif
#else
# define MRF_EXPORT
#endif
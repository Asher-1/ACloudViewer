
#ifndef CV_CORE_LIB_API_H
#define CV_CORE_LIB_API_H

#ifdef CV_CORE_LIB_STATIC_DEFINE
#  define CV_CORE_LIB_API
#  define CV_CORE_LIB_NO_EXPORT
#else
#  ifndef CV_CORE_LIB_API
#      define CV_CORE_LIB_API __declspec(dllexport)
//#    ifdef CV_CORE_LIB_EXPORTS
//        /* We are building this library */
//#      define CV_CORE_LIB_API __declspec(dllexport)
//#    else
//        /* We are using this library */
//#      define CV_CORE_LIB_API __declspec(dllimport)
//#    endif
#  endif

#  ifndef CV_CORE_LIB_NO_EXPORT
#    define CV_CORE_LIB_NO_EXPORT 
#  endif
#endif

#ifndef CV_CORE_LIB_DEPRECATED
#  define CV_CORE_LIB_DEPRECATED __declspec(deprecated)
#endif

#ifndef CV_CORE_LIB_DEPRECATED_EXPORT
#  define CV_CORE_LIB_DEPRECATED_EXPORT CV_CORE_LIB_API CV_CORE_LIB_DEPRECATED
#endif

#ifndef CV_CORE_LIB_DEPRECATED_NO_EXPORT
#  define CV_CORE_LIB_DEPRECATED_NO_EXPORT CV_CORE_LIB_NO_EXPORT CV_CORE_LIB_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef CV_CORE_LIB_NO_DEPRECATED
#    define CV_CORE_LIB_NO_DEPRECATED
#  endif
#endif

#endif /* CV_CORE_LIB_API_H */

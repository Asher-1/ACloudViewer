###############################################################################
# Check for the presence of SSE and figure out the flags to use for it.
macro(CLOUDVIEWER_CHECK_FOR_SSE)
    set(SSE_FLAGS)
    set(SSE_DEFINITIONS)

    if(NOT CMAKE_CROSSCOMPILING)
        # Test GCC/G++ and CLANG
        if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_CLANG)
            include(CheckCXXCompilerFlag)
            check_cxx_compiler_flag("-march=native" HAVE_MARCH)
            if(HAVE_MARCH)
                list(APPEND SSE_FLAGS "-march=native")
            else()
                list(APPEND SSE_FLAGS "-mtune=native")
            endif()
            message(STATUS "Using CPU native flags for SSE optimization: ${SSE_FLAGS}")
        endif()
    endif()

    # Unfortunately we need to check for SSE to enable "-mfpmath=sse" alongside
    # "-march=native". The reason for this is that by default, 32bit architectures
    # tend to use the x87 FPU (which has 80 bit internal precision), thus leading
    # to different results than 64bit architectures which are using SSE2 (64 bit internal
    # precision). One solution would be to use "-ffloat-store" on 32bit (see
    # http://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html), but that slows code down,
    # so the preferred solution is to try "-mpfmath=sse" first.
    include(CheckCXXSourceRuns)
    set(CMAKE_REQUIRED_FLAGS)

    check_cxx_source_runs("
        // Intel compiler defines an incompatible _mm_malloc signature
        #if defined(__INTEL_COMPILER)
            #include <malloc.h>
        #else
            #include <mm_malloc.h>
        #endif
        int main()
        {
          void* mem = _mm_malloc (100, 16);
          return 0;
        }"
        HAVE_MM_MALLOC)

    check_cxx_source_runs("
        #include <stdlib.h>
        int main()
        {
          void* mem;
          return posix_memalign (&mem, 16, 100);
        }"
        HAVE_POSIX_MEMALIGN)

    if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_CLANG)
        set(CMAKE_REQUIRED_FLAGS "-msse4.2")
    endif()

    check_cxx_source_runs("
        #include <emmintrin.h>
        #include <nmmintrin.h>
        int main ()
        {
          long long a[2] = {  1, 2 };
          long long b[2] = { -1, 3 };
          long long c[2];
          __m128i va = _mm_loadu_si128 ((__m128i*)a);
          __m128i vb = _mm_loadu_si128 ((__m128i*)b);
          __m128i vc = _mm_cmpgt_epi64 (va, vb);

          _mm_storeu_si128 ((__m128i*)c, vc);
          if (c[0] == -1LL && c[1] == 0LL)
            return (0);
          else
            return (1);
        }"
        HAVE_SSE4_2_EXTENSIONS)

    if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_CLANG)
        set(CMAKE_REQUIRED_FLAGS "-msse4.1")
    endif()

    check_cxx_source_runs("
        #include <smmintrin.h>
        int main ()
        {
          __m128 a, b;
          float vals[4] = {1, 2, 3, 4};
          const int mask = 123;
          a = _mm_loadu_ps (vals);
          b = a;
          b = _mm_dp_ps (a, a, mask);
          _mm_storeu_ps (vals,b);
          return (0);
        }"
        HAVE_SSE4_1_EXTENSIONS)

    if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_CLANG)
        set(CMAKE_REQUIRED_FLAGS "-mssse3")
    endif()

    check_cxx_source_runs("
        #include <tmmintrin.h>
        int main ()
        {
          __m128i a, b;
          int vals[4] = {-1, -2, -3, -4};
          a = _mm_loadu_si128 ((const __m128i*)vals);
          b = _mm_abs_epi32 (a);
          _mm_storeu_si128 ((__m128i*)vals, b);
          return (0);
        }"
        HAVE_SSSE3_EXTENSIONS)

    if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_CLANG)
        set(CMAKE_REQUIRED_FLAGS "-msse3")
    endif()

    check_cxx_source_runs("
        #include <pmmintrin.h>
        int main ()
        {
            __m128d a, b;
            double vals[2] = {0};
            a = _mm_loadu_pd (vals);
            b = _mm_hadd_pd (a,a);
            _mm_storeu_pd (vals, b);
            return (0);
        }"
        HAVE_SSE3_EXTENSIONS)

    if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_CLANG)
        set(CMAKE_REQUIRED_FLAGS "-msse2")
    elseif(MSVC AND NOT CMAKE_CL_64)
        set(CMAKE_REQUIRED_FLAGS "/arch:SSE2")
    endif()

    check_cxx_source_runs("
        #include <emmintrin.h>
        int main ()
        {
            __m128d a, b;
            double vals[2] = {0};
            a = _mm_loadu_pd (vals);
            b = _mm_add_pd (a,a);
            _mm_storeu_pd (vals,b);
            return (0);
        }"
        HAVE_SSE2_EXTENSIONS)

    if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_CLANG)
        set(CMAKE_REQUIRED_FLAGS "-msse")
    elseif(MSVC AND NOT CMAKE_CL_64)
        set(CMAKE_REQUIRED_FLAGS "/arch:SSE")
    endif()

    check_cxx_source_runs("
        #include <xmmintrin.h>
        int main ()
        {
            __m128 a, b;
            float vals[4] = {0};
            a = _mm_loadu_ps (vals);
            b = a;
            b = _mm_add_ps (a,b);
            _mm_storeu_ps (vals,b);
            return (0);
        }"
        HAVE_SSE_EXTENSIONS)

    set(CMAKE_REQUIRED_FLAGS)

    if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_CLANG)
        if(HAVE_SSE4_2_EXTENSIONS)
            list(APPEND SSE_FLAGS "-msse4.2" "-mfpmath=sse")
        elseif(HAVE_SSE4_1_EXTENSIONS)
            list(APPEND SSE_FLAGS "-msse4.1" "-mfpmath=sse")
        elseif(HAVE_SSSE3_EXTENSIONS)
            list(APPEND SSE_FLAGS "-mssse3" "-mfpmath=sse")
        elseif(HAVE_SSE3_EXTENSIONS)
            list(APPEND SSE_FLAGS "-msse3" "-mfpmath=sse")
        elseif(HAVE_SSE2_EXTENSIONS)
            list(APPEND SSE_FLAGS "-msse2" "-mfpmath=sse")
        elseif(HAVE_SSE_EXTENSIONS)
            list(APPEND SSE_FLAGS "-msse" "-mfpmath=sse")
        else()
            # Setting -ffloat-store to alleviate 32bit vs 64bit discrepancies on non-SSE
            # platforms.
            list(APPEND SSE_FLAGS "-ffloat-store")
        endif()
    elseif(MSVC AND NOT CMAKE_CL_64)
        if(HAVE_SSE2_EXTENSIONS)
            list(APPEND SSE_FLAGS "/arch:SSE2")
        elseif(HAVE_SSE_EXTENSIONS)
            list(APPEND SSE_FLAGS "/arch:SSE")
        endif()
    endif()

    if(MSVC)
        if(HAVE_SSSE3_EXTENSIONS)
            set(SSE_DEFINITIONS "${SSE_DEFINITIONS} -D__SSSE3__")
        endif()
        if(HAVE_SSE2_EXTENSIONS)
            set(SSE_DEFINITIONS "${SSE_DEFINITIONS} -D__SSE2__")
        endif()
        if(HAVE_SSE_EXTENSIONS)
            set(SSE_DEFINITIONS "${SSE_DEFINITIONS} -D__SSE__")
        endif()
    endif()
    string(REPLACE ";" " " SSE_FLAGS_STR "${SSE_FLAGS}")
endmacro()

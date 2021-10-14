// ----------------------------------------------------------------------------
// -                        cloudViewer: asher-1.github.io                          -
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

#include "Cpu.h"
#include "CVPlatform.h"

#ifdef CV_WINDOWS
#include <windows.h>
namespace cloudViewer {
namespace system {

int cpu_clock_by_os(void)
{
	HKEY key;
	DWORD result;
	DWORD size = 4;
	
	if (RegOpenKeyEx(HKEY_LOCAL_MACHINE, TEXT("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"), 0, KEY_READ, &key) != ERROR_SUCCESS)
		return -1;
	
	if (RegQueryValueEx(key, TEXT("~MHz"), NULL, NULL, (LPBYTE) &result, (LPDWORD) &size) != ERROR_SUCCESS) {
		RegCloseKey(key);
		return -1;
	}
	RegCloseKey(key);
	
	return (int)result;
}
}}
#else
#ifdef CV_MAC_OS
#include <sys/types.h>
#include <sys/sysctl.h>
/* Assuming Mac OS X with hw.cpufrequency sysctl */
namespace cloudViewer {
namespace system {

int cpu_clock_by_os(void)
{
	long long result = -1;
	size_t size = sizeof(result);
	if (sysctlbyname("hw.cpufrequency", &result, &size, NULL, 0))
		return -1;
	return (int) (result / (long long) 1000000);
}
}}
#else
#include <cstdio>
#include <cstdlib>
#include <cstring>
namespace cloudViewer {
namespace system {

/* Assuming Linux with /proc/cpuinfo */
int cpu_clock_by_os(void)
{
	FILE *f;
	char line[1024], *s;
	int result;
	
	f = fopen("/proc/cpuinfo", "rt");
	if (!f) return -1;
	
	while (fgets(line, sizeof(line), f)) {
		if (!strncmp(line, "cpu MHz", 7)) {
			s = strchr(line, ':');
			if (s && 1 == sscanf(s, ":%d.", &result)) {
				fclose(f);
				return result;
			}
		}
	}
	fclose(f);
	return -1;
}
}}
#endif /* CV_MAC_OS */
#endif /* CV_WINDOWS */


/* get_total_cpus() system specific code: uses OS routines to determine total number of CPUs */
#ifdef CV_MAC_OS
#include <unistd.h>
#include <mach/clock_types.h>
#include <mach/clock.h>
#include <mach/mach.h>
namespace cloudViewer {
namespace system {

int get_total_cpus(void)
{
	kern_return_t kr;
	host_basic_info_data_t basic_info;
	host_info_t info = (host_info_t)&basic_info;
	host_flavor_t flavor = HOST_BASIC_INFO;
	mach_msg_type_number_t count = HOST_BASIC_INFO_COUNT;
	kr = host_info(mach_host_self(), flavor, info, &count);
	if (kr != KERN_SUCCESS) return 1;
	return basic_info.avail_cpus;
}
}}
#define GET_TOTAL_CPUS_DEFINED
#endif

#ifdef CV_WINDOWS
#include <windows.h>
namespace cloudViewer {
namespace system {

int get_total_cpus(void)
{
	SYSTEM_INFO system_info;
	GetSystemInfo(&system_info);
	return system_info.dwNumberOfProcessors;
}
}}
#define GET_TOTAL_CPUS_DEFINED
#endif

#if defined CV_LINUX
#include <sys/sysinfo.h>
#include <unistd.h>
namespace cloudViewer {
namespace system {

int get_total_cpus(void)
{
	return sysconf(_SC_NPROCESSORS_ONLN);
}
}}
#define GET_TOTAL_CPUS_DEFINED
#endif

#if defined __FreeBSD__ || defined __OpenBSD__ || defined __NetBSD__ || defined __bsdi__ || defined __QNX__
#include <sys/types.h>
#include <sys/sysctl.h>
namespace cloudViewer {
namespace system {

int get_total_cpus(void)
{
	int mib[2] = { CTL_HW, HW_NCPU };
	int ncpus;
	size_t len = sizeof(ncpus);
	if (sysctl(mib, 2, &ncpus, &len, (void *) 0, 0) != 0) return 1;
	return ncpus;
}
}}
#define GET_TOTAL_CPUS_DEFINED
#endif

#ifndef GET_TOTAL_CPUS_DEFINED
namespace cloudViewer {
namespace system {

int get_total_cpus(void)
{
	return 1;
}
}}

#endif /* GET_TOTAL_CPUS_DEFINED */


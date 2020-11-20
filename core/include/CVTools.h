#ifndef __TOOLS_H__
#define __TOOLS_H__

// local
#include "CVCoreLib.h"

// system
#include <string>
#include <vector>
#include <QTime>
#include <QString>
#include <QTextCodec>

using std::string;
using std::vector;

class CVTools
{
public:
	CV_CORE_LIB_API static string GetFileName(const string file_name);
	CV_CORE_LIB_API static void TimeStart();
	CV_CORE_LIB_API static QString TimeOff();

	// string to QString
	CV_CORE_LIB_API static QString toQString(const string& s);
	// QString to string
	CV_CORE_LIB_API static string fromQString(const QString& qs);

	CV_CORE_LIB_API static string joinStrVec(const vector<string>& v, string splitor = " ");

	// QString(Unicode) -> std::string (GBK)
	CV_CORE_LIB_API static string fromUnicode(const QString& qstr);

	// std::string (GBK) -> QString(Unicode)
	CV_CORE_LIB_API static QString toUnicode(const string& cstr);

	CV_CORE_LIB_API static int TranslateKeyCode(int key);

    CV_CORE_LIB_API static bool FileMappingReader(const std::string &filename, void* data, unsigned long& size);
    CV_CORE_LIB_API static bool FileMappingWriter(const std::string &filename, const void* data, unsigned long size);

	CV_CORE_LIB_API static bool QMappingReader(const std::string &filename, std::vector<size_t>& indices);
    CV_CORE_LIB_API static bool QMappingWriter(const std::string &filename, const void* data, std::size_t length);
	CV_CORE_LIB_API static std::wstring Char2Wchar(const char* szStr);
	CV_CORE_LIB_API static std::string Wchar2Char(const wchar_t* szStr);
private:
	CVTools() = delete;

	static QTime s_time;

};


#endif  // __TOOLS_H__

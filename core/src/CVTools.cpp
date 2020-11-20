#include "CVTools.h"

// CV_CORE_LIB
#include "CVLog.h"
#include "CVPlatform.h"

// QT
#include <QFile>

#include <locale> 
#include <codecvt> 

// SYSTEM
#if defined(CV_WINDOWS)
#include "windows.h"
#include "stdio.h"
#endif // (CV_WINDOWS)

QTime CVTools::s_time;

string CVTools::GetFileName(const string file_name)
{
	string subname;
	for (auto i = file_name.end() - 1; *i != '/'; i--)
	{
		subname.insert(subname.begin(), *i);
	}
	return subname;
}

void CVTools::TimeStart()
{
	s_time.start();
}

QString CVTools::TimeOff()
{
	int timediff = s_time.elapsed();
    double f = timediff / 1000.0;
	return QString("%1").arg(f); //float->QString
}

string CVTools::fromUnicode(const QString& qstr)
{
	QTextCodec* pCodec = QTextCodec::codecForName("system");
	if (!pCodec) return "";

	QByteArray arr = pCodec->fromUnicode(qstr);
	string cstr = arr.data();
	return cstr;
}

QString CVTools::toUnicode(const string& cstr)
{
	QTextCodec* pCodec = QTextCodec::codecForName("system");
	if (!pCodec) return "";

	QString qstr = pCodec->toUnicode(cstr.c_str(), static_cast<int>(cstr.length()));
	return qstr;
}

QString CVTools::toQString(const string& s)
{
#if defined(CV_WINDOWS)
	return toUnicode(s);
#else // do not support coding in Linux or mac platform!
    return QString(s.c_str());
#endif
}

string CVTools::fromQString(const QString& qs) 
{

#if defined(CV_WINDOWS)
    return fromUnicode(qs);
#else // do not support coding in Linux or mac platform!
    return qs.toStdString();
#endif
}


bool CVTools::FileMappingReader(const std::string & filename, void * data, unsigned long& size)
{
#if defined(CV_WINDOWS)
	HANDLE hFile = CreateFile(filename.c_str(),
		GENERIC_READ | GENERIC_WRITE,
		FILE_SHARE_READ,
		NULL,
		OPEN_EXISTING,
		FILE_ATTRIBUTE_NORMAL,
		NULL);

	if (hFile == INVALID_HANDLE_VALUE)
	{
		CVLog::Error("FileMappingReader: %d", GetLastError());
		CloseHandle(hFile);
		return false;
	}

	DWORD dwFileSize = GetFileSize(hFile, NULL);
	size = dwFileSize / sizeof(char);

	HANDLE hFileMapping = CreateFileMapping(hFile,
		NULL,
		PAGE_READWRITE,
		0,
		dwFileSize+sizeof(char),
		NULL);

	if (hFileMapping == NULL)
	{
		CVLog::Error("FileMappingReader: %d", GetLastError());
		CloseHandle(hFile);
		return false;
	}

	char* pbFile = (char*)MapViewOfFile(hFileMapping, FILE_MAP_READ, 0, 0, 0);
	if (pbFile == NULL)
	{
		CVLog::Error("FileMappingReader: %d", GetLastError());
		CloseHandle(hFile);
		CloseHandle(hFileMapping);
		return false;
	}

	//memcpy(pbFile, data, size);
	//MoveMemory(pbFile, data, size);
	if (data)
	{
		delete data;
	}
	data = new char[size];
	CopyMemory(data, pbFile, size);

	UnmapViewOfFile(pbFile);
	CloseHandle(hFileMapping);

	CloseHandle(hFile);
	return true;
#else
	CVLog::Warning("[FileMappingReader] only support windows!");
	return false;
#endif
}

bool CVTools::FileMappingWriter(const std::string &filename, const void* data, unsigned long size)
{
#if defined(CV_WINDOWS)
    HANDLE hFile = CreateFile(filename.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        NULL,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        NULL);

    if (hFile == INVALID_HANDLE_VALUE)
    {
        CVLog::Error("FileMappingWriter: %d", GetLastError());
        CloseHandle(hFile);
        return false;
    }

    HANDLE hFileMapping = CreateFileMapping(hFile,
        NULL,
        PAGE_READWRITE,
        0,
        size,
        NULL);

    if (hFileMapping == NULL)
    {
        CVLog::Error("FileMappingWriter: %d", GetLastError());
        CloseHandle(hFileMapping);
        return false;
    }

    TCHAR* pbFile = (TCHAR*)MapViewOfFile(hFileMapping, FILE_MAP_WRITE, 0, 0, 0);
    if (pbFile == NULL)
    {
        CVLog::Error("FileMappingWriter: %d", GetLastError());
        UnmapViewOfFile(pbFile);
        return false;
    }

    //memcpy(pbFile, data, size);
    //MoveMemory(pbFile, data, size);
    CopyMemory(pbFile, data, size);

    UnmapViewOfFile(pbFile);
    CloseHandle(hFileMapping);

    CloseHandle(hFile);
    return true;
#else
    CVLog::Warning("[FileMappingWriter] only support windows!");
    return false;
#endif
}

bool CVTools::QMappingReader(const std::string &filename, std::vector<size_t>& indices)
{
	indices.clear();

	QFile input(filename.c_str());
	if (!input.open(QIODevice::ReadOnly))
	{
        CVLog::Error(QString("[CVTools::QMappingReader] Cannot open file : %1").arg(filename.c_str()));
		return false;
	}

	int size = static_cast<int>(input.size());
	// memory map
	uchar* fptr = input.map(0, input.size());
	if (!fptr)
	{
        CVLog::Error(QString("[CVTools::QMappingReader] Cannot open file : %1").arg(filename.c_str()));
		return false;
	}

    char skipChar1[2] = "\r";
    char skipChar2[2] = "\n";
    char* buf = reinterpret_cast<char *>(fptr);
    std::string currentLine;
    int value;
    int index = 0; // line index
    for (int i = 0; i < size; ++i)
    {
        // skip \r if exist
        if (buf[i] == *skipChar1)
        {
            continue;
        }
        // detect \n and output current line
        if (buf[i] == *skipChar2)
        {
            value = std::stoi(currentLine);
            currentLine.clear();
            if (value < 0)
            {
                continue;
            }
            indices.push_back(static_cast<size_t>(value));
            index++;
            continue;
        }

        currentLine += buf[i];
    }

	input.unmap(fptr);
	input.close();
    return true;
}

bool CVTools::QMappingWriter(const std::string &filename, const void *data, std::size_t length)
{
    QFile file(filename.c_str());
    if (!file.exists())
    {
        file.open(QIODevice::WriteOnly);
        file.close();
    }

    if (!file.resize(static_cast<qint64>(length)))
    {
        CVLog::Error("[CVTools::QMappingWriter] Reserve space error! May have not enough space avaliable!");
        return false;
    }

    if (!file.open(QIODevice::ReadWrite))
    {
        CVLog::Error(QString("[CVTools::QMappingWriter] Cannot open file : %1").arg(filename.c_str()));
        return false;
    }

    // memory map
    uchar* fptr = file.map(0, file.size());
    if (!fptr)
    {
        CVLog::Error(QString("[CVTools::QMappingWriter] Mapping file(%1) failed!").arg(filename.c_str()));
        fptr = file.map(0, static_cast<qint64>(length));
        return false;
    }

    char* buf = reinterpret_cast<char *>(fptr);
    if (!buf)
    {
        CVLog::Error(QString("[CVTools::QMappingWriter] Converting uchar* to char* failed!"));
        return false;
    }

    memcpy(buf, data, length);

    file.unmap(fptr);
    file.close();
    return true;
}

string CVTools::joinStrVec(const vector<string>& v, string splitor) {
	string s = "";
	if (v.size() == 0) return s;
    for (std::size_t i = 0; i != v.size()  - 1; ++i) {
		s += (v[i] + splitor);
	}
	s += v[v.size() - 1];
	return s;
}

int CVTools::TranslateKeyCode(int key)
{
	int k = key;
	bool legal = true;
	if (k >= Qt::Key_0 && k <= Qt::Key_9)
	{
	}
	else if (k >= Qt::Key_A && k <= Qt::Key_Z)
	{
	}
	else if (k >= Qt::Key_F1 && k <= Qt::Key_F24)
	{
		k &= 0x000000ff;
		k += 0x40;
	}
	else if (k == Qt::Key_Tab)
	{
		k = 0x09;
	}
	else if (k == Qt::Key_Backspace)
	{
		k = 0x08;
	}
	else if (k == Qt::Key_Return)
	{
		k = 0x0d;
	}
	else if (k <= Qt::Key_Down && k >= Qt::Key_Left)
	{
		int off = k - Qt::Key_Left;
		k = 0x25 + off;
	}
	else if (k == Qt::Key_Shift)
	{
		k = 0x10;
	}
	else if (k == Qt::Key_Control)
	{
		k = 0x11;
	}
	else if (k == Qt::Key_Alt)
	{
		k = 0x12;
	}
	else if (k == Qt::Key_Meta)
	{
		k = 0x5b;
	}
	else if (k == Qt::Key_Insert)
	{
		k = 0x2d;
	}
	else if (k == Qt::Key_Delete)
	{
		k = 0x2e;
	}
	else if (k == Qt::Key_Home)
	{
		k = 0x24;
	}
	else if (k == Qt::Key_End)
	{
		k = 0x23;
	}
	else if (k == Qt::Key_PageUp)
	{
		k = 0x21;
	}
	else if (k == Qt::Key_Down)
	{
		k = 0x22;
	}
	else if (k == Qt::Key_CapsLock)
	{
		k = 0x14;
	}
	else if (k == Qt::Key_NumLock)
	{
		k = 0x90;
	}
	else if (k == Qt::Key_Space)
	{
		k = 0x20;
	}
	else
		legal = false;

	if (!legal)
		return 0;
	return k;
}

std::wstring CVTools::Char2Wchar(const char* szStr)
{
	std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
	std::wstring wideStr = conv.from_bytes(szStr);
	return wideStr;
}

std::string CVTools::Wchar2Char(const wchar_t* szStr)
{
	std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
	std::string wideStr = conv.to_bytes(szStr);
	return wideStr;
}

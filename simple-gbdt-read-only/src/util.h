#ifndef __UTIL_H__
#define __UTIL_H__

#include <time.h>
#include <sys/time.h>
#include <vector>
#include <string>

static int64_t Milliseconds()
{
    struct timeval t;
    ::gettimeofday(&t, NULL);
    int64_t curTime;
    curTime = t.tv_sec;
    curTime *= 1000;              // sec -> msec
    curTime += t.tv_usec / 1000;  // usec -> msec
    return curTime;
}

static void InplaceTrimLeft(std::string& strValue) {
	size_t pos = 0;
	for (size_t i = 0; i < strValue.size(); ++i) {
		if (isspace((unsigned char) strValue[i]))
			++pos;
		else
			break;
	}
	if (pos > 0)
		strValue.erase(0, pos);
}

static void InplaceTrimRight(std::string& strValue) {
	size_t n = 0;
	for (size_t i = 0; i < strValue.size(); ++i) {
		if (isspace((unsigned char) strValue[strValue.length() - i - 1]))
			++n;
		else
			break;
	}
	if (n != 0)
		strValue.erase(strValue.length() - n);
}

static void InplaceTrim(std::string& strValue) {
	InplaceTrimRight(strValue);
	InplaceTrimLeft(strValue);
}

static void Split(const std::string& strMain, char chSpliter,
		std::vector<std::string>& strList, bool bReserveNullString) {
	strList.clear();

	if (strMain.empty())
		return;

	size_t nPrevPos = 0;
	size_t nPos;
	std::string strTemp;
	while ((nPos = strMain.find(chSpliter, nPrevPos)) != std::string::npos) {
		strTemp.assign(strMain, nPrevPos, nPos - nPrevPos);
		InplaceTrim(strTemp);
		if (bReserveNullString || !strTemp.empty())
			strList.push_back(strTemp);
		nPrevPos = nPos + 1;
	}

	strTemp.assign(strMain, nPrevPos, strMain.length() - nPrevPos);
	InplaceTrim(strTemp);
	if (bReserveNullString || !strTemp.empty())
		strList.push_back(strTemp);
}
#endif

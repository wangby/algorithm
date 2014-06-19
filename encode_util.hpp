/*
 * encode_util.hpp
 *
 *       Author: wangbiyao
 */

#ifndef ENCODE_UTIL_HPP_
#define ENCODE_UTIL_HPP_

#include <string.h>
#include <iconv.h>
using namespace std;

class EncodeUtil {
public:
	static inline bool IsGBKCH(unsigned char ch1, unsigned char ch2) {//只能处理GBK编码问题，需要确认所有编码为GBK，如果不是，转化成GBK
		return ((ch2 >= 64) && (ch2 <= 254) && (ch2 != 127)
				&& ((ch1 >= 129 && ch1 <= 160) || (ch1 >= 170 && ch1 < 254)
						|| (ch1 == 254 && ch2 <= 160)));
	}

	static inline bool IsASCII(unsigned char ch1) {
		return ch1<=127;
	}

	static void SplitGBKStr2Vec(const string& str, vector<string>& ret) {
		int len = str.size();
		int i = 0;
		while(i<len) {
			unsigned char c = str[i];
			if(IsASCII(c)) {
				ret.push_back(str.substr(i,1));
				++i;
			} else if(i<len-1) {
				if(IsGBKCH(str[i], str[i+1])) {
					ret.push_back(str.substr(i,2));
					i += 2 ;
				}
			} else {
				++i;
			}
		}
	}

	static string UTF8ToGBK(const string& instr) {
		iconv_t conveter = iconv_open("GBK", "UTF-8");
		if (conveter == (iconv_t) -1 ) {
			cout << "conveter is -1" <<endl;
			return "";
		}

		const char* szInstr = instr.c_str();
		size_t in_len = instr.length();
		size_t out_len = in_len * 3 + 1;

		char* szOutstr = (char*) malloc(sizeof(char)*out_len);
		memset(szOutstr, 0, out_len);
		char *in = (char *) szInstr;
		char *out =(char *) szOutstr;

		size_t ret = iconv(conveter, (char **) &in, (size_t *) &in_len,
				(char **) &out, (size_t *) &out_len);
		if (ret == -1) {
			cout << "ret is -1" <<endl;
			return "";
		}
		iconv_close(conveter);

		string outstr(szOutstr);
		free (szOutstr);
		return outstr;
	}

	static string GBKToUTF8(const string& instr) {
		iconv_t conveter = iconv_open("UTF-8", "GBK");
		if (conveter == (iconv_t) -1 ) {
			cout << "conveter is -1" <<endl;
			return "";
		}

		const char* szInstr = instr.c_str();
		size_t in_len = instr.size();
		size_t out_len = in_len * 3 + 1;

		char* szOutstr =(char*) malloc(sizeof(char)*out_len);
		memset(szOutstr,0,out_len);

		char *in = (char*) szInstr;
		char *out = (char*) szOutstr;

		size_t ret = iconv(conveter, (char **) &in, (size_t *) &in_len,
				(char **) &out, (size_t *) &out_len);
		if (ret == -1) {
			cout << "ret is -1" <<endl;
			return "";
		}

		iconv_close(conveter);

		string outstr(szOutstr);
		free (szOutstr);

		return outstr;
	}

	static unsigned char ToHex(unsigned char x) {
		return x > 9 ? x + 55 : x + 48;
	}

	static unsigned char FromHex(unsigned char x) {
		unsigned char y = 0;
		if (x >= 'A' && x <= 'Z')
			y = x - 'A' + 10;
		else if (x >= 'a' && x <= 'z')
			y = x - 'a' + 10;
		else if (x >= '0' && x <= '9')
			y = x - '0';

		return y;
	}

	static std::string UrlEncode(const std::string& str) {
		std::string strTemp = "";
		size_t length = str.length();
		for (size_t i = 0; i < length; i++) {
			if (isalnum((unsigned char) str[i]) || (str[i] == '-')
					|| (str[i] == '_') || (str[i] == '.') || (str[i] == '~'))
				strTemp += str[i];
			else if (str[i] == ' ')
				strTemp += "+";
			else {
				strTemp += '%';
				strTemp += ToHex((unsigned char) str[i] >> 4);
				strTemp += ToHex((unsigned char) str[i] % 16);
			}
		}
		return strTemp;
	}

	static std::string UrlDecode(const std::string& str) {
		std::string strTemp = "";
		size_t length = str.length();
		for (size_t i = 0; i < length; i++) {
			if (str[i] == '+')
				strTemp += ' ';
			else if (str[i] == '%') {
//				assert(i + 2 < length);
				unsigned char high = FromHex((unsigned char) str[++i]);
				unsigned char low = FromHex((unsigned char) str[++i]);
				strTemp += high * 16 + low;
			} else
				strTemp += str[i];
		}
		return strTemp;
	}


};

#endif /* ENCODE_UTIL_HPP_ */

/*
 * StringUtils.hpp
 *
 *  Created on: 2013年12月17日
 *      Author: wangbiyao
 */

#ifndef STRINGUTILS_HPP_
#define STRINGUTILS_HPP_

#include "std.h"
using namespace std;
class StringUtil {
public:
	StringUtil();
	virtual ~StringUtil();
/*
	const static std::string HAS = "has";
	const static std::string HAS_NOT = "miss";
	const static std::string LESS_THAN = "lt";
	const static std::string GREATER_THAN = "gt";
*/
	static std::string trim(const std::string& text) {
		return rtrim(ltrim(text));
	}

	static std::string ltrim(const std::string& text) {
		return text.substr(text.find_first_not_of(" \t\n\r"));
//		string::iterator i;
//		for (i = str.begin(); i != str.end(); i++) {
//		    if (!isspace(*i)) {
//		        break;
//		    }
//		}
//		if (i == str.end()) {
//		    str.clear();
//		} else {
//		    str.erase(str.begin(), i);
//		}
	}

	static std::string rtrim(const std::string& text) {
		return text.substr(0, text.find_last_not_of(" \t\n\r") + 1);
//		string::iterator i;
//		for (i = str.end() - 1; ;i--) {
//		    if (!isspace(*i)) {
//		        str.erase(i + 1, str.end());
//		        break;
//		    }
//		    if (i == str.begin()) {
//		        str.clear();
//		        break;
//		    }
//		}
	}

	static std::string trim(const std::string& text, const std::string& trim) {
		return rtrim(ltrim(text, trim), trim);
	}

	static std::string rtrim(const std::string& text, const std::string& trim) {
		return text.substr(0, text.find_last_not_of(trim) + 1);
	}

	static std::string ltrim(const std::string& text, const std::string& trim) {
		return text.substr(text.find_first_not_of(trim));
	}

	static void split_any(const std::string text, const std::string sep,
			std::vector<std::string>& items) {
		items.clear();
		size_t begin = 0, end = 0;
		int length = text.length();

		while (begin < length) {
			end = text.find_first_of(sep, begin);
			if (end == std::string::npos) {
				items.push_back(text.substr(begin));
				break;
			} else {
				if (end > begin) {
					items.push_back(text.substr(begin, end - begin));
				}
				begin = text.find_first_not_of(sep, end);
				if (begin == std::string::npos) {
					break;
				}
			}
		}
	}

	static void split(const std::string text, const std::string sep,
			std::vector<std::string>& items) {
		items.clear();
		size_t begin = 0, end = 0;
		int length = text.length();
		int sub_length = sep.length();

		while (begin < length) {
			end = text.find(sep, begin);
			if (end == std::string::npos) {
				items.push_back(text.substr(begin));
				break;
			} else {
				if(end > begin) {
					items.push_back(text.substr(begin, end - begin));
				}

				begin = end + sub_length;
			}
		}
	}

	static std::string join(const std::vector<std::string>& items,
			const std::string& join_str = " ") {
		if (items.size() == 0) {
			return "";
		}

		std::string tmp = items[0];
		for (int i = 1; i < items.size(); i++) {
			tmp += join_str + items[i];
		}
		return tmp;
	}

	static std::string join(const std::vector<std::string>& items, int start, int end,
			const std::string& join_str = " ") {
		if (items.size() == 0 ||items.size()<end+1) {
			return "";
		}

		if(end < start) {
			return "";
		}

		std::string tmp = items[start];
		for (int i = start+1; i <= end; i++) {
			tmp += join_str + items[i];
		}
		return tmp;
	}

	static bool startswith(const std::string& text, const std::string& sub) {
		//return text.find(sub) == 0;
		if (sub.length() > text.length()) {
			return false;
		}

		return text.substr(0, sub.length()) == sub;
	}

	static bool endswith(const std::string& text, const std::string& sub) {
		if (sub.length() > text.length()) {
			return false;
		}

		return text.find(sub, text.length() - sub.length()) != std::string::npos;
	}

	static std::string repeat(const std::string& str, int n) {
		std::string s(str);

		// easy
//	    for (int i = 0; i < n; i++) {
//	        s += str;
//	    }

		if (n <= 0) {
			return "";
		}

		if (n == 1) {
			return str;
		}

		int i = 1;
		int m = n >> 1;
		while (i <= m) {
			s += s;
			i = i << 1;
		}

		s += repeat(str, n - i);

		// 非递归
//	    int j = n-i;
//	    for(int k=0;k<j;k++) {
//	    	s += str;
//	    }

		return s;
	}

	static std::string repeat(char c, int n) {
		return std::string(n, c);
	}

	static bool contains(const std::string& text, const std::string& sub) {
		return text.find(sub) != std::string::npos;
	}

	static std::string lower(const std::string& str) {
		std::string t = str;
		std::transform(t.begin(), t.end(), t.begin(), (int (*)(int))std::tolower);
		return t;
	}

	static std::string upper(const std::string& str) {
		std::string t = str;
		std::transform(t.begin(), t.end(), t.begin(), (int (*)(int))std::toupper);
		return t;
	}

	static bool equalsIgnoreCase(const std::string& str1, const std::string& str2) {
		return lower(str1) == lower(str2);
	}

	// 提取开始字符串和结束字符串之间的字符
	static void find_interval(const std::string& text, const std::string& start_str,
			const std::string& end_str, std::vector<std::string>& items) {
		items.clear();
		size_t begin = 0, end = 0;
		int length = text.length();
		int start_length = start_str.length();
		int end_length = end_str.length();
//		std::cout << text << std::endl;
		while(begin < length) {
			begin = text.find(start_str,begin);
//			std::cout << begin << std::endl;
			if(begin == std::string::npos) {
				break;
			}

			end = text.find(end_str,begin);
//			std::cout << end << std::endl;
			if (end == std::string::npos) {
				break;
			}
			std::string item = text.substr(begin+start_length, end-begin-start_length);
			items.push_back(item);
			begin = end + end_length;
		}
	}

	static std::string replace(const std::string& text, const std::string& old_str,
			const std::string& new_str) {

		if(text.empty() || old_str == new_str) {
			return text;
		}
		std::string tmp(text);
		int pos = tmp.find(old_str);
		while(pos != std::string::npos) {
			tmp.replace(pos, old_str.size(), new_str);
			pos = tmp.find(old_str);
		}

		return tmp;
	}

	static int length(const std::string& text, std::string encoding)
	{
		size_t src_len=0, unicode_len=0;
		int char_num=0;
		size_t i = 0;
		const char *p=text.c_str(); //p用于后面的遍历
		const char *addr = text.c_str();

		if(equalsIgnoreCase(encoding,"ASCII")) {
			char_num = text.length();
		} else if(equalsIgnoreCase(encoding,"UNICODE")) {
			while(1) {
				if(src_len==0xFFFFFFFF) {
					return -1;
				}

				if(text[src_len] == 0x00 && text[src_len+1] == 0x00) {
					break;
				}

				src_len+=2;
			}

			char_num = src_len /2;
		} else if(equalsIgnoreCase(encoding,"UTF8") || equalsIgnoreCase(encoding,"UTF-8")) {

			for (i = 0; i < text.length(); i++)
			{
				if ((unsigned int)text[i] < 0x80) {
					;
				} else if ((unsigned int)text[i] < 0xE0) {
					i += 1;
				} else {
					i += 2;
				}

				char_num++;
			}

//			unicode_len = char_num*2;
		} else if(equalsIgnoreCase(encoding,"GBK") || equalsIgnoreCase(encoding,"GB2312")) {
			while (*p) {
				if ((*p<0)&&(*(p+1)<0||*(p+1)>63)) {
					addr++;
					p+=2;
				} else {
					p++;
				}
			}
			char_num = p-addr;
		} else {
			char_num = -1;
		}

		return char_num;
	}
/*
	// 只保留符合条件的
	static void filter(const std::vector<std::string>& in, std::vector<std::string>& out, std::string pattern, std::string cmp) {
		out.clear();
		for(int i=0;i<in.size();i++) {
			if (equalsIgnoreCase(pattern, "has")) {
				if (contains(in[i],cmp)) {
					out.push_back(in[i]);
				}
			} else {
				if (!contains(in[i],cmp)) {
					out.push_back(in[i]);
				}
			}
		}
	}

	// 只保留符合条件的
	static void filter(const std::vector<std::string>& in, std::vector<std::string>& out,
			std::string encoding, std::string pattern, int cmp) {
		out.clear();
		for(int i=0;i<in.size();i++) {
			if (equalsIgnoreCase(pattern, "lt")) {
				if (length(in[i], encoding) < cmp) {
					out.push_back(in[i]);
				}
			} else {
				if (length(in[i], encoding) > cmp) {
					out.push_back(in[i]);
				}
			}
		}
	}
*/
	static bool isPunc(const string& src) {
	    if(src.size() == 2)
	    {
	        if(((*(unsigned char*)&src[0] >= 0xa1 && *(unsigned char*)&src[0] <= 0xa9) && (*(unsigned char*)&src[1] >= 0xa1 && *(unsigned char*)&src[1] <= 0xfe))
		 ||((*(unsigned char*)&src[0] >= 0xa8 && *(unsigned char*)&src[0] <= 0xa9) && (*(unsigned char*)&src[1] >= 0x40 && *(unsigned char*)&src[1] <= 0xa0)))
	        {
	            return true;
	        }
	    }
	    else if(src.size() == 1)
	    {
	        if(!isalnum(src[0])) {
	            return true;
	        }
	    }
	    return false;
	}

	static bool isInvalidChar(const string& word) {
		if(isPunc(word))
		{
			return true;
		}
		// else if(word == "的")
		// {
		// 	return true;
		// }
		return false;
	}

	static bool all_english(const string& text) {
		if(text.empty()) {
			return false;
		}

		for (int i = 0; i < text.size(); i++) {
			if((text[i] <= 'z' && text[i] >= 'a') || (text[i] <= 'Z' && text[i] >= 'A')) {

			} else {
				return false;
			}
		}

		return true;
	}

	static bool all_digital(const string& text) {
		if(text.empty()) {
			return false;
		}

		for (int i = 0; i < text.size(); i++) {
			if(text[i] <= '9' && text[i] >= '0') {

			} else {
				return false;
			}
		}

		return true;
	}
};

#endif /* STRINGUTILS_HPP_ */

/*
 * extract_summary.hpp
 *
 *  Created on: 2014年6月20日
 *      Author: wangbiyao
 */

#ifndef EXTRACT_SUMMARY_HPP_
#define EXTRACT_SUMMARY_HPP_

#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <queue>
#include <tr1/unordered_map>

using namespace std;
class ExtractSummary {
public:
	//已切好词

	static bool FindShortestAbstract(const std::vector<std::string> & doc, std::set<std::string> & query, int &a,
			int &b) {
		std::set<std::string> not_find(query.begin(), query.end()); // 维护一个不在窗口内的词set,find的次数
		std::tr1::unordered_map<std::string, int> find;

		a = 0;
		b = 0;
		int i = 0, j = 0;
		int shortest = 0;
		int len = doc.size();
		while (i < len && j < len) {
			if(!not_find.empty()) {
				std::set<std::string>::iterator it = not_find.find(doc[j++]);
				if (it != not_find.end()) {    // 如果找到了，则删除
					not_find.erase(it);
					find[*it] = 1;
				} else { //
					if (query.find(doc[j-1]) != query.end()) {
						find[doc[j-1]] += 1;
					}
				}
			}

			if (not_find.empty()) {                //如果全部找到
				while (query.find(doc[i++]) == query.end()) {
					; //寻找第一个出现的query
				}

				if (find[doc[i - 1]] <= 1) {
					not_find.insert(doc[i - 1]);
					if (shortest > j - i || shortest == 0) {
						shortest = j - i;        //记录最小距离
						a = i - 1;
						b = j - 1;
					}
				}
				find[doc[i - 1]] -= 1;
			}
		}

		if (shortest == 0) {
			return false;
		}
		return true;
	}

	static bool FindShortestAbstract2(const std::vector<std::string> & doc, std::set<std::string> & keywords, int &a,
			int &b) {
		std::set<std::string> not_find(keywords.begin(), keywords.end()); // 维护一个不在窗口内的词set,
		std::tr1::unordered_map<std::string, int> find; // 窗口内各词的次数
		std::queue<int> keywords_index;
		a = 0;
		b = 0;
		int j = 0;
		int shortest = 0;
		int len = doc.size();

		while (j < len) {
			if (keywords.find(doc[j]) == keywords.end()) {
				++j;
				continue;
			}
			keywords_index.push(j);
			std::set<std::string>::iterator it = not_find.find(doc[j]);
			if (it != not_find.end()) { // find in not_find set
				not_find.erase(it);
				find[*it] = 1; // first
			} else {
				find[doc[j]] += 1; // in find set
			}

			if (not_find.empty()) { // all find
				while (!keywords_index.empty()) {
					int i = keywords_index.front();
					keywords_index.pop();
					if (find[doc[i]] <= 1) {
						not_find.insert(doc[i]);
						if (shortest > j - i || shortest == 0) {
							shortest = j - i; //记录最小距离
							a = i;
							b = j;
						}
						++j;
						break;
					}
					find[doc[i]] -= 1;
				}
			} else {
				++j;
			}
		}

		if (shortest == 0) {
			return false;
		}
		return true;
	}
};


#endif /* EXTRACT_SUMMARY_HPP_ */

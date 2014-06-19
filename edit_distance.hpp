/**
 *  \file edit_distance.hpp
 *  \author wangbiyao
 *  \date  2014-06-19
 */
#ifndef _EDIT_DISTANCE_HPP_
#define _EDIT_DISTANCE_HPP_
#include <string>
#include <vector>
#include <iostream>

class EditDistance {
public:
	const static int INSERT = 2;
	const static int REPLACE = 1;
	const static int SWAP = 1;

	inline int min3(int a, int b, int c) {
		return std::min(std::min(a,b),c);
	}

	// zhongwen
	int distance(const std::string& a, const std::string& b, bool is_swap = false) {
		if(a.empty()) {
			return b.size() * INSERT;
		}

		if(b.empty()) {
			return a.size() * INSERT;
		}

		int d = 0;
		if(a[0] != b[0]) {
			d = REPLACE;
		}

		int m = min3(distance(a, b.substr(1)) + INSERT,
				distance(a.substr(1), b) + INSERT,
				distance(a.substr(1), b.substr(1)) + d);
		//swap
		if(is_swap && a[0] != b[0]) {
			if(b.size() > 1 && a[0] == b[1]) {
				std::string c = b;
				std::swap(c[0], c[1]);
				return std::min(m, SWAP + distance(a, c));
			}

			if(a.size() > 1 && b[0] == a[1]) {
				std::string c = a;
				std::swap(c[0], c[1]);
				return std::min(m, SWAP + distance(b, c));
			}
		}

		return m;
	}

	//中英文混杂 传入vector
	int distance(const std::vector<std::string>& a, int starta,
			     const std::vector<std::string>& b, int startb,
			     bool is_swap = false) {
//		std::cout <<" starta:" <<starta << " startb:"<<startb<<std::endl;
		if (a.size() <= starta) {
			int len = b.size() - startb;
			return std::max(0, len) * INSERT;
		}

		if (b.size() <= startb) {
			int len = a.size() - starta;
			return std::max(0, len) * INSERT;
		}

		int d = 0;
		if (a[starta] != b[startb]) {
			d = REPLACE;
		}

		int m = min3(distance(a, starta, b, startb + 1) + INSERT,
				distance(a, starta + 1, b, startb) + INSERT,
				distance(a, starta + 1, b, startb + 1) + d);

		if(is_swap && a[starta] != b[startb]) {
			if(b.size() > startb+1 && a[starta] == b[startb+1]) {
				std::vector<std::string> c(b);
				std::swap(c[startb], c[startb + 1]);
				return std::min(m, SWAP + distance(a, starta, c, startb));
			}

			if(a.size() > starta+1 && b[startb] == a[starta+1]) {
				std::vector<std::string> c(a);
				std::swap(c[starta], c[starta + 1]);
				return std::min(m, SWAP + distance(c, starta, b, startb));
			}
		}

		return m;
	}
};

#endif


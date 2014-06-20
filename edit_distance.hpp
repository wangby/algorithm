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
//	int distance(const std::string& a, const std::string& b, bool is_swap) {
//		if(a.empty()) {
//			return b.size() * INSERT;
//		}
//
//		if(b.empty()) {
//			return a.size() * INSERT;
//		}
//
//		int d = 0;
//		if(a[0] != b[0]) {
//			d = REPLACE;
//		}
//
//		int m = min3(distance(a, b.substr(1), is_swap) + INSERT,
//				distance(a.substr(1), b, is_swap) + INSERT,
//				distance(a.substr(1), b.substr(1), is_swap) + d);
//		//swap
//		if(is_swap && a[0] != b[0]) {
//			if(b.size() > 1 && a[0] == b[1]) {
//				std::string c = b;
//				std::swap(c[0], c[1]);
//				return std::min(m, SWAP + distance(a, c, is_swap));
//			}
//
//			if(a.size() > 1 && b[0] == a[1]) {
//				std::string c = a;
//				std::swap(c[0], c[1]);
//				return std::min(m, SWAP + distance(b, c, is_swap));
//			}
//		}
//
//		return m;
//	}

	//中英文混杂 传入vector
//	int distance(const std::vector<std::string>& a, int starta,
//			     const std::vector<std::string>& b, int startb,
//			     bool is_swap) {
////		std::cout <<" starta:" <<starta << " startb:"<<startb<<std::endl;
//		if (a.size() <= starta) {
//			int len = b.size() - startb;
//			return std::max(0, len) * INSERT;
//		}
//
//		if (b.size() <= startb) {
//			int len = a.size() - starta;
//			return std::max(0, len) * INSERT;
//		}
//
//		int d = 0;
//		if (a[starta] != b[startb]) {
//			d = REPLACE;
//		}
//
//		int m = min3(distance(a, starta, b, startb + 1, is_swap) + INSERT,
//				distance(a, starta + 1, b, startb, is_swap) + INSERT,
//				distance(a, starta + 1, b, startb + 1, is_swap) + d);
//
//		if(is_swap && a[starta] != b[startb]) {
//			if(b.size() > startb+1 && a[starta] == b[startb+1]) {
//				std::vector<std::string> c(b);
//				std::swap(c[startb], c[startb + 1]);
//				return std::min(m, SWAP + distance(a, starta, c, startb, is_swap));
//			}
//
//			if(a.size() > starta+1 && b[startb] == a[starta+1]) {
//				std::vector<std::string> c(a);
//				std::swap(c[starta], c[starta + 1]);
//				return std::min(m, SWAP + distance(c, starta, b, startb, is_swap));
//			}
//		}
//
//		return m;
//	}

	// 非递归版，用数组保存结果 string or vector
	template <class T>
	int distance(const T& a, const T& b) {
		if(a.empty()) {
			return b.size() * INSERT;
		}

		if(b.empty()) {
			return a.size() * INSERT;
		}

		int m = a.size() + 1; //rows mxn
		int n = b.size() + 1; //cols
		std::vector<std::vector<int> > d;
		d.resize(m);

		for (int i = 0; i < m; i++) {
			d[i].resize(n);
			d[i][0] = i * INSERT;
		}

		for (int j = 0; j < n; j++) {
			d[0][j] = j * INSERT;
		}

		for (int i = 1; i < m; i++) {
			for (int j = 1; j < n; j++) {
				int df = (a[i - 1] == b[j - 1]) ? 0 : REPLACE;
				d[i][j] = min3(d[i - 1][j] + INSERT, d[i][j - 1] + INSERT, d[i - 1][j - 1] + df);
			}
		}

		return d[m-1][n-1];
	}

	template<class T>
	int distance(const T& a, const T& b, bool is_swap) {
		int m = distance(a, b);
		//swap
		if(is_swap && a[0] != b[0]) {
			if(b.size() > 1 && a[0] == b[1]) {
				T c = b;
				std::swap(c[0], c[1]);
				return std::min(m, SWAP + distance(a, c));
			}

			if(a.size() > 1 && b[0] == a[1]) {
				T c = a;
				std::swap(c[0], c[1]);
				return std::min(m, SWAP + distance(b, c));
			}
		}

		return m;
	}
};

#endif


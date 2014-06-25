// Copyright 2012 Jike Inc. All Rights Reserved.
// Author: wangbiyao@jike.com

#ifndef _TOPICMODEL_PLSA_HPP
#define _TOPICMODEL_PLSA_HPP

#include "std.h"
#include <math.h>
#include <time.h>
#include <tr1/unordered_set>
#include <tr1/unordered_map>

#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include "../util/string_util.hpp"

typedef std::tr1::unordered_multiset<std::string> word_set_t;
typedef std::tr1::unordered_map<std::string, int> word_idx_t;
typedef std::vector<std::string> idx_word_t;
typedef std::vector<std::vector<double> > prob_matrix_t;
typedef std::vector<prob_matrix_t> td_matrix_t;

//
class Document {
public:
	explicit Document(const std::string& file_path) :
			lines_(0), file_path_(file_path) {
	}

	void Init() {
		cout << "file:" << file_path_<<endl;
		words_.clear();
		lines_ = 0;
		std::ifstream fin(file_path_.c_str());
		std::string s;
		while (std::getline(fin, s)) {
			std::vector<std::string> words;
			StringUtil::split(s, " ", words);
			BOOST_FOREACH(std::string & word, words) {
				words_.insert(word);
				++lines_;
			}
		}
	}

	const word_set_t& GetWords() {
		return words_;
	}

	string GetFilePath() {
		return file_path_;
	}
private:
	int lines_;
	std::string file_path_;
	word_set_t words_;
};

//
class PLSA {
public:
	PLSA(int number_of_topics, int max_iter) :
			number_of_topics_(number_of_topics), max_iter_(max_iter) {
		srand((unsigned) time(0)); // init once
	}

	void InitDocAndVoc(const std::string& doc_dir_path) {
		namespace fs = boost::filesystem;
		// init document
		fs::directory_iterator end;
		for (fs::directory_iterator pos(doc_dir_path); pos != end; ++pos) { // each file
			if(fs::is_directory(*pos)) {
				continue;
			}

			Document doc(pos->path().string());
			doc.Init();
			documents_.push_back(doc);
//			documents_[documents_.size() -1].Init(); //
		}

		// init vocabulary
		int index = 0;
		BOOST_FOREACH(Document & doc, documents_) {
			BOOST_FOREACH(const std::string& word, doc.GetWords()) {
				word_idx_t::const_iterator word_iter = vocabulary_.find(word);
				if (word_iter == vocabulary_.end()) { // new word
					vocabulary_.insert(std::make_pair(word, index++));
					idx_word_.push_back(word);
				}
			}
		}
	}

	void BuildTermDocMatrix() {
		term_doc_matrix_.resize(documents_.size());
		for (int d = 0; d < documents_.size(); ++d) {
			term_doc_matrix_[d].resize(vocabulary_.size());
			BOOST_FOREACH(const std::string & word, documents_[d].GetWords()) {
				word_idx_t::const_iterator word_iter = vocabulary_.find(word);
				if (word_iter == vocabulary_.end()) {
					continue;
				}
				int w_index = word_iter->second;
				++term_doc_matrix_[d][w_index];
			}
		}
	}

	void CreateCounterArrays() {
		document_topic_prob_.resize(documents_.size()); // size = d * z
		topic_prob_.resize(documents_.size());
		for (int d = 0; d < documents_.size(); ++d) {
			document_topic_prob_[d].resize(number_of_topics_);
			for (int z = 0; z < number_of_topics_; ++z) { // random init
				document_topic_prob_[d][z] = GenRandomNum();
			}
			Normalize(document_topic_prob_[d]);

			topic_prob_[d].resize(vocabulary_.size()); // size = d * w * z
			for (int w = 0; w < vocabulary_.size(); ++w) {
				topic_prob_[d][w].resize(number_of_topics_);
			}
		}

		topic_word_prob_.resize(number_of_topics_); // size = z * w
		for (int z = 0; z < number_of_topics_; ++z) {
			topic_word_prob_[z].resize(vocabulary_.size());
			for (int w = 0; w < vocabulary_.size(); ++w) { // random init
				topic_word_prob_[z][w] = GenRandomNum();
			}
			Normalize(topic_word_prob_[z]);
		}
	}

	void Normalize(std::vector<double>& prob) {
		double sum = 0.0;
		for (int i = 0; i < prob.size(); ++i) {
			sum += prob[i];
		}

		//if (sum == 0) { // something wrong
		//  return;
		//}

		for (int i = 0; i < prob.size(); ++i) {
			prob[i] /= sum;
		}
	}

	double GenRandomNum() {
		return rand() / (double) (RAND_MAX);
	}

	void EM() {
		for (int i = 0; i < max_iter_; ++i) {
			// E STEP
			for (int d = 0; d < documents_.size(); ++d) {
				for (int w = 0; w < vocabulary_.size(); ++w) {
					for (int z = 0; z < number_of_topics_; ++z) {
						topic_prob_[d][w][z] = document_topic_prob_[d][z] * topic_word_prob_[z][w];
					}
					Normalize(topic_prob_[d][w]);
				}
			}

			// M step
			// update p(w|z)
			for (int z = 0; z < number_of_topics_; ++z) {
				for (int w = 0; w < vocabulary_.size(); ++w) {
					double s = 0.0;
					for (int d = 0; d < documents_.size(); ++d) {
						int count = term_doc_matrix_[d][w];
						s += count * topic_prob_[d][w][z];
					}
					topic_word_prob_[z][w] = s;
				}
				Normalize(topic_word_prob_[z]);
			}

			// update p(z|d)
			for (int d = 0; d < documents_.size(); ++d) {
				for (int z = 0; z < number_of_topics_; ++z) {
					double s = 0.0;
					for (int w = 0; w < vocabulary_.size(); ++w) {
						int count = term_doc_matrix_[d][w];
						s += count * topic_prob_[d][w][z];
					}
					document_topic_prob_[d][z] = s;
				}
				Normalize(document_topic_prob_[d]);
			}
		}
	}

	int GetDocNum() {
		return documents_.size();
	}

	int GetVocNum() {
		return vocabulary_.size();
	}

	const prob_matrix_t& GetDocTopicProb() {
		return document_topic_prob_;
	}

	const prob_matrix_t& GetTopicWordProb() {
		return topic_word_prob_;
	}

	void PrintProb(int max_len, std::ostream& ost=std::cout) {
		ost << "p(z|d):"<<endl;
		for (int i = 0; i < document_topic_prob_.size(); ++i) {
			ost<< documents_[i].GetFilePath()<<" ";
			for (int j = 0; j < document_topic_prob_[i].size(); ++j) {
				double prob = document_topic_prob_[i][j];
				ost << std::setprecision(2) << std::fixed << prob << " ";
			}
			ost << std::endl;
		}
		ost << "--------------"<< std::endl;
		ost << "words:" << idx_word_.size() << std::endl;
//		for(int i=0; i<idx_word_.size(); ++i) {
//			ost << idx_word_[i] << " ";
//		}
//		ost << std::endl;

		ost << "p(w|z):"<<endl;

		for (int i = 0; i < topic_word_prob_.size(); ++i) {
			for (int j = 0; j < topic_word_prob_[i].size(); ++j) {
				double prob = topic_word_prob_[i][j];
				if (prob >= 0.01) {
					ost << idx_word_[j] << ":" << std::setprecision(2) << std::fixed << prob << " ";
				}
			}
			ost << std::endl;
		}

		ost << "p(z|w):"<<endl;
		for (int i = 0; i < word_topic_prob_.size(); ++i) {
			for (int j = 0; j < word_topic_prob_[i].size(); ++j) {
				double prob = word_topic_prob_[i][j];
				if (prob >= 0.01) {
					ost << idx_word_[i] << " topic"<<j<<":" << std::setprecision(2) << std::fixed << prob << " ";
				}
			}
			ost << std::endl;
		}
	}

	void CalcWordTopicProb() { //  p(z|w)
		word_topic_prob_.resize(vocabulary_.size());
		for (int w = 0; w < word_topic_prob_.size(); ++w) {
			word_topic_prob_[w].resize(number_of_topics_);
			for (int z = 0; z < number_of_topics_; ++z) {
				word_topic_prob_[w][z] = 0.0;
				for (int d = 0; d < topic_prob_.size(); ++d) {
					word_topic_prob_[w][z] += topic_prob_[d][w][z];
				}
			}

			Normalize(word_topic_prob_[w]);
		}
	}

	void Run(const std::string& doc_dir_path) {
		InitDocAndVoc(doc_dir_path);
		BuildTermDocMatrix();
		CreateCounterArrays();
		EM();
		CalcWordTopicProb();
	}
private:
	int number_of_topics_;
	int max_iter_;
	std::vector<Document> documents_;
	word_idx_t vocabulary_;
	idx_word_t idx_word_;
	std::vector<std::vector<int> > term_doc_matrix_; // term count in a doc

	prob_matrix_t document_topic_prob_; // p(z | d) d*z
	prob_matrix_t topic_word_prob_; // p(w | z) z*w

	td_matrix_t topic_prob_; // p(z | d, w) d*z*w


	prob_matrix_t word_topic_prob_; // p(z | w) w*z = sum(p(z|di,w),di)
	// 可计算 p(z|w)
};

#endif

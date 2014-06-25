// Copyright 2012 Jike Inc. All Rights Reserved.
// Author: wangbiyao@jike.com

#ifndef _TOPICMODEL_PLSA_HPP
#define _TOPICMODEL_PLSA_HPP

#include "std.h"
#include <math.h>
#include <time.h>
#include <tr1/unordered_set>
#include <tr1/unordered_map>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include "../util/string_util.hpp"

typedef std::tr1::unordered_multiset<std::string> word_set_t;
typedef std::tr1::unordered_map<std::string, int> word_map_t;
typedef std::vector<std::vector<double> > prob_matrix_t;
typedef std::vector<prob_matrix_t> td_matrix_t;

//
class Document {
public:
	Document() :
			lines_(0) {
	}

	explicit Document(const std::string& file_path) :
			lines_(0), file_path_(file_path) {
	}

	void SetFilePath(const std::string& file_path) {
		file_path_ = file_path;
	}

	void Init() {
		//
		words_.clear();
		lines_ = 0;

		std::ifstream fin(file_path_.c_str());
		std::string s;
		std::vector<std::string> words;
		while (std::getline(fin, s)) {
			//
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
		srand((unsigned) time(0));
	}

	void InitDocAndVoc(const std::string& doc_dir_path) {
		namespace fs = boost::filesystem;
		// init document
		fs::directory_iterator end;
		Document doc;
		for (fs::directory_iterator pos(doc_dir_path); pos != end; ++pos) {
			doc.SetFilePath(pos->path().string());
			doc.Init();
			documents_.push_back(doc);
		}

		// init vocabulary
		int index = 0;
		BOOST_FOREACH(Document & doc, documents_) {
			BOOST_FOREACH(const std::string& word, doc.GetWords()) {
				word_map_t::const_iterator word_iter = vocabulary_.find(word);
				if (word_iter == vocabulary_.end()) {
					vocabulary_.insert(std::make_pair(word, index++));
				}
			}
		}
	}

	void BuildTermDocMatrix() {
		for (int d = 0; d < documents_.size(); ++d) {
			std::vector<int> term_count(vocabulary_.size());
			BOOST_FOREACH(const std::string & word, documents_[d].GetWords()) {
				word_map_t::const_iterator word_iter = vocabulary_.find(word);
				if (word_iter == vocabulary_.end()) {
					continue;
				}
				int w_index = word_iter->second;
				++term_count[w_index];
			}
			term_doc_matrix_.push_back(term_count);
		}
	}

	void CreateCounterArrays() {
		for (int d = 0; d < documents_.size(); ++d) {
			std::vector<double> doc_topic(number_of_topics_);
			for (int z = 0; z < number_of_topics_; ++z) { // random init
				doc_topic[z] = GenRandomNum();
			}
			Normalize(doc_topic);
			document_topic_prob_.push_back(doc_topic);

			prob_matrix_t voc(vocabulary_.size());
			std::vector<double> topic(number_of_topics_);
			for (int w = 0; w < vocabulary_.size(); ++w) {
				voc.push_back(topic);
			}
			topic_prob_.push_back(voc);
		}

		for (int z = 0; z < number_of_topics_; ++z) {
			std::vector<double> topic_word(vocabulary_.size());
			for (int w = 0; w < vocabulary_.size(); ++w) { // random init
				topic_word[w] = GenRandomNum();
			}
			Normalize(topic_word);
			topic_word_prob_.push_back(topic_word);
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
		std::vector<double> prob(number_of_topics_);
		for (int i = 0; i < max_iter_; ++i) {
			// E STEP
			for (int d = 0; d < documents_.size(); ++d) {
				for (int w = 0; w < vocabulary_.size(); ++w) {
					for (int z = 0; z < number_of_topics_; ++z) {
						prob[z] = document_topic_prob_[d][z] * topic_word_prob_[z][w];
					}
					Normalize(prob);
					topic_prob_[d][w] = prob;
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

	void PrintProb(int max_len) {

	}

	void Run(const std::string& doc_dir_path) {
		InitDocAndVoc(doc_dir_path);
		BuildTermDocMatrix();
		CreateCounterArrays();
		EM();
	}
private:
	int number_of_topics_;
	int max_iter_;
	std::vector<Document> documents_;
	word_map_t vocabulary_;
	std::vector<std::vector<int> > term_doc_matrix_;

	prob_matrix_t document_topic_prob_; // p(z | d)
	prob_matrix_t topic_word_prob_; // p(w | z)

	td_matrix_t topic_prob_; // p(z | d, w)
};

#endif

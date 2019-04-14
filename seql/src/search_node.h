/**
 *   \file SNode.h
 *   \brief Class for SNodes in the searchtree for SEQL
 *
 *  Class for SNodes in the search tree of SEQL
 *
 *
 */
#ifndef SNODE_H
#define SNODE_H

#include <algorithm>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using token_type = std::string;
using input_mat = std::vector<std::vector<token_type>>;

class SNode {
  private:
    // Last docid.
    int last_docid = -1;

    // Pointer to previous ngram.
    SNode *prev = nullptr;

  public:
    static bool use_char_token;
    static int totalWildcardLimit;
    static int consecWildcardLimit;
    static bool hasWildcardConstraints;
    static std::set<std::string> single_node_minsup_cache;
    static unsigned int minsup;
    static int verbosity;
    static bool tilde;

    bool has_no_childs = false;
    bool is_shrunk = false;
    // Unigram of the node
    std::string ne;

    std::string ngram;
    // Total list of occurrences in the entire collection for an
    // ngram. A sort of expanded inverted index.
    std::vector<int> loc;

    // Vector of ngrams which are extensions of current ngram.
    std::vector<std::unique_ptr<SNode>> next;

    // Shrink the list of total occurrences to contain just support
    // doc_ids, instead of doc_ids and occurences.
    void shrink();

    // Return the support of current ngram.
    // Simply count the negative loc as doc_ids.
    unsigned int support() const;

    // Returns the full ngram of which this node represents
    std::string getNgram();

    // Set up of the wildcard constraints
    // there are two types of wildcard constraint:
    // 1. total wildcard limit
    // 2. number of consecutive wildcards
    // The rules for setup are the following:
    // If both are zero => no constraints
    // total limit set but no consecutive limit set => consecutive limit set to
    // total limit consecutive limit set but no total limit set => total limit
    // set to max int. consecutive limit greater than total limit => consecutive
    // limit set to total limit
    static void setupWildcardConstraint(int _totalWildcardLimit,
                                        int _consecWildcardLimit);

    // checks if this ngram violates any of the wildcard constraints,
    bool violateWildcardConstraint();

    // Add a doc_id and position of occurrence to the list of occurrences,
    // for this ngram.
    // Encode the doc_id in the vector of locations.
    // Negative entry means new doc_id.
    void add(unsigned int docid, int pos);
    void expand_node(input_mat const &x);
    void collect_candidates(input_mat const &x,
                            std::map<token_type, SNode> &candidates);
};

void deleteUndersupportedUnigrams(std::map<std::string, SNode> &seed);
std::map<std::string, SNode> prepareInvertedIndex(input_mat const &x);

#endif

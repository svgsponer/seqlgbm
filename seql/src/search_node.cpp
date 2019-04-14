/**
 * Author: Severin Gsponer (svgsponer@gmail.com)
 *
 * SNode: represents a node in a searchtree for SEQL
 *
 * License:
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation.
 *
 */

#include "search_node.h"

void SNode::shrink() {
    std::vector<int> tmp;

    for (auto const &currLoc : loc) {
        if (currLoc < 0) {
            tmp.push_back(currLoc);
        }
    }
    loc = std::move(tmp);
    loc.shrink_to_fit();

    // Does not shrink the capacity of the location erase remove idome
    // loc.erase( std::remove_if(loc.begin(), loc.end(), [](int i){return i >=
    // // 0;}),loc.end());
    last_docid = -1;
    is_shrunk  = true;
}

unsigned int SNode::support() const {
    return std::count_if(std::begin(loc), std::end(loc), [](int currLoc) {
        return currLoc < 0;
    });
}

std::string SNode::getNgram() { return ngram; }

bool SNode::violateWildcardConstraint() {
    int numberOfWildcards       = 0;
    int numberOfConsecWildcards = 0;

    for (SNode *t = this; t != nullptr; t = t->prev) {
        if (t->ne.compare("*") == 0) {
            numberOfWildcards++;
            numberOfConsecWildcards++;
            if (numberOfWildcards > totalWildcardLimit) {
                return true;
            }
        } else {
            if (numberOfConsecWildcards > consecWildcardLimit) {
                return true;
            }
            numberOfConsecWildcards = 0;
        }
    }
    return false;
}

void SNode::setupWildcardConstraint(int _totalWildcardLimit,
                                    int _consecWildcardLimit) {
    if (_totalWildcardLimit == 0) {
        if (_consecWildcardLimit == 0) {
            hasWildcardConstraints = false;
        } else {
            hasWildcardConstraints = true;
            consecWildcardLimit    = _consecWildcardLimit;
            totalWildcardLimit     = std::numeric_limits<int>::max();
        }
    } else {
        hasWildcardConstraints = true;
        if (_consecWildcardLimit == 0 ||
            _consecWildcardLimit > _totalWildcardLimit) {
            totalWildcardLimit  = _totalWildcardLimit;
            consecWildcardLimit = totalWildcardLimit;
        } else {
            totalWildcardLimit  = _totalWildcardLimit;
            consecWildcardLimit = _consecWildcardLimit;
        }
    }
}

void SNode::add(unsigned int docid, int pos) {
    if (last_docid != static_cast<int>(docid)) {
        loc.push_back(-static_cast<int>(docid + 1));
    }
    loc.push_back(pos);
    last_docid = static_cast<int>(docid);
}

void SNode::expand_node(input_mat const &x) {
    std::map<std::string, SNode> candidates;
    collect_candidates(x, candidates);

    shrink();

    if (candidates.empty()) {
        has_no_childs = true;
    } else {
        next.clear();
        next.reserve(candidates.size());
        // Prepare the candidate extensions.
        for (auto const &currCandidate : candidates) {
            std::unique_ptr<SNode> c(new SNode());
            c->loc = currCandidate.second.loc;
            c->ne  = currCandidate.first;
            if (SNode::use_char_token) {
                c->ngram = getNgram() + currCandidate.first;
            } else {
                c->ngram = getNgram() + " " + currCandidate.first;
            }
            c->prev = this;
            c->next.clear();
            // Keep all the extensions of the current feature for future
            // iterations. If we need to save memory we could sacrifice this
            // storage.
            next.push_back(std::move(c));
        }
    }
    // Adjust capacity of next vector
    // std::vector<std::unique_ptr<SNode>>(next).swap (next);
}

void SNode::collect_candidates(input_mat const &x,
                               std::map<std::string, SNode> &candidates) {
    unsigned int docid = 0;
    for (const auto currLoc : loc) {
        if (currLoc < 0) {
            // currLoc indicates a new document
            docid = static_cast<unsigned int>(-currLoc) - 1;
        } else {
            // currLoc points to a token
            const auto next_pos = currLoc + 1;

            if (next_pos < x[docid].size()) {
                const std::string next_unigram = x[docid][next_pos];
                if (next_unigram == " ") {
                    continue;
                }
                if (minsup == 1 ||
                    single_node_minsup_cache.find(next_unigram) !=
                        single_node_minsup_cache.end()) {
                    candidates[next_unigram].add(docid, next_pos);

                    if (hasWildcardConstraints) {
                        // We treat a gap as an additional unigram "*".
                        candidates["*"].add(docid, next_pos);
                    }
                }
            }
        }
    }
}

std::map<std::string, SNode> prepareInvertedIndex(input_mat const &x) {

    std::map<std::string, SNode> seed;

    if (SNode::verbosity >= 1) {
        std::cout << "\nprepare inverted index for unigrams\n";
    }
    for (unsigned int docid = 0; docid < x.size(); ++docid) {
        for (unsigned int pos = 0; pos < x[docid].size(); ++pos) {
            auto unigram = x[docid][pos];
            if (unigram == " ") {
                continue;
            }
            seed[unigram].add(docid, pos);
        }
    }

    for (auto &node : seed) {
        node.second.next.clear();
        node.second.ne    = node.first;
        node.second.ngram = node.first;
    }
    return seed;
}

void deleteUndersupportedUnigrams(std::map<std::string, SNode> &seed) {
    // Keep only unigrams above minsup threshold.
    for (auto it = seed.cbegin(); it != seed.cend();) {
        if (it->second.support() < SNode::minsup) {
            if (SNode::verbosity >= 1) {
                std::cout << "\nremove unigram (minsup):" << it->first;
                std::cout.flush();
            }
            seed.erase(it++);
        } else {
            SNode::single_node_minsup_cache.insert(it->second.ne);
            ++it;
        }
    }

    if (SNode::single_node_minsup_cache.size() == 0) {
        std::cout << "\n>>> NO UNIGRAM LEFT\nMaybe adjust the minsup parameter";
        std::exit(1);
    };

    if (SNode::verbosity >= 1) {
        std::cout << "\ndistinct unigram:" << '\n';
        for (const auto &it : seed) {
            std::cout << it.first << '\n';
        }
    }
}

bool SNode::use_char_token         = true;
bool SNode::hasWildcardConstraints = true;
int SNode::totalWildcardLimit      = 0;
int SNode::consecWildcardLimit     = 0;
unsigned int SNode::minsup         = 1;
int SNode::verbosity               = 1;
bool SNode::tilde                  = false;
std::set<std::string> SNode::single_node_minsup_cache;

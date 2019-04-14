#include "../src/seql_learn.h"
#include <catch2/catch.hpp>

// TEST_CASE("mytest","[bla]"){
//     std::string bla = "ABC";
//     size_t found = bla.find_first_not_of(" ", 13);
//     REQUIRE(found == std::string::npos);
// }

TEST_CASE("SNodes without wildcards are expandable",
          "[representation][noGap]") {
    const std::vector<std::vector<std::string>> transaction = {
        {"A", "B", "C", "A", "B", "C"}, {"B", "C", "A", "B", "B", "A"}};
    SNode node;
    SNode::hasWildcardConstraints = false;
    SNode::minsup = 1;
    SNode::use_char_token = true;

    SECTION("potential candidates can be found") {
        node.ngram = "A";
        node.ne = "A";
        node.loc = {-1, 0, 3, -2, 2, 5};
        REQUIRE(node.loc.size() == 6);
        std::map<std::string, SNode> candidates;
        node.collect_candidates(transaction, candidates);
        REQUIRE(candidates.count("A") == 0);
        REQUIRE(candidates.count("B") == 1);
        REQUIRE(candidates.count("C") == 0);
        REQUIRE(candidates.count("*") == 0);
        REQUIRE(candidates.size() == 1);
        std::vector<int> B_loc = {-1, 1, 4, -2, 3};
        REQUIRE(candidates["B"].loc == B_loc);
    }

    SECTION("potential candidates can be found") {
        node.ngram = "B";
        node.ne = "B";
        node.loc = {-1, 1, 4, -2, 0, 3, 4};
        REQUIRE(node.loc.size() == 7);
        std::map<std::string, SNode> candidates;
        node.collect_candidates(transaction, candidates);
        REQUIRE(candidates.count("A") == 1);
        REQUIRE(candidates.count("B") == 1);
        REQUIRE(candidates.count("C") == 1);
        REQUIRE(candidates.count("*") == 0);
        REQUIRE(candidates.size() == 3);
    }

    SECTION("potential candidates can be found") {
        node.ngram = "C";
        node.ne = "C";
        node.loc = {-1, 2, 5, -2, 1};
        REQUIRE(node.loc.size() == 5);
        std::map<std::string, SNode> candidates;
        node.collect_candidates(transaction, candidates);
        REQUIRE(candidates.count("A") == 1);
        REQUIRE(candidates.count("B") == 0);
        REQUIRE(candidates.count("C") == 0);
        REQUIRE(candidates.count("*") == 0);
        REQUIRE(candidates.size() == 1);
    }
    SECTION("potential candidates can be found") {
        node.ngram = "BB";
        node.ne = "B";
        node.loc = {-2, 4};
        REQUIRE(node.loc.size() == 2);
        std::map<std::string, SNode> candidates;
        node.collect_candidates(transaction, candidates);
        REQUIRE(candidates.count("A") == 1);
        REQUIRE(candidates.count("B") == 0);
        REQUIRE(candidates.count("C") == 0);
        REQUIRE(candidates.count("*") == 0);
        REQUIRE(candidates.size() == 1);
    }
}

TEST_CASE("SNodes with wildcards are expandable overlapping case",
          "[representation][noGap]") {
    const std::vector<std::vector<std::string>> transaction = {
        {"A", "A", "A", "A", "B", "C"}, {"B", "C", "A", "B", "B", "A"}};
    SNode node;
    SNode::hasWildcardConstraints = true;
    SNode::minsup = 1;
    SNode::use_char_token = true;

    SECTION("wildcard expansion is correct") {
        node.ngram = "A";
        node.ne = "A";
        node.loc = {-1, 0, 1, 2, 3, -2, 2, 5};
        std::vector<int> gold_loc_A = {-1, 1, 2, 3};
        std::vector<int> gold_loc_B = {-1, 4, -2, 3};
        std::vector<int> gold_loc_star = {-1, 1, 2, 3, 4, -2, 3};
        REQUIRE(node.loc.size() == 8);
        std::map<std::string, SNode> candidates;
        node.collect_candidates(transaction, candidates);
        REQUIRE(candidates.count("A") == 1);
        REQUIRE(candidates.count("B") == 1);
        REQUIRE(candidates.count("C") == 0);
        REQUIRE(candidates.count("*") == 1);
        REQUIRE(candidates["A"].loc == gold_loc_A);
        REQUIRE(candidates["B"].loc == gold_loc_B);
        REQUIRE(candidates["*"].loc == gold_loc_star);
        REQUIRE(candidates.size() == 3);
    }

    SECTION("wildcard expansion is correct") {
        node.ngram = "A*";
        node.ne = "*";
        node.loc = {-1, 1, 2, 3, 4, -2, 3};
        std::vector<int> gold_loc_A = {-1, 2, 3};
        std::vector<int> gold_loc_B = {-1, 4, -2, 4};
        std::vector<int> gold_loc_star = {-1, 2, 3, 4, 5, -2, 4};
        REQUIRE(node.loc.size() == 7);
        std::map<std::string, SNode> candidates;
        node.collect_candidates(transaction, candidates);
        REQUIRE(candidates.count("A") == 1);
        REQUIRE(candidates.count("B") == 1);
        REQUIRE(candidates.count("C") == 1);
        REQUIRE(candidates.count("*") == 1);
        REQUIRE(candidates["A"].loc == gold_loc_A);
        REQUIRE(candidates["B"].loc == gold_loc_B);
        REQUIRE(candidates["*"].loc == gold_loc_star);
        REQUIRE(candidates.size() == 4);
    }
}

TEST_CASE("SNodes with wildcards are expandable", "[representation][noGap]") {
    const std::vector<std::vector<std::string>> transaction = {
        {"A", "B", "C", "A", "B", "C"}, {"B", "C", "A", "B", "B", "A"}};
    SNode node;
    SNode::hasWildcardConstraints = true;
    SNode::minsup = 1;
    SNode::use_char_token = true;

    SECTION("wildcard expansion is correct") {
        node.ngram = "A";
        node.ne = "A";
        node.loc = {-1, 0, 3, -2, 2, 5};
        std::vector<int> gold_loc = {-1, 1, 4, -2, 3};
        REQUIRE(node.loc.size() == 6);
        std::map<std::string, SNode> candidates;
        node.collect_candidates(transaction, candidates);
        REQUIRE(candidates.count("A") == 0);
        REQUIRE(candidates.count("B") == 1);
        REQUIRE(candidates.count("C") == 0);
        REQUIRE(candidates.count("*") == 1);
        REQUIRE(candidates["*"].loc == gold_loc);
        REQUIRE(candidates.size() == 2);
    }

    SECTION("wildcard expansion is correct") {
        node.ngram = "A*";
        node.ne = "*";
        node.loc = {-1, 1, 4, -2, 3};
        std::vector<int> gold_locB = {-2, 4};
        std::vector<int> gold_loc_star = {-1, 2, 5, -2, 4};
        REQUIRE(node.loc.size() == 5);
        std::map<std::string, SNode> candidates;
        node.collect_candidates(transaction, candidates);
        REQUIRE(candidates.count("A") == 0);
        REQUIRE(candidates.count("B") == 1);
        REQUIRE(candidates.count("C") == 1);
        REQUIRE(candidates.count("*") == 1);
        REQUIRE(candidates["B"].loc == gold_locB);
        REQUIRE(candidates["*"].loc == gold_loc_star);
        REQUIRE(candidates.size() == 3);
    }

    SECTION("correct token get found") {
        node.ngram = "A";
        node.ne = "A";
        node.loc = {-1, 0, 3, -2, 2, 5};
        REQUIRE(node.loc.size() == 6);
        std::map<std::string, SNode> candidates;
        node.collect_candidates(transaction, candidates);
        REQUIRE(candidates.count("A") == 0);
        REQUIRE(candidates.count("B") == 1);
        REQUIRE(candidates.count("C") == 0);
        REQUIRE(candidates.count("*") == 1);
        REQUIRE(candidates.size() == 2);
    }

    SECTION("candidates have correct loc vector") {
        node.ngram = "A";
        node.ne = "A";
        node.loc = {-1, 0, 3, -2, 2, 5};
        REQUIRE(node.loc.size() == 6);
        std::map<std::string, SNode> candidates;
        node.collect_candidates(transaction, candidates);
        std::vector<int> B_loc = {-1, 1, 4, -2, 3};
        REQUIRE(candidates["B"].loc == B_loc);
    }

    SECTION("No wildcard will be inserted at the end of the a doc") {
        node.ngram = "A";
        node.ne = "A";
        node.loc = {-1, 0, 3, -2, 2, 5};
        std::vector<int> star_loc = {-1, 1, 4, -2, 3};
        std::map<std::string, SNode> candidates;
        node.collect_candidates(transaction, candidates);
        REQUIRE(candidates["*"].loc == star_loc);
    }

    SECTION("potential candidates can be found") {
        node.ngram = "B";
        node.ne = "B";
        node.loc = {-1, 1, 4, -2, 0, 3, 4};
        REQUIRE(node.loc.size() == 7);
        std::map<std::string, SNode> candidates;
        node.collect_candidates(transaction, candidates);
        REQUIRE(candidates.count("A") == 1);
        REQUIRE(candidates.count("B") == 1);
        REQUIRE(candidates.count("C") == 1);
        REQUIRE(candidates.count("*") == 1);
        REQUIRE(candidates.size() == 4);
    }

    SECTION("potential candidates can be found") {
        node.ngram = "C";
        node.ne = "C";
        node.loc = {-1, 2, 5, -2, 1};
        REQUIRE(node.loc.size() == 5);
        std::map<std::string, SNode> candidates;
        node.collect_candidates(transaction, candidates);
        REQUIRE(candidates.count("A") == 1);
        REQUIRE(candidates.count("B") == 0);
        REQUIRE(candidates.count("C") == 0);
        REQUIRE(candidates.count("*") == 1);
        REQUIRE(candidates.size() == 2);
    }
    SECTION("expansion of 2-mer are working") {
        node.ngram = "BB";
        node.ne = "B";
        node.loc = {-2, 4};
        REQUIRE(node.loc.size() == 2);
        std::map<std::string, SNode> candidates;
        node.collect_candidates(transaction, candidates);
        REQUIRE(candidates.count("A") == 1);
        REQUIRE(candidates.count("B") == 0);
        REQUIRE(candidates.count("C") == 0);
        REQUIRE(candidates.count("*") == 1);
        REQUIRE(candidates.size() == 2);
    }
}

TEST_CASE("SNodes for word token without wildcards are expandable",
          "[representation][noGap]") {
    const std::vector<std::vector<std::string>> transaction = {
        {"AB", "A", "BC"}, {"BC", "ABB", "AB", "AA"}};
    SNode node;
    SNode::hasWildcardConstraints = false;
    SNode::minsup = 1;
    SNode::use_char_token = false;
    SECTION("potential candidates can be found") {
        node.ngram = "AB";
        node.ne = "AB";
        node.loc = {-1, 0, -2, 2};
        std::map<std::string, SNode> candidates;
        node.collect_candidates(transaction, candidates);
        REQUIRE(candidates.count("A") == 1);
        REQUIRE(candidates.count("B") == 0);
        REQUIRE(candidates.count("AA") == 1);
        REQUIRE(candidates.count("ABB") == 0);
        REQUIRE(candidates.count("*") == 0);
        REQUIRE(candidates.size() == 2);
    }

    SECTION("potential candidates can be found") {
        node.ngram = "A";
        node.ne = "A";
        node.loc = {-1, 1};
        std::map<std::string, SNode> candidates;
        node.collect_candidates(transaction, candidates);
        REQUIRE(candidates.count("A") == 0);
        REQUIRE(candidates.count("BC") == 1);
        REQUIRE(candidates.count("AA") == 0);
        REQUIRE(candidates.count("ABB") == 0);
        REQUIRE(candidates.count("*") == 0);
        REQUIRE(candidates.size() == 1);
    }
}

TEST_CASE("SNodes with wildcards are expandable overlapping case wordtoken",
          "[representation][noGap]") {
    const std::vector<std::vector<std::string>> transaction = {
        {"AB", "AB", "AB", "A", "BC"}, {"BC", "ABB", "AB", "AA"}};
    SNode node;
    SNode::hasWildcardConstraints = true;
    SNode::minsup = 1;
    SNode::use_char_token = false;

    SECTION("wildcard expansion is correct") {
        node.ngram = "AB";
        node.ne = "AB";
        node.loc = {-1, 0, 1, 2, -2, 2};
        std::vector<int> gold_loc_AB = {-1, 1, 2};
        std::vector<int> gold_loc_A = {-1, 3};
        std::vector<int> gold_loc_AA = {-2, 3};
        std::vector<int> gold_loc_star = {-1, 1, 2, 3, -2, 3};
        REQUIRE(node.loc.size() == 6);
        std::map<std::string, SNode> candidates;
        node.collect_candidates(transaction, candidates);
        REQUIRE(candidates.count("AB") == 1);
        REQUIRE(candidates.count("A") == 1);
        REQUIRE(candidates.count("BC") == 0);
        REQUIRE(candidates.count("BA") == 0);
        REQUIRE(candidates.count("*") == 1);
        REQUIRE(candidates["AB"].loc == gold_loc_AB);
        REQUIRE(candidates["A"].loc == gold_loc_A);
        REQUIRE(candidates["AA"].loc == gold_loc_AA);
        REQUIRE(candidates["*"].loc == gold_loc_star);
        REQUIRE(candidates.size() == 4);
    }

    SECTION("wildcard expansion is correct") {
        node.ngram = "AB *";
        node.ne = "*";
        node.loc = {-1, 1, 2, 3, -2, 3};
        std::vector<int> gold_loc_AB = {-1, 2};
        std::vector<int> gold_loc_A = {-1, 3};
        std::vector<int> gold_loc_BC = {-1, 4};
        std::vector<int> gold_loc_star = {-1, 2, 3, 4};
        REQUIRE(node.loc.size() == 6);
        std::map<std::string, SNode> candidates;
        node.collect_candidates(transaction, candidates);
        REQUIRE(candidates.count("AA") == 0);
        REQUIRE(candidates.count("BC") == 1);
        REQUIRE(candidates.count("A") == 1);
        REQUIRE(candidates.count("BA") == 0);
        REQUIRE(candidates.count("AB") == 1);
        REQUIRE(candidates.count("*") == 1);
        REQUIRE(candidates["AB"].loc == gold_loc_AB);
        REQUIRE(candidates["A"].loc == gold_loc_A);
        REQUIRE(candidates["BC"].loc == gold_loc_BC);
        REQUIRE(candidates["*"].loc == gold_loc_star);
        REQUIRE(candidates.size() == 4);
    }
}

TEST_CASE("Substring chartoken representation ", "") {
    const std::vector<std::vector<std::string>> docs = {
        {"A", "B", "C", "A", "B", "C"}, {"B", "C", "A", "B", "B", "A"}};
    SNode::use_char_token = true;
    SNode::verbosity = 0;
    SECTION("generate seed works") {
        std::map<std::string, SNode> cor_seed;
        cor_seed["A"].loc = {-1, 0, 3, -2, 2, 5};
        cor_seed["B"].loc = {-1, 1, 4, -2, 0, 3, 4};
        cor_seed["C"].loc = {-1, 2, 5, -2, 1};
        auto seed = prepareInvertedIndex(docs);
        REQUIRE(seed.size() == 3);
        REQUIRE(cor_seed["A"].loc == seed["A"].loc);
        REQUIRE("A" == seed["A"].ne);
        REQUIRE("A" == seed["A"].ngram);
        REQUIRE(cor_seed["B"].loc == seed["B"].loc);
        REQUIRE("B" == seed["B"].ne);
        REQUIRE("B" == seed["B"].ngram);
        REQUIRE(cor_seed["C"].loc == seed["C"].loc);
        REQUIRE("C" == seed["C"].ne);
        REQUIRE("C" == seed["C"].ngram);
    }
}

TEST_CASE("Substring wordtoken representation ", "") {
    SNode::use_char_token = false;
    SNode::verbosity = 0;
    SECTION("generate seed works") {
        const std::vector<std::vector<std::string>> docs = {{"AB", "C", "ABC"},
                                                            {"C", "AB", "BA"}};
        std::map<std::string, SNode> cor_seed;
        cor_seed["AB"].loc = {-1, 0, -2, 1};
        cor_seed["ABC"].loc = {-1, 2};
        cor_seed["BA"].loc = {-2, 2};
        cor_seed["C"].loc = {-1, 1, -2, 0};
        auto seed = prepareInvertedIndex(docs);
        // REQUIRE(seed.size() == 4);
        REQUIRE(cor_seed["AB"].loc == seed["AB"].loc);
        REQUIRE("AB" == seed["AB"].ne);
        REQUIRE("AB" == seed["AB"].ngram);
        REQUIRE(cor_seed["ABC"].loc == seed["ABC"].loc);
        REQUIRE("ABC" == seed["ABC"].ne);
        REQUIRE("ABC" == seed["ABC"].ngram);
        REQUIRE(cor_seed["BA"].loc == seed["BA"].loc);
        REQUIRE("BA" == seed["BA"].ne);
        REQUIRE("BA" == seed["BA"].ngram);
        REQUIRE(cor_seed["C"].loc == seed["C"].loc);
        REQUIRE("C" == seed["C"].ne);
        REQUIRE("C" == seed["C"].ngram);
    }
}

#include "../src/search_node.h"
#include <catch2/catch.hpp>

TEST_CASE("Creation of inverted Index", "") {
    SNode::tilde = false;
    std::vector<std::vector<std::string>> x = {{"A", "B", "C", "D"},
                                               {"B", "C", "D", "A"}};
    std::map<std::string, SNode> seed = prepareInvertedIndex(x);

    SECTION("size of map is correct") { REQUIRE(seed.size() == 4); }
    SECTION("All unigrams are there") {
        REQUIRE(seed.find("A") != std::end(seed));
        REQUIRE(seed.find("B") != std::end(seed));
        REQUIRE(seed.find("C") != std::end(seed));
        REQUIRE(seed.find("D") != std::end(seed));
    }
    SECTION("Nodes have correct unigram stored") {
        REQUIRE(seed["A"].ne == "A");
        REQUIRE(seed["B"].ne == "B");
        REQUIRE(seed["C"].ne == "C");
        REQUIRE(seed["D"].ne == "D");
    }
}

#include "../src/linear_model.h"
#include <catch2/catch.hpp>

TEST_CASE("Correct scores are calculated") {
    SECTION("Score is correct if match is found") {
        std::vector<std::string> input = {"A", "B", "C", "A", "A"};
        SEQL::LinearModel model;
        model.use_char_token = true;
        model.insert_or_add("AAAA", 1.1);
        model.build_tree();

        auto score = model.predict(input);
        REQUIRE(score == 0);
    }

    SECTION("Score is correctly alculated") {
        std::vector<std::string> input = {"A", "B", "C", "A", "A"};
        SEQL::LinearModel model;
        model.use_char_token = true;
        model.insert_or_add("AB", 1.1);
        model.insert_or_add("BCA", 2.2);
        model.insert_or_add("AA", 3.3);
        model.insert_or_add("ACB", 4.4);
        model.build_tree();

        auto score = model.predict(input);
        REQUIRE(score == Approx(6.6));
    }

    SECTION("Score is correctly calculated including INTERCEPT") {
        std::vector<std::string> input = {"A", "B", "C", "A", "A"};
        std::unordered_map<std::string, double> m;
        SEQL::insert_or_add(m, "*0*", 10);
        SEQL::insert_or_add(m, "AB", 1.1);
        SEQL::insert_or_add(m, "BCA", 2.2);
        SEQL::insert_or_add(m, "AA", 3.3);
        SEQL::insert_or_add(m, "ACB", 4.4);
        SEQL::LinearModel model(m, 1);
        model.use_char_token = true;
        model.seperate_sf();
        model.build_tree();

        auto score = model.predict({1}, input);
        REQUIRE(score == Approx(16.6));
    }

    SECTION("Score with wildcards are correctly calculated") {
        std::vector<std::string> input = {"A", "B", "C", "A", "A"};
        SEQL::LinearModel model;
        model.use_char_token = true;
        model.insert_or_add("AB", 1.1);
        model.insert_or_add("BCA", 2.2);
        model.insert_or_add("AA", 3.3);
        model.insert_or_add("A*B", 4.4);
        model.build_tree();

        auto score = model.predict(input);
        REQUIRE(score == Approx(6.6));
    }

    SECTION("Scores will be correctly computed, word tokens") {
        std::vector<std::string> input = {"AA", "BB", "CC", "AB", "AA", "CB"};
        SEQL::LinearModel model;
        model.use_char_token = false;
        model.insert_or_add("AB AA", 1.1);
        model.insert_or_add("AA", 2.2);
        model.insert_or_add("CC AA AA", 3.3);
        model.build_tree();

        auto score = model.predict(input);
        REQUIRE(score == Approx(3.3));
    }

    SECTION("Scores will be correctly computed, word tokens with INTERCEPT") {
        std::vector<std::string> input = {"AA", "BB", "CC", "AB", "AA", "CB"};
        std::unordered_map<std::string, double> m;
        SEQL::insert_or_add(m, "*0*", 10);
        SEQL::insert_or_add(m, "AB AA", 1.1);
        SEQL::insert_or_add(m, "AA", 2.2);
        SEQL::insert_or_add(m, "CC AA AA", 3.3);

        SEQL::LinearModel model(m, 1);
        model.use_char_token = false;
        model.seperate_sf();
        model.build_tree();

        auto score = model.predict({1}, input);
        REQUIRE(score == Approx(13.3));
    }

    SECTION("Scores with wildcard will be correctly computed, word tokens") {
        std::vector<std::string> input = {"AA", "BB", "CC", "AB", "AA", "CB"};
        SEQL::LinearModel model;
        model.use_char_token = false;
        model.insert_or_add("AB AA", 1.1);
        model.insert_or_add("AA", 2.2);
        model.insert_or_add("CC * AA", 3.3);
        model.build_tree();

        auto score = model.predict(input);
        REQUIRE(score == Approx(6.6));
    }
}

TEST_CASE("Correct pattern are found", "[prediction]") {

    SECTION("Simple patterns will be found") {
        std::vector<std::string> input = {"A", "B", "C", "A", "A"};
        SEQL::LinearModel model;
        model.use_char_token = true;
        model.insert_or_add("AB", 1);
        model.insert_or_add("BCA", 2.1);
        model.insert_or_add("AA", 3.2);
        model.insert_or_add("ACB", 4.3);
        model.build_tree();
        auto matches = model.find_matches(input);
        CHECK(matches.size() == 3);
        REQUIRE(matches.count("AA") == 1);
        REQUIRE(model["AA"] == 3.2);
        REQUIRE(matches.count("AB") == 1);
        REQUIRE(model["AB"] == 1);
        REQUIRE(matches.count("BCA") == 1);
        REQUIRE(model["BCA"] == 2.1);
    }

    SECTION("Simple patterns with wildcards will be found") {
        std::vector<std::string> input = {"A", "B", "C", "A", "A", "C"};
        SEQL::LinearModel model;
        model.use_char_token = true;
        model.insert_or_add("AB", 1);
        model.insert_or_add("AA", 2);
        model.insert_or_add("CC", 3);
        model.insert_or_add("A*C", 4.4);
        model.build_tree();
        auto matches = model.find_matches(input);
        REQUIRE(matches.size() == 3);
        REQUIRE(matches.count("A*C") == 1);
        REQUIRE(model["A*C"] == 4.4);
        REQUIRE(matches.count("AA") == 1);
        REQUIRE(model["AA"] == 2);
        REQUIRE(matches.count("AB") == 1);
        REQUIRE(model["AB"] == 1);
    }

    SECTION("Paterns will only be counted once") {
        std::vector<std::string> input = {"A", "B", "C", "A", "B", "C"};
        SEQL::LinearModel model;
        model.use_char_token = true;
        model.insert_or_add("AB", 1);
        model.insert_or_add("AA", 2);
        model.insert_or_add("CC", 3);
        model.insert_or_add("BC", 4);
        model.build_tree();
        auto matches = model.find_matches(input);
        REQUIRE(matches.size() == 2);
        REQUIRE(matches.count("AB") == 1);
        REQUIRE(model["AB"] == 1);
        REQUIRE(matches.count("BC") == 1);
        REQUIRE(model["BC"] == 4);
    }

    SECTION("Simple patterns will be found, word tokens") {
        std::vector<std::string> input = {"AA", "BB", "CC", "AB", "AA", "CB"};
        SEQL::LinearModel model;
        model.use_char_token = false;
        model.insert_or_add("AB AA", 1);
        model.insert_or_add("AA", 2);
        model.insert_or_add("CC AB BB", 3);
        model.build_tree();
        auto matches = model.find_matches(input);
        REQUIRE(matches.size() == 2);
        REQUIRE(matches.count("AA") == 1);
        REQUIRE(model["AA"] == 2);
        REQUIRE(matches.count("AB AA") == 1);
        REQUIRE(model["AB AA"] == 1);
    }

    SECTION("Patterns with wildcard will be found, word tokens") {
        std::vector<std::string> input = {"AA", "BB", "CC", "AB", "AA", "CB"};
        SEQL::LinearModel model;
        model.use_char_token = false;
        model.insert_or_add("AB AA", 1);
        model.insert_or_add("AA", 2);
        model.insert_or_add("CC * AA", 3);
        model.build_tree();
        auto matches = model.find_matches(input);
        REQUIRE(matches.size() == 3);
        REQUIRE(matches.count("AA") == 1);
        REQUIRE(model["AA"] == 2);
        REQUIRE(matches.count("AB AA") == 1);
        REQUIRE(model["AB AA"] == 1);
        REQUIRE(matches.count("CC * AA") == 1);
        REQUIRE(model["CC * AA"] == 3);
    }
}

TEST_CASE("Static feature prediction works") {
    std::unordered_map<std::string, double> list{{"*1*", 2}, {"*2*", -3}};
    SEQL::LinearModel model(list, true);
    SEQL::Data d;
    auto pred = model.predict({1.5, 2.5, 3.5, 99}, {"A"});
    REQUIRE(pred == Approx(-5.5));
}

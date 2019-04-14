#include <catch2/catch.hpp>

#include "../src/evaluation.h"

TEST_CASE("Mean Squared error of two vectors", "[MSE]") {
    using SEQL::Eval::mse;
    SECTION("mse of two 0 vector is 0") {
        std::vector<double> a = {0, 0, 0};
        std::vector<double> b = {0, 0, 0};
        REQUIRE(mse(a, b) == 0);
    }
    SECTION("mse of equal vector is 0") {
        std::vector<double> a = {1, 1, 1};
        std::vector<double> b = {1, 1, 1};
        REQUIRE(mse(a, b) == 0);
    }
    SECTION("mse of (0, 1, 2) and (2, 3, 4) is 4") {
        std::vector<double> a = {0, 1, 2};
        std::vector<double> b = {2, 3, 4};
        REQUIRE(mse(a, b) == 4);
    }
    SECTION("mse works with doubles") {
        std::vector<double> a = {0, 1.5, 2.5};
        std::vector<double> b = {2, 3.5, 4};
        REQUIRE(mse(a, b) == Approx(3.416666666));
    }
}

TEST_CASE("F1 Score for Mulit class confusion Matrix", "[MC]") {
    using SEQL::Eval::f1_score;

    SECTION("No zero entry in the confusion matrix") {
        arma::umat cm(3, 3, arma::fill::ones);
        REQUIRE(f1_score(cm) == Approx(0.33333333333333));
    }

    SECTION("Confusion matrix ha one zero entry") {
        arma::umat cm(3, 3, arma::fill::ones);
        cm(0, 1) = 0;
        REQUIRE(f1_score(cm) == Approx(0.37777777777));
    }

    SECTION("Confusion matrix ha one zero entry") {
        arma::umat cm(3, 3, arma::fill::ones);
        cm(1, 1) = 0;
        REQUIRE(f1_score(cm) == Approx(0.222222222222));
    }

    SECTION("Confusion matrix ha one zero entry") {
        arma::umat cm(3, 3, arma::fill::zeros);
        cm(1, 0) = 9;
        cm(2, 0) = 1;
        REQUIRE(f1_score(cm) == 0);
    }
}

TEST_CASE("Weighted F1 Score for Mulit class confusion Matrix", "[MC]") {
    using SEQL::Eval::f1_score;

    SECTION("No zero entry in the confusion matrix") {
        arma::umat cm(3, 3, arma::fill::ones);
        REQUIRE(f1_score(cm, SEQL::Eval::weighted) == Approx(0.33333333333333));
    }

    SECTION("Confusion matrix ha one zero entry") {
        arma::umat cm(3, 3, arma::fill::ones);
        cm(0, 1) = 0;
        REQUIRE(f1_score(cm, SEQL::Eval::weighted) == Approx(0.375));
    }

    SECTION("Confusion matrix ha one zero entry") {
        arma::umat cm(3, 3, arma::fill::ones);
        cm(1, 1) = 0;
        REQUIRE(f1_score(cm, SEQL::Eval::weighted) == Approx(0.25));
    }

    SECTION("Confusion matrix ha one zero entry") {
        arma::umat cm(3, 3, arma::fill::zeros);
        cm(1, 0) = 9;
        cm(2, 0) = 1;
        REQUIRE(f1_score(cm, SEQL::Eval::weighted) == 0);
    }
}

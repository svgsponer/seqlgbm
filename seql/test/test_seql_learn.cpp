#include "../src/seql_learn.h"
#include <catch2/catch.hpp>

TEST_CASE("Loss function, squared error", "[loss][mse]") {

    std::vector<std::string> x = {"std"};
    std::vector<double> y = {0};
    LinearModel::LinearModel model;
    Seql_trainer learner{x, y,
                         3, // Squared Error Objective
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, model};
    SECTION("of two 0 vector is 0") {
        std::vector<double> a = {0, 0, 0};
        std::vector<double> b = {0, 0, 0};
        REQUIRE(learner.computeLoss(a, b) == 0);
    }
    SECTION("of equal vector is 0") {
        std::vector<double> a = {1, 1, 1};
        std::vector<double> b = {1, 1, 1};
        REQUIRE(learner.computeLoss(a, b) == 0);
    }
    SECTION("of (0, 1, 2) and (2, 3, 4) is 4") {
        std::vector<double> a = {0, 1, 2};
        std::vector<double> b = {2, 3, 4};
        REQUIRE(learner.computeLoss(a, b) == 12);
    }
    SECTION("works with doubles") {
        std::vector<double> a = {0, 1.5, 2.5};
        std::vector<double> b = {2, 3.5, 4};
        REQUIRE(learner.computeLoss(a, b) == Approx(10.25));
    }
    SECTION("intercept gradient calculation is correct") {
        std::vector<double> a = {0, 1, 2};
        std::vector<double> b = {2, 3, 4};
        REQUIRE(learner.calc_intercept_gradient(a, b) == Approx(-6));
    }
}

TEST_CASE("Loss function, logistic regression", "[loss][logreg]") {

    std::vector<std::string> x = {"std"};
    std::vector<double> y = {0};
    LinearModel::LinearModel model;
    Seql_trainer learner{x, y,
                         0, // Logistic Regreesion Objective
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, model};
    SECTION("of two 0 vectors") {
        std::vector<double> a = {0, 0, 0};
        std::vector<double> b = {0, 0, 0};
        REQUIRE(learner.computeLoss(a, b) == Approx(2.0794415));
    }
    SECTION("of equal vector is") {
        std::vector<double> a = {1, 1, 1};
        std::vector<double> b = {1, 1, 1};
        REQUIRE(learner.computeLoss(a, b) == Approx(0.93978506));
    }
    SECTION("of (0, 1, 2) and (2, 3, 4)") {
        std::vector<double> a = {0, 1, 2};
        std::vector<double> b = {2, 3, 4};
        REQUIRE(learner.computeLoss(a, b) == Approx(0.7420699));
    }
    SECTION("works with doubles") {
        std::vector<double> a = {0, 1.5, 2.5};
        std::vector<double> b = {2, 3.5, 4};
        REQUIRE(learner.computeLoss(a, b) == Approx(0.698426));
    }
}

TEST_CASE("Loss function, l2SVM", "[loss][l2SVM]") {

    std::vector<std::string> x = {"std"};
    std::vector<double> y = {0};
    LinearModel::LinearModel model;
    Seql_trainer learner{x, y,
                         2, // Logistic Regreesion Objective
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, model};
    SECTION("of two 0 vectors") {
        std::vector<double> a = {0, 0, 0};
        std::vector<double> b = {0, 0, 0};
        REQUIRE(learner.computeLoss(a, b) == 3);
    }
    SECTION("of equal vector is") {
        std::vector<double> a = {1, 1, 1};
        std::vector<double> b = {1, 1, 1};
        REQUIRE(learner.computeLoss(a, b) == 0);
    }
    SECTION("of (0, 1, 2) and (2, 3, 4)") {
        std::vector<double> a = {0, 1, 2};
        std::vector<double> b = {2, 3, 4};
        REQUIRE(learner.computeLoss(a, b) == 1);
    }
    SECTION("works with doubles") {
        std::vector<double> a = {0, 1.5, 2.5};
        std::vector<double> b = {2, 3.5, 4};
        REQUIRE(learner.computeLoss(a, b) == 1);
    }
};

TEST_CASE("Loss term, l2SVM", "[lossterm][l2SVM]") {

    std::vector<std::string> x = {"std"};
    std::vector<double> y = {0};
    LinearModel::LinearModel model;
    Seql_trainer learner{x, y,
                         2, // Squared Hinge Objective
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, model};
    SECTION("both values are 0") {
        double y = 0;
        double pred = 0;
        long double exp = 0;
        REQUIRE(learner.computeLossTerm(pred, y, exp) == 1);
    }
    SECTION("of correct prediction is 0") {
        double y = 1;
        double pred = 1;
        long double exp = 0;
        REQUIRE(learner.computeLossTerm(pred, y, exp) == 0);
    }
    SECTION("wrong prediction is") {
        double y = -1;
        double pred = 1;
        long double exp = 0;
        REQUIRE(learner.computeLossTerm(pred, y, exp) == 4);
    }
    SECTION("wrong prediction is") {
        double y = 1;
        double pred = -1;
        long double exp = 0;
        REQUIRE(learner.computeLossTerm(pred, y, exp) == 4);
    }
    SECTION("of prediction on the right sides") {
        double y = -1;
        double pred = 4;
        long double exp = 0;
        REQUIRE(learner.computeLossTerm(pred, y, exp) == 25);
    }
};

SCENARIO("loss will be updated correctly", "loss") {
    GIVEN(" A seql_learner class with squared error loss") {
        std::vector<std::string> x = {"std"};
        std::vector<double> y = {1, -1, 2};
        LinearModel::LinearModel model;
        Seql_trainer learner{x, y,
                             3, // Squared Hinge Objective
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, model};
        std::vector<double> pred = {-1, -1, 1};
        long double loss = learner.computeLoss(pred, y);
        REQUIRE(loss == 5);
        WHEN("one prediction changes") {
            std::vector<double> new_pred = {1, -1, 1};
            std::vector<unsigned int> loc = {0};
            learner.updateLoss(y, loss, new_pred, pred, loc);
            REQUIRE(loss == 1);
            REQUIRE(loss == learner.computeLoss(new_pred, y));
        }
        WHEN("two predictions change") {
            std::vector<double> new_pred = {1, -1, 2};
            std::vector<unsigned int> loc = {0, 2};
            learner.updateLoss(y, loss, new_pred, pred, loc);
            REQUIRE(loss == 0);
            REQUIRE(loss == learner.computeLoss(new_pred, y));
        }
        WHEN("one prediction changes with double values") {
            std::vector<double> new_pred = {1.5, -1, 1};
            std::vector<unsigned int> loc = {0};
            learner.updateLoss(y, loss, new_pred, pred, loc);
            REQUIRE(loss == Approx(1.25));
            REQUIRE(loss == learner.computeLoss(new_pred, y));
        }
    }
}

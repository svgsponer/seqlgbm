#include "../src/seql.h"
#include <catch2/catch.hpp>
#include <limits>

TEST_CASE("SEQL Loss function squared error", "[loss][mse]") {

    SEQL::Loss loss_funct;
    loss_funct.objective = SEQL::SqrdL;

    SECTION("single term loss is correct") {
        double a = 4;
        double b = 8;
        REQUIRE(loss_funct.computeLossTerm(a, b) == 16);
    }
    // SECTION("single term loss is correct with additional term") {
    //     double a = 4;
    //     double b = 8;
    //     long double c = 8;
    //     REQUIRE(loss_funct.computeLossTerm(a, b) == 16);
    // }
    SECTION("of two 0 vector is 0") {
        std::vector<double> a = {0, 0, 0};
        std::vector<double> b = {0, 0, 0};
        REQUIRE(loss_funct.computeLoss(a, b) == 0);
    }
    SECTION("of equal vector is 0") {
        std::vector<double> a = {1, 1, 1};
        std::vector<double> b = {1, 1, 1};
        REQUIRE(loss_funct.computeLoss(a, b) == 0);
    }
    SECTION("of (0, 1, 2) and (2, 3, 4) is 4") {
        std::vector<double> a = {0, 1, 2};
        std::vector<double> b = {2, 3, 4};
        REQUIRE(loss_funct.computeLoss(a, b) == 4);
    }
    SECTION("works with doubles") {
        std::vector<double> a = {0, 1.5, 2.5};
        std::vector<double> b = {2, 3.5, 4};
        REQUIRE(loss_funct.computeLoss(a, b) == Approx(10.25 / 3));
    }
    SECTION("intercept gradient calculation is correct") {
        std::vector<double> a = {0, 1, 2};
        std::vector<double> b = {2, 3, 4};
        REQUIRE(loss_funct.calc_intercept_gradient(a, b) == Approx(-6 / 3));
    }
}

TEST_CASE("SEQL Loss function Mean Absolute error", "[loss][mae]") {

    SEQL::Loss loss_funct;
    loss_funct.objective = SEQL::MAE;

    SECTION("single term loss is correct") {
        double a = 4;
        double b = 8;
        REQUIRE(loss_funct.computeLossTerm(a, b) == 4);
    }
    SECTION("single term loss is correct for negative numbers") {
        double a = -4;
        double b = 2;
        REQUIRE(loss_funct.computeLossTerm(a, b) == 6);
    }
    SECTION("of two 0 vector is 0") {
        std::vector<double> a = {0, 0, 0};
        std::vector<double> b = {0, 0, 0};
        REQUIRE(loss_funct.computeLoss(a, b) == 0);
    }
    SECTION("of equal vector is 0") {
        std::vector<double> a = {1, 1, 1};
        std::vector<double> b = {1, 1, 1};
        REQUIRE(loss_funct.computeLoss(a, b) == 0);
    }
    SECTION("of (0, 1, 2) and (2, 3, 4) is 4") {
        std::vector<double> a = {0, 1, 2};
        std::vector<double> b = {2, 3, 4};
        REQUIRE(loss_funct.computeLoss(a, b) == 2);
    }
    SECTION("works with doubles") {
        std::vector<double> a = {0, 1.5, 2.5};
        std::vector<double> b = {2, 3.5, 4};
        REQUIRE(loss_funct.computeLoss(a, b) == Approx(5.5 / 3));
    }
    SECTION("intercept gradient calculation is correct") {
        std::vector<double> a = {0, 1, 2};
        std::vector<double> b = {2, 3, 4};
        REQUIRE(loss_funct.calc_intercept_gradient(a, b) == Approx(-1));
    }
    SECTION("intercept gradient calculation is correct also with negative "
            "numbers") {
        std::vector<double> a = {0, 1, 2};
        std::vector<double> b = {-2, 3, 4};
        REQUIRE(loss_funct.calc_intercept_gradient(a, b) == Approx(-1 / 3.0));
    }
}

TEST_CASE("SEQL: Loss function logistic regression", "[loss][logreg]") {

    SEQL::Loss loss_funct;
    loss_funct.objective = SEQL::SLR;
    SECTION("of two 0 vectors") {
        std::vector<double> a = {0, 0, 0};
        std::vector<double> b = {0, 0, 0};
        REQUIRE(loss_funct.computeLoss(a, b) == Approx(2.0794415 / 3));
    }
    SECTION("of equal vector is") {
        std::vector<double> a = {1, 1, 1};
        std::vector<double> b = {1, 1, 1};
        REQUIRE(loss_funct.computeLoss(a, b) == Approx(0.93978506 / 3));
    }
    SECTION("of (0, 1, 2) and (2, 3, 4)") {
        std::vector<double> a = {0, 1, 2};
        std::vector<double> b = {2, 3, 4};
        REQUIRE(loss_funct.computeLoss(a, b) == Approx(0.7420699 / 3));
    }
    SECTION("works with doubles") {
        std::vector<double> a = {0, 1.5, 2.5};
        std::vector<double> b = {2, 3.5, 4};
        REQUIRE(loss_funct.computeLoss(a, b) == Approx(0.698426 / 3));
    }
    SECTION("gradient calculation is correct") {
        std::vector<double> a = {0, 1, 2, 0, 0};
        std::vector<double> b = {2, 3, 4, 1, -1};
        std::vector<double> grad;
        grad.reserve(5);
        loss_funct.calc_doc_gradients(a, b, grad);
        REQUIRE(grad[0] == Approx(1));
        REQUIRE(grad[1] == Approx(0.1422771695));
        REQUIRE(grad[2] == Approx(0.00134140052));
        REQUIRE(grad[3] == Approx(0.5));
        REQUIRE(grad[3] == Approx(0.5));
    }
    SECTION("intercept gradient calculation is correct") {
        std::vector<double> a = {0, 1, 2};
        std::vector<double> b = {2, 3, 4};
        REQUIRE(loss_funct.calc_intercept_gradient(a, b) ==
                Approx(-1.14361 / 3));
    }
}

TEST_CASE("SEQL: Loss function l2SVM", "[loss][l2SVM]") {

    SEQL::Loss loss_funct;
    loss_funct.objective = SEQL::l2SVM;
    SECTION("of two 0 vectors") {
        std::vector<double> a = {0, 0, 0};
        std::vector<double> b = {0, 0, 0};
        REQUIRE(loss_funct.computeLoss(a, b) == 3 / 3.0);
    }
    SECTION("of equal vector is") {
        std::vector<double> a = {1, 1, 1};
        std::vector<double> b = {1, 1, 1};
        REQUIRE(loss_funct.computeLoss(a, b) == 0 / 3.0);
    }
    SECTION("of (0, 1, 2) and (2, 3, 4)") {
        std::vector<double> a = {0, 1, 2};
        std::vector<double> b = {2, 3, 4};
        REQUIRE(loss_funct.computeLoss(a, b) == 1 / 3.0);
    }
    SECTION("works with doubles") {
        std::vector<double> a = {0, 1.5, 2.5};
        std::vector<double> b = {2, 3.5, 4};
        REQUIRE(loss_funct.computeLoss(a, b) == 1 / 3.0);
    }
    SECTION("gradient calculation is correct") {
        std::vector<double> a = {0, 1, 2};
        std::vector<double> b = {2, 3, 4};
        std::vector<double> grad;
        grad.reserve(3);
        loss_funct.calc_doc_gradients(a, b, grad);
        REQUIRE(grad[0] == Approx(4));
        REQUIRE(grad[1] == Approx(0));
        REQUIRE(grad[2] == Approx(0));
    }
    SECTION("intercept gradient calculation is correct") {
        std::vector<double> a = {0, 1, 2};
        std::vector<double> b = {2, 3, 4};
        REQUIRE(loss_funct.calc_intercept_gradient(a, b) == Approx(-4 / 3.0));
    }
}

TEST_CASE("SEQL: Loss term l2SVM", "[lossterm][l2SVM]") {

    SEQL::Loss loss_funct;
    loss_funct.objective = SEQL::l2SVM;
    SECTION("both values are 0") {
        double y    = 0;
        double pred = 0;
        REQUIRE(loss_funct.computeLossTerm(pred, y) == 1);
    }
    SECTION("of correct prediction is 0") {
        double y    = 1;
        double pred = 1;
        REQUIRE(loss_funct.computeLossTerm(pred, y) == 0);
    }
    SECTION("wrong prediction is") {
        double y    = -1;
        double pred = 1;
        REQUIRE(loss_funct.computeLossTerm(pred, y) == 4);
    }
    SECTION("wrong prediction is") {
        double y    = 1;
        double pred = -1;
        REQUIRE(loss_funct.computeLossTerm(pred, y));
    }
    SECTION("of prediction on the right sides") {
        double y    = -1;
        double pred = 4;
        REQUIRE(loss_funct.computeLossTerm(pred, y) == 25);
    }
}
/* UPDATE CURRENTLY DISABLED
SCENARIO("SEQL: loss will be updated correctly", "loss") {
    GIVEN(" A seql_learner class with squared error loss") {
        std::vector<double> y = {1, -1, 2};
        SEQL::Loss loss_funct;
        loss_funct.objective = SEQL::SqrdL;
        std::vector<double> pred = {-1, -1, 1};
        long double loss = loss_funct.computeLoss(pred, y);
        REQUIRE(loss == 5/3.0);
        WHEN("one prediction changes") {
            std::vector<double> new_pred = {1, -1, 1};
            std::vector<unsigned int> loc = {0};
            loss_funct.updateLoss(y, loss, new_pred, pred, loc);
            REQUIRE(loss == Approx(1/3.0));
            REQUIRE(loss == Approx(loss_funct.computeLoss(new_pred, y)));
        }
        // WHEN("two predictions change") {
        //     std::vector<double> new_pred = {1, -1, 2};
        //     std::vector<unsigned int> loc = {0, 2};
        //     loss_funct.updateLoss(y, loss, new_pred, pred, loc);
        //     REQUIRE(loss == Approx(0.0).margin(0.00001));
        //     REQUIRE(loss == Approx(loss_funct.computeLoss(new_pred, y)));
        // }
        WHEN("one prediction changes with double values") {
            std::vector<double> new_pred = {1.5, -1, 1};
            std::vector<unsigned int> loc = {0};
            loss_funct.updateLoss(y, loss, new_pred, pred, loc);
            REQUIRE(loss == Approx(1.25/3));
            REQUIRE(loss == Approx(loss_funct.computeLoss(new_pred, y)));
        }
    }
}
*/

SCENARIO("Tokenize a document into char tokens", "tokenize") {
    GIVEN("a document as a string") {
        std::string doc = "ABCBBCBDA";
        WHEN("using char token") {
            std::vector<std::string> gold = {
                "A", "B", "C", "B", "B", "C", "B", "D", "A"};
            auto ret = SEQL::tokenize(doc, 1);
            REQUIRE(ret == gold);
        }
    }
}

SCENARIO("Tokenize a document into word tokens", "tokenize") {
    GIVEN("a document as a string") {
        std::string doc = "AB C BBC BDA";
        WHEN("using word token") {
            std::vector<std::string> gold = {"AB", "C", "BBC", "BDA"};
            auto ret                      = SEQL::tokenize(doc, 0);
            REQUIRE(ret == gold);
        }
    }
}

TEST_CASE("SEQL: Regularization term computation", "[Regularization]") {
    SEQL::regularization_param rp{0.0, 0.0, 0, 1};
    SECTION("L1 Regularization works") {
        rp.C     = 1;
        rp.alpha = 1;
        REQUIRE(add_regularization(0.0, rp, 0, 1) == 1);
        REQUIRE(add_regularization(0.0, rp, 0, -1) == 1);
        REQUIRE(add_regularization(1.5, rp, 0, 1) == 2.5);
        REQUIRE(add_regularization(1.5, rp, 0, 2) == 3.5);
        REQUIRE(add_regularization(1.5, rp, 0, -2) == 3.5);
        REQUIRE(add_regularization(2.5, rp, 1.5, 2) == 3);
        REQUIRE(add_regularization(2.5, rp, -1.5, 2) == 3);
        REQUIRE(add_regularization(2.5, rp, -1.5, -2) == 3);
    }
    SECTION("L2 Regularization works") {
        rp.C     = 1;
        rp.alpha = 0;
        REQUIRE(add_regularization(0.0, rp, 0, 1) == 0.5);
        REQUIRE(add_regularization(0.0, rp, 0, -1) == 0.5);
        REQUIRE(add_regularization(1.5, rp, 0, 1) == 2);
        REQUIRE(add_regularization(1.5, rp, 0, 2) == 3.5);
        REQUIRE(add_regularization(1.5, rp, 0, -2) == 3.5);
        REQUIRE(add_regularization(2.5, rp, 1.5, 2) == 3.375);
        REQUIRE(add_regularization(2.5, rp, -1.5, 2) == 3.375);
        REQUIRE(add_regularization(2.5, rp, -1.5, -2) == 3.375);
        rp.sum_squared_betas = 25;
        REQUIRE(add_regularization(0.0, rp, 0, 1) == 13);
        REQUIRE(add_regularization(0.0, rp, 2, 3) == 15);
    }
}

TEST_CASE("SEQL: Gradient Calculation", "[SF]") {
    SEQL::Loss loss_funct;
    loss_funct.objective = SEQL::SLR;
    SECTION("Logistic regression gradient") {
        std::vector<double> pred             = {0.0, 0.0};
        std::vector<double> gs               = {1.0, -1.0};
        std::vector<std::vector<double>> sfs = {{1.0, -1.0}};
        auto grad = loss_funct.calc_sf_gradient(pred, gs, sfs);
        REQUIRE(Approx(-0.5) == grad[0]);
    }
    SECTION("Logistic regression gradient given prediction") {
        loss_funct.objective                 = SEQL::SLR;
        std::vector<double> pred             = {-2.0, 2.0};
        std::vector<double> y                = {1.0, -1.0};
        std::vector<std::vector<double>> sfs = {{1.0, -1.0}};
        std::vector<double> gradients(pred.size());
        loss_funct.calc_doc_gradients(pred, y, gradients);
        REQUIRE(Approx(0.88080) == gradients[0]);
        REQUIRE(Approx(-0.88080) == gradients[1]);
        auto grad = loss_funct.calc_sf_gradient(pred, y, sfs);
        REQUIRE(Approx(-0.88080) == grad[0]);
    }
    SECTION("Logistic regression gradient given prediction and sf") {
        loss_funct.objective                 = SEQL::SLR;
        std::vector<double> pred             = {-1.0, 1.5};
        std::vector<double> y                = {1.0, -1.0};
        std::vector<std::vector<double>> sfs = {{0.8, -2.1}};
        std::vector<double> gradients(pred.size());
        loss_funct.calc_doc_gradients(pred, y, gradients);
        REQUIRE(Approx(0.73106) == gradients[0]);
        REQUIRE(Approx(-0.81757) == gradients[1]);
        auto grad = loss_funct.calc_sf_gradient(pred, y, sfs);
        REQUIRE(Approx(-1.15088) == grad[0]);
    }
}

TEST_CASE("SEQL: parsing input", "[SF]") {
    SECTION("Logistic regression gradient") {
        std::stringstream input("#T 1 2 1\n"
                                "1 0.2 0.1 ABC\n"
                                "-1 -1.2 0 BCA\n");
        auto d = SEQL::read_input(input);
        REQUIRE(d.size() == 2);
        REQUIRE(d.x.size() == 2);
        REQUIRE(d.x_sf.size() == 3); // +1 due to intercept
        REQUIRE(d.x_sf[0].size() == 2);
        REQUIRE(d.x_sf[0][0] == 0.2);
        REQUIRE(d.x_sf[0][1] == -1.2);
        REQUIRE(d.x_sf[1][0] == 0.1);
        REQUIRE(d.x_sf[1][1] == 0.0);
        REQUIRE(d.x.size() == 2);
        REQUIRE(d.x[0] == std::vector<std::string>{"A", "B", "C"});
        REQUIRE(d.y.size() == 2);
        REQUIRE(d.y[0] == 1);
        REQUIRE(d.y[1] == -1);
    }
}

TEST_CASE("SEQL: regularization works", "[SF]") {
    SECTION("add_regularization works to 0 loss for L2") {
        SEQL::regularization_param rp{0, 0, 0, 1};
        auto reg_loss = SEQL::add_regularization(0, rp, 0, 1);
        REQUIRE(reg_loss == 0.5);
    }
    SECTION("add_regularization works to 0 loss for L1") {
        SEQL::regularization_param rp{0, 0, 1, 1};
        auto reg_loss = SEQL::add_regularization(0, rp, 0, 1);
        REQUIRE(reg_loss == 1);
    }
    SECTION("add_regularization works to 10 loss for L2") {
        SEQL::regularization_param rp{0, 0, 0, 1};
        auto reg_loss = SEQL::add_regularization(10, rp, 0, 1);
        REQUIRE(reg_loss == 10.5);
    }
    SECTION("add_regularization works to 10 loss for L1") {
        SEQL::regularization_param rp{0, 0, 1, 1};
        auto reg_loss = SEQL::add_regularization(10, rp, 0, 1);
        REQUIRE(reg_loss == 11);
    }
    SECTION("add_regularization works to 10 loss and previous betas for L2") {
        SEQL::regularization_param rp{10, 100, 0, 1};
        auto reg_loss = SEQL::add_regularization(10, rp, 10, 1);
        REQUIRE(reg_loss == 10.5);
    }
}

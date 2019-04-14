#include "../src/common.h"
#include "../src/preprocessing.h"
#include <catch2/catch.hpp>
#include <cmath>

TEST_CASE("SEQL: standardization", "[SF]") {
    SECTION("Standardization training works") {
        std::vector<double> a{1, 2, 0};
        SEQL::Preprocessing::StandardScaler stdscl;
        stdscl.fit_transform(a);
        REQUIRE(stdscl.mean == Approx(1));
        REQUIRE(stdscl.var == Approx(2 / 3.0));
        REQUIRE(stdscl.std == Approx(std::sqrt(2 / 3.0)));
        REQUIRE(a[0] == Approx(0));
        REQUIRE(a[1] == Approx(1.2247448));
        REQUIRE(a[2] == Approx(-1.2247448));
    }
    SECTION("Standardization applying works") {
        std::vector<double> a{1, 2, 0};
        SEQL::Preprocessing::StandardScaler stdscl;
        stdscl.mean = 1;
        stdscl.var  = 2 / 3.0;
        stdscl.std  = std::sqrt(2 / 3.0);
        stdscl.transform(a);
        // auto std_params = SEQL::standardize(a, params);
        REQUIRE(stdscl.mean == Approx(1));
        REQUIRE(stdscl.var == Approx(2 / 3.0));
        REQUIRE(stdscl.std == Approx(std::sqrt(2 / 3.0)));
        REQUIRE(a[0] == Approx(0));
        REQUIRE(a[1] == Approx(1.2247448));
        REQUIRE(a[2] == Approx(-1.2247448));
    }
    SECTION("Standardization fit transform over multiple dimensions works") {
        std::vector<std::vector<double>> a{{1, 2, 0}, {2, 3, 5}};
        auto ts = SEQL::Preprocessing::fit_apply_transformer<
            SEQL::Preprocessing::StandardScaler>(a.begin(), a.end());
        REQUIRE(ts[0].mean == Approx(1));
        REQUIRE(ts[0].var == Approx(2 / 3.0));
        REQUIRE(ts[0].std == Approx(std::sqrt(2 / 3.0)));
        REQUIRE(ts[1].mean == Approx(10 / 3.0));
        REQUIRE(ts[1].var == Approx(1.55555555));
        REQUIRE(ts[1].std == Approx(std::sqrt(1.55555555)));
        REQUIRE(a[0][0] == Approx(0));
        REQUIRE(a[0][1] == Approx(1.2247448));
        REQUIRE(a[0][2] == Approx(-1.2247448));
        REQUIRE(a[1][0] == Approx(-1.06904497));
        REQUIRE(a[1][1] == Approx(-.26726124));
        REQUIRE(a[1][2] == Approx(1.336306));
    }
}

TEST_CASE("SEQL: normalization", "[SF]") {
    SECTION("Norm2 calculation of basic vector is correct") {
        std::vector<double> a{1, 2, 3};
        double norm = SEQL::norm2(a);
        REQUIRE(norm == Approx(3.74165739));
    }
    SECTION("Normalization training works") {
        std::vector<double> a{1, 2, 3};
        SEQL::Preprocessing::NormScaler normscl;
        normscl.fit_transform(a);
        REQUIRE(normscl.norm == Approx(3.74165739));
        REQUIRE(a[0] == Approx(1 / 3.74165739));
        REQUIRE(a[1] == Approx(2 / 3.74165739));
        REQUIRE(a[2] == Approx(3 / 3.74165739));
    }
    SECTION("Normalization applying works") {
        std::vector<double> a{1, 2, 3};
        SEQL::Preprocessing::NormScaler normscl;
        normscl.fit_transform(a);
        REQUIRE(normscl.norm == Approx(std::sqrt(14)));
        REQUIRE(a[0] == Approx(1 / std::sqrt(14)));
        REQUIRE(a[1] == Approx(2 / std::sqrt(14)));
        REQUIRE(a[2] == Approx(3 / std::sqrt(14)));
    }
    SECTION("Normalization data works") {
        std::vector<std::vector<double>> x = {{1, 2, 3}, {1, 1, 1}, {4, 1, 1}};
        SEQL::Preprocessing::NormScaler normscl;
        auto ts = SEQL::Preprocessing::fit_apply_transformer<
            SEQL::Preprocessing::NormScaler>(x.begin(), x.end());
        REQUIRE(ts.size() == 3);
        REQUIRE(ts[0].norm == Approx(sqrt(14)));
        REQUIRE(ts[1].norm == Approx(sqrt(3)));
        REQUIRE(ts[2].norm == Approx(sqrt(18)));
        REQUIRE(x[0][0] == Approx(1 / sqrt(14)));
        REQUIRE(x[0][1] == Approx(2 / sqrt(14)));
        REQUIRE(x[0][2] == Approx(3 / sqrt(14)));
        REQUIRE(x[1][0] == Approx(1 / sqrt(3)));
        REQUIRE(x[1][1] == Approx(1 / sqrt(3)));
        REQUIRE(x[1][2] == Approx(1 / sqrt(3)));
        REQUIRE(x[2][0] == Approx(4 / sqrt(18)));
        REQUIRE(x[2][1] == Approx(1 / sqrt(18)));
        REQUIRE(x[2][2] == Approx(1 / sqrt(18)));
    }
}

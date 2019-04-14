#include "linear_model.h"
#include "seql.h"
#include <algorithm>
#include <armadillo>
#include <cmath>
#include <numeric>
#include <vector>

#ifndef EVALUATION_H
#define EVALUATION_H

namespace SEQL {
namespace Eval {

using stats_map = std::map<std::string, double>;

struct ConfusionMatrix {
    unsigned int TP = 0;
    unsigned int FP = 0;
    unsigned int FN = 0;
    unsigned int TN = 0;
    unsigned int P  = 0;
    unsigned int N  = 0;
    ConfusionMatrix(const std::vector<double> &true_labels,
                    const std::vector<double> &predictions);
};

arma::umat confusion_matrix(const std::vector<double> &true_labels,
                            const std::vector<double> &predictions,
                            const int num_class);

enum f1_weight_schema : int8_t { macro, weighted };

double accuracy(const arma::umat &cm);
double f1_score(const arma::umat &cm, const f1_weight_schema schema = macro);

// Classification
stats_map eval_classifier(const std::vector<double> &true_values,
                          const std::vector<double> &predictions);

// Regression

struct RegressionStats {
    unsigned int numberOfDataPoints{0};
    double sumY{0};
    double sumX{0};
    double sumSqrdE{0};
    double sumAbsE{0};
    double sumx2{0};
    double sumy2{0};
    double sumxy{0};
};

stats_map eval_regressor(const std::vector<double> &true_values,
                         const std::vector<double> &predictions);

double calcROC(const std::vector<double> &true_values,
               const std::vector<double> &predictions, unsigned int P,
               unsigned int N);

double calcROC(std::vector<double> y, std::vector<double> predictions);

std::vector<std::pair<double, double>>
calcROCpoints(const std::vector<std::pair<double, double>> &scores,
              unsigned int P, unsigned int N);

// Compute the area under the ROC50 curve.
// Fixes the number of negatives to 50.
// Stop computing curve after seeing 50 negatives.
double calcROC50(const std::vector<std::pair<double, double>> &scores);

// Compute r-absolute error
double calcRabs(const std::vector<double> &y,
                const std::vector<double> &predictions, const double meanY);

// Compute r-squared score
double calcR2(const std::vector<double> &y,
              const std::vector<double> &predictions, const double meanY);

// Metrics

double pearson_correlation(RegressionStats reg_stats);
double mse(std::vector<double> y, std::vector<double> pred);

} // namespace Eval
} // namespace SEQL
#endif

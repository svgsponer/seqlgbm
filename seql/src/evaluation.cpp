// Basic file for evaluation metrics

#include "evaluation.h"

// ClASSIFICATION

SEQL::Eval::ConfusionMatrix::ConfusionMatrix(
    const std::vector<double> &true_labels,
    const std::vector<double> &predictions) {
    const auto n = true_labels.size();
    for (auto i = 0u; i < n; ++i) {
        auto y          = true_labels[i];
        auto prediction = predictions[i];
        if (y > 0) {
            P++;
            if (prediction > 0)
                TP++;
            else
                FN++;
        } else {
            N++;
            if (prediction > 0)
                FP++;
            else
                TN++;
        }
    }
}

arma::umat SEQL::Eval::confusion_matrix(const std::vector<double> &true_labels,
                                        const std::vector<double> &predictions,
                                        const int num_class) {
    arma::umat c_matrix(num_class, num_class, arma::fill::zeros);
    for (decltype(true_labels.size()) i = 0, end = true_labels.size(); i < end;
         ++i) {
        ++c_matrix(true_labels[i], predictions[i]);
    }
    return c_matrix;
}
// Calculates the accuracy from a confusion matrix
double SEQL::Eval::accuracy(const arma::umat &cm) {
    auto dcm          = arma::conv_to<arma::dmat>::from(cm);
    auto sum_TP       = trace(dcm);
    auto total_points = accu(dcm);
    return sum_TP / total_points;
}

// calculates the f1 score from a confusion matrix
// currently only supports macro average
double SEQL::Eval::f1_score(const arma::umat &cm,
                            const f1_weight_schema schema) {
    arma::dvec const sum_pred =
        arma::conv_to<arma::dmat>::from(arma::sum(cm, 0).t());
    arma::dvec const sum_true =
        arma::conv_to<arma::dmat>::from(arma::sum(cm, 1));
    arma::dvec const true_pos =
        arma::conv_to<arma::dmat>::from(arma::diagvec(cm));

    arma::dvec precision = true_pos / sum_pred;
    arma::dvec recall    = true_pos / sum_true;
    if (precision.has_nan() || recall.has_nan()) {
        std::cerr << "F-score is ill-defined, division by zero." << '\n';
        precision.transform(
            [](double val) { return (std::isnan(val) ? double(0) : val); });
        recall.transform(
            [](double val) { return (std::isnan(val) ? double(0) : val); });
    }
    arma::dvec f1 = 2 * ((precision % recall) / (precision + recall));
    f1.elem(arma::find(true_pos == 0)).zeros();

    arma::dvec weights(arma::size(f1));
    switch (schema) {
    case macro:
        weights = weights.fill(1);
        break;
    case weighted:
        weights = sum_true;
        break;
    default:
        weights = weights.fill(1);
        break;
    };
    double const f1_avg = arma::sum(f1 % weights) / arma::sum(weights);
    return f1_avg;
}

SEQL::Eval::stats_map
SEQL::Eval::eval_classifier(const std::vector<double> &true_values,
                            const std::vector<double> &predictions) {

    if (true_values.size() != predictions.size()) {
        std::cerr << "\n"
                  << SEQL::Color::FG_RED
                  << "ERROR: Vectors of predictions and true values must have "
                     "same length:\n"
                  << SEQL::Color::FG_DEFAULT
                  << "Size predictions: " << predictions.size() << "\n"
                  << "Size true values: " << true_values.size() << std::endl;
        std::exit(-1);
    }
    // for (int i = 0; i < true_values.size(); ++i) {
    //     std::cout << true_values[i] << " " << predictions[i] << std::endl;
    // }

    SEQL::Eval::ConfusionMatrix cm(true_values, predictions);
    double prec = (cm.TP + cm.FP == 0) ? 0 : 1.0 * cm.TP / (cm.TP + cm.FP);
    double rec  = (cm.TP + cm.FN == 0) ? 0 : 1.0 * cm.TP / (cm.TP + cm.FN);
    double f1   = (prec + rec == 0) ? 0 : 2 * rec * prec / (prec + rec);
    double specificity =
        (cm.TN + cm.FP == 0) ? 0 : 1.0 * cm.TN / (cm.TN + cm.FP);
    // sensitivity = recall
    double sensitivity =
        (cm.TP + cm.FN == 0) ? 0 : 1.0 * cm.TP / (cm.TP + cm.FN);
    double fss =
        (specificity + sensitivity == 0)
            ? 0
            : 2 * specificity * sensitivity / (specificity + sensitivity);
    double MCC = ((cm.TP * cm.TN) - (cm.FP * cm.FN)) /
                 sqrt((1.0 * cm.TP + cm.FP) * (1.0 * cm.TP + cm.FN) *
                      (1.0 * cm.TN + cm.FP) * (1.0 * cm.TN + cm.FN));

    // double AUC = SEQLCm::calcROC(scores, cm.P, cm.N);
    // double AUC50 = SEQLCm::calcROC50(scores);
    double balanced_error = 0.5 * ((1.0 * cm.FN / (cm.TP + cm.FN)) +
                                   (1.0 * cm.FP / (cm.FP + cm.TN)));
    unsigned int correct  = cm.TP + cm.TN;
    unsigned int all      = cm.TP + cm.FP + cm.FN + cm.TN;

    std::map<std::string, double> stats_map{
        // {"Classif Threshold",	 -model->get_bias()},
        {"Accuracy", 100.0 * correct / all},
        {"Error", 100.0 - 100.0 * correct / all},
        {"Balanced Error", 100.0 * balanced_error},
        // {"AUC",			 AUC},
        {"Precision", 100.0 * prec},
        {"Recall", 100.0 * rec},
        {"F1", 100.0 * f1},
        {"Specificity", 100.0 * specificity},
        {"Sensitivity", 100.0 * sensitivity},
        {"FSS", 100.0 * fss},
        {"TruePositive", cm.TP},
        {"FalsePositive", cm.FP},
        {"TrueNegative", cm.TN},
        {"FalseNegative", cm.FN},
        {"MCC", 100 * MCC},
        // {"OOV docs",		 getOOVDocs()},
        // {"number of rules",  model->weights.size()}
    };
    return stats_map;
}

// void SEQLPredictor::evalFile_classifier(std::string filename, std::string
// outFileRequested){
//     std::vector<double> y;
//     std::vector<std::string> x;
//     std::tie(x,y, std::ignore) = SEQL::read_input(filename);
//     auto predictions = predict(x);
//     eval_classifier(y, predictions);
// }

// Compute the area under the ROC curve. The scores are expected to be sorted
std::vector<std::pair<double, double>>
SEQL::Eval::calcROCpoints(const std::vector<std::pair<double, double>> &scores,
                          unsigned int P, unsigned int N) {
    std::vector<std::pair<double, double>> ROC_points;
    double prevscore = -std::numeric_limits<double>::infinity();
    double FP = 0, TP = 0;
    for (const auto &pred : scores) {
        if (pred.first != prevscore) {
            ROC_points.emplace_back(std::make_pair(FP / N, TP / P));
            prevscore = pred.first;
        }
        if (pred.second > 0)
            TP++;
        else
            FP++;
    }
    return ROC_points;
}

// Compute the area under the ROC curve. The scores are expected to be sorted
double SEQL::Eval::calcROC(const std::vector<double> &true_values,
                           const std::vector<double> &predictions,
                           unsigned int P, unsigned int N) {
    double area = 0;
    double TP = 0, TPprev = 0;
    double FP = 0, FPprev = 0;
    double prevscore      = -std::numeric_limits<double>::infinity();
    auto ritr_predictions = predictions.crbegin();
    for (auto ritr = true_values.crbegin(); ritr != true_values.crend();
         std::advance(ritr, 1)) {
        double prediction = *ritr_predictions;
        int true_value    = *ritr;
        if (prediction != prevscore) {
            area += (TP - TPprev) * ((FP + FPprev) / 2.0);
            TPprev    = TP;
            FPprev    = FP;
            prevscore = prediction;
        }
        if (true_value > 0)
            FP++;
        else
            TP++;

        std::advance(ritr_predictions, 1);
    }
    area += (TP - TPprev) * ((FP + FPprev) / 2.0); // the last bin
    if (0 == FP || TP == 0)
        area = 0.0; // degenerate case
    else
        area = area / (P * N);
    return area;
}

double SEQL::Eval::calcROC(std::vector<double> y,
                           std::vector<double> predictions) {
    auto p      = SEQL::sort_permutation(predictions);
    predictions = SEQL::apply_permutation(predictions, p);
    y           = SEQL::apply_permutation(y, p);

    int P = std::count_if(
        std::begin(y), std::end(y), [](double ele) { return ele > 0; });
    int N = y.size() - P;

    return SEQL::Eval::calcROC(y, predictions, P, N);
}

double
SEQL::Eval::calcROC50(const std::vector<std::pair<double, double>> &scores) {
    double area50 = 0;
    double x = 0, xbreak = 0;
    double y = 0, ybreak = 0;
    double prevscore = -std::numeric_limits<double>::infinity();
    for (auto ritr = scores.rbegin(); ritr != scores.rend(); ritr++) {
        double score = ritr->first;
        int label    = ritr->second;

        if (score != prevscore && x < 50) {
            area50 += (x - xbreak) * (y + ybreak) / 2.0;
            xbreak    = x;
            ybreak    = y;
            prevscore = score;
        }
        if (label > 0)
            y++;
        else if (x < 50)
            x++;
    }
    area50 += (x - xbreak) * (y + ybreak) / 2.0; // the last bin
    if (0 == y || x == 0)
        area50 = 0.0; // degenerate case
    else
        area50 = 100.0 * area50 / (50 * y);
    return area50;
}

// REGRESSION
SEQL::Eval::stats_map
SEQL::Eval::eval_regressor(const std::vector<double> &true_values,
                           const std::vector<double> &predictions) {
    if (true_values.size() != predictions.size()) {
        std::cerr << "\n"
                  << SEQL::Color::FG_RED
                  << "ERROR: Vectors of predictions and true values must have "
                     "same length:\n"
                  << SEQL::Color::FG_DEFAULT
                  << "Size predictions: " << predictions.size() << "\n"
                  << "Size true values: " << true_values.size() << std::endl;
        std::exit(-1);
    }
    const auto n = true_values.size();

    SEQL::Eval::RegressionStats reg_stats;

    reg_stats.numberOfDataPoints = n;
    for (auto i = 0u; i < n; ++i) {
        auto y          = true_values[i];
        auto prediction = predictions[i];

        // REGRESSION STATS UPDATE
        reg_stats.sumSqrdE += pow(prediction - y, 2);
        reg_stats.sumAbsE += std::abs(prediction - y);
        reg_stats.sumx2 += pow(prediction, 2);
        reg_stats.sumy2 += pow(y, 2);
        reg_stats.sumxy += prediction * y;
        reg_stats.sumY += y;
        reg_stats.sumX += prediction;
    }

    auto meanY = reg_stats.sumY / reg_stats.numberOfDataPoints;
    std::map<std::string, double> stats_map{
        {"mean", meanY},
        {"mean-squared-error",
         reg_stats.sumSqrdE / reg_stats.numberOfDataPoints},
        {"mean-absolute-error",
         reg_stats.sumAbsE / reg_stats.numberOfDataPoints},
        {"R2", SEQL::Eval::calcR2(true_values, predictions, meanY)},
        {"RAbs", SEQL::Eval::calcRabs(true_values, predictions, meanY)},
        {"Pearson", pearson_correlation(reg_stats)}};

    return stats_map;
}

double SEQL::Eval::mse(std::vector<double> y, std::vector<double> pred) {
    std::transform(std::begin(y),
                   std::end(y),
                   std::begin(pred),
                   std::begin(y),
                   [](double y, double pred) { return pow(pred - y, 2); });
    double sum = std::accumulate(std::begin(y), std::end(y), 0.0);
    sum        = sum / y.size();
    return sum;
}

double SEQL::Eval::pearson_correlation(RegressionStats reg_stats) {
    return (reg_stats.sumxy - ((reg_stats.sumX * reg_stats.sumY) /
                               reg_stats.numberOfDataPoints)) /
           (sqrt(reg_stats.sumx2 -
                 pow(reg_stats.sumX, 2) / reg_stats.numberOfDataPoints) *
            sqrt(reg_stats.sumy2 -
                 pow(reg_stats.sumY, 2) / reg_stats.numberOfDataPoints));
}

double SEQL::Eval::calcR2(const std::vector<double> &y,
                          const std::vector<double> &predictions,
                          const double meanY) {
    double SSTotal = 0;
    double SSRes   = 0;
    auto itr_pred  = std::cbegin(predictions);
    for (auto itr_y = std::cbegin(y); itr_y != std::cend(y);
         std::advance(itr_y, 1)) {
        SSTotal += pow(*itr_y - meanY, 2);
        SSRes += pow(*itr_y - *itr_pred, 2);
        std::advance(itr_pred, 1);
    }
    double r2 = (SSRes / SSTotal);
    return r2;
}

// Compute r-absolute error
double SEQL::Eval::calcRabs(const std::vector<double> &y,
                            const std::vector<double> &predictions,
                            const double meanY) {
    double SSTotal = 0;
    double SSRes   = 0;
    auto itr_pred  = std::cbegin(predictions);
    for (auto itr_y = std::cbegin(y); itr_y != std::cend(y);
         std::advance(itr_y, 1)) {
        SSTotal += std::abs(*itr_y - meanY);
        SSRes += std::abs(*itr_y - *itr_pred);
        std::advance(itr_pred, 1);
    }
    double r = (SSRes / SSTotal);
    return r;
}

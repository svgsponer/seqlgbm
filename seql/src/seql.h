#ifndef SEQL_H
#define SEQL_H

#include "common.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <set>
#include <vector>

namespace SEQL {
enum Lossfunction {
    SLR   = 0,
    l1SVM = 1,
    l2SVM = 2,
    SqrdL = 3,
    MAE   = 4
    // EXP   = 5
};

using token_type = std::string;
struct Data {
    std::vector<std::vector<token_type>> x;
    // Save each static featrue as vector in this vector
    // so that we can access a single feature as vector later
    std::vector<std::vector<double>> x_sf;
    std::vector<double> y;

    void clear() {
        x.clear();
        x_sf.clear();
        y.clear();
    }

    size_t size() const { return std::size(y); }
    size_t num_sf() const { return std::size(x_sf); }
};

struct Configuration {
    unsigned int objective{0};
    // Word or character token type. By default char token.
    bool use_char_token{true};

    // Pattern properties
    unsigned int maxpat{0xffffffff};
    unsigned int minpat{1};
    // max iterations used SEQL
    unsigned int maxitr{5000};
    unsigned int minsup{1};

    // Max # of total wildcards allowed in a feature.
    unsigned int maxgap{0};
    // Max # of consec wildcards allowed in a feature.
    unsigned int maxcongap{0};

    // BFS vs DFS traversal. By default BFS.
    bool use_bfs{true};

    // The C regularizer parameter in regularized loss formulation. It
    // constraints the weights of features. C = 0 no constraints (standard SLR),
    // the larger the C, the more the weights are shrinked towards each other
    // (using L2) or towards 0 (using L1)
    double C{1};
    // The alpha parameter decides weight on L1 vs L2 regularizer: alpha * L1 +
    // (1 - alpha) * L2. By default we use an L2 regularizer.
    double alpha{0.2};

    double convergence_threshold{0.001};
    int verbosity{1};

    bool csv_log{true};

    double mean{0};
    // Max iterations used by GBM
    unsigned int max_itr_gbm{5};
    double shrinkage{1.0};

    std::string train_file;
    std::string test_file;
    std::string basename;
    std::string csv_file;
    std::string model_creation_file;
    std::string model_bin_file;
    std::string model_file;
    std::string prediction_file;
    std::string train_prediction_file;
    std::string stats_file;

    Configuration() = default;

    Configuration(std::string basename) :
        basename{basename}, csv_file{basename + ".csv"},
        model_creation_file{basename + ".modelCreation"},
        model_bin_file{basename + ".bin"}, model_file{basename + ".model"},
        prediction_file{basename + ".conc.pred"},
        train_prediction_file{basename + ".train.conc.pred"},
        stats_file{basename + ".stats.json"} {};
    void set_basename(const std::string &new_basename);
    friend std::ostream &operator<<(std::ostream &os,
                                    const Configuration &config);
};

// Needed to make gcc happy
std::ostream &operator<<(std::ostream &os, const Configuration &config);

void to_json(nlohmann::json &j, const SEQL::Configuration &c);
void from_json(const nlohmann::json &j, SEQL::Configuration &c);

struct regularization_param {
    double sum_abs_betas     = 0; //! Sum of current absolute weights
    double sum_squared_betas = 0; //! Sum of current squared weights
    double alpha = 0; //! Ratio bewteen L1 and L2 regularization. Elasticnet
    double C     = 0; //! Weight of regularization term

    double get_reg_term() const {
        return C * ((alpha * sum_abs_betas) +
                    ((1 - alpha) * 0.5 * sum_squared_betas));
    }

    friend std::ostream &operator<<(std::ostream &o,
                                    const regularization_param &a) {
        o << "C: " << a.C << ", alpha: " << a.alpha << " (weight of L1 vs L2)"
          << "\nL1 penalty: " << a.sum_abs_betas
          << "\nL2 penalty: " << 0.5 * a.sum_squared_betas
          << "\nRegularization term: " << a.get_reg_term();
        return o;
    }
};

/** \brief Calculates an updated regularized loss by substraction an old beta
 * value and adding a new one.
 *
 */
long double add_regularization(const double loss, const regularization_param rp,
                               const double old_bc, const double new_bc);
struct Loss {
    Lossfunction objective{};

    long double computeLossTerm(const double &y_pred,
                                const double &y_true) const;

    long double computeLossTerm(const double &y_pred, const double &y_true,
                                long double &exp_fraction) const;

    void updateLoss(const std::vector<double> &y_vec, long double &loss,
                    const std::vector<double> &y_pred_opt,
                    const std::vector<double> &y_pred,
                    const std::vector<unsigned int> loc) const;

    double computeLoss(const std::vector<double> &predictions,
                       const std::vector<double> &y_vec) const;

    void updateLoss(const std::vector<double> &y_vec, long double &loss,
                    const std::vector<double> &y_pred_opt,
                    const std::vector<double> &y_pred,
                    const std::vector<unsigned int> loc,
                    double &sum_abs_scalar_prod_diff,
                    double &sum_abs_scalar_prod,
                    std::vector<double long> &exp_fraction) const;

    // void update_exp_fraction(const std::vector<double>&  predictions,
    //                          const std::vector<double>&  y);

    std::vector<double>
    calc_sf_gradient(const std::vector<double> &y_pred,
                     const std::vector<double> &y_true,
                     const std::vector<std::vector<double>> &sfs) const;

    void calc_doc_gradients(const std::vector<double> &y_pred,
                            const std::vector<double> &y_true,
                            std::vector<double> &gradient_vec) const;

    double calc_intercept_gradient(const std::vector<double> &y_pred,
                                   const std::vector<double> &y_true) const;
    Loss() : objective{SEQL::SqrdL} {};
    Loss(unsigned int objective) :
        objective{static_cast<Lossfunction>(objective)} {};
};

double update_convthreshold(const std::vector<double> &y_pred,
                            const std::vector<double> &y_pred_opt);

bool hit_stoping_criterion(const double convergence_rate,
                           const double convergence_threshold);

bool hit_stoping_criterion(const std::vector<double> &y_pred,
                           const std::vector<double> &y_pred_old,
                           const double convergence_threshold);

SEQL::Data read_sax(std::istream &is, int limit = -1);
SEQL::Data read_sax(std::filesystem::path filename, int limit = -1);
SEQL::Data read_input(std::istream &is, int limit = -1);
SEQL::Data read_input(std::filesystem::path filename, int limit = -1);

struct SeqlFileHeader {
    int version{0};
    int num_sf{0};
    bool use_char_token{true};
};

SeqlFileHeader parse_seql_file_header(const char *hlc);

std::map<int, int> print_class_stats(SEQL::Data const &data);
std::tuple<double, double> print_reg_stats(SEQL::Data const &data);
std::vector<std::string> tokenize(std::string doc, bool use_char_token);

void decompose_ngram(std::string ngram, std::set<std::string> &set);

template <typename T>
std::ostream &print_vector(const T &t, std::ostream &of,
                           const char *sep = ", ") {
    std::copy(t.cbegin(),
              t.cend(),
              std::ostream_iterator<typename T::value_type>(of, sep));
    of << std::endl;
    return of;
}

template <typename T>
void print_vector(const T &t, const std::string filename,
                  const char *sep = ", ") {
    std::ofstream of(filename, std::ios::app);
    print_vector(t, of, sep);
}

template <typename M, typename K, typename V>
void insert_or_add(M &m, const K k, const V v) {
    m[k] += v;
}
namespace Color {
constexpr auto FG_RED     = "\033[31m";
constexpr auto FG_GREEN   = "\033[32m";
constexpr auto FG_BLUE    = "\033[34m";
constexpr auto FG_DEFAULT = "\033[39m";
constexpr auto BG_RED     = "\033[41m";
constexpr auto BG_GREEN   = "\033[42m";
constexpr auto BG_BLUE    = "\033[44m";
constexpr auto BG_DEFAULT = "\033[49m";
} // namespace Color

// Calculates the initial weight for different objective functions
// Mean squared error => mean(y)
// Logistic regression => 0.5 * ((1+mean(y))/(1-mean(y)))
std::vector<double> calculate_init_model(const std::vector<double> &y,
                                         SEQL::Lossfunction objective);
} // namespace SEQL
#endif

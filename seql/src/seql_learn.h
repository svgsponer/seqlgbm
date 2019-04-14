/*
 * (squared-hinge-loss) SVM for Classifying Sequences in the feature space of
 * all possible subsequences in the given training set. Elastic Net regularizer:
 * alpha * L1 + (1 - alpha) * L2, which combines L1 and L2 penalty effects. L1
 * influences the sparsity of the model, L2 corrects potentially high
 * coeficients resulting due to feature correlation (see Regularization Paths
 * for Generalized Linear Models via Coordinate Descent, by Friedman et al,
 * 2010).
 *
 * License:
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation.
 *
 */

#ifndef SEQL_LEARN_H
#define SEQL_LEARN_H

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>
// #include <ctime>
#include "csv_writer.h"
#include "linear_model.h"
#include "search_node.h"
#include "seql.h"
#include "sys/time.h"

using SeedType = std::map<string, SNode>;

class Seql_trainer {

  private:
    // Best ngram rule.
    struct rule_t {
        int f_idx = -1;
        double gradient{0};
        // Length of ngram.
        unsigned int size{0};
        // Ngram label.
        std::string ngram = "";
        // Ngram support, e.g. docids where it occurs in the collection.
        std::vector<unsigned int> loc;
        friend bool operator<(const rule_t &r1, const rule_t &r2) {
            return r1.ngram < r2.ngram;
        }
        rule_t() = default;
    };

    struct bound_t {
        double gradient;
        double upos;
        double uneg;
        unsigned int support;
    };

    /** Struct that repre
     *
     */
    struct line_search_point {
        std::vector<double> pred;
        long double loss{0};
        long double step_length{0};
        long double update_step{0};

        line_search_point(std::vector<double> pred, double loss,
                          double step_length, double update_step) :
            pred{pred},
            loss{loss}, step_length{step_length}, update_step{update_step} {};

        friend std::ostream &operator<<(std::ostream &o,
                                        const line_search_point &a) {
            o << "loss: " << a.loss << "\tstep_length: " << a.step_length
              << "\tupdate_step: " << a.update_step;
            return o;
        }
    };

    // Entire collection of documents, each doc represented as a string.
    // The collection is a vector of strings.
    // const std::vector<string> transaction;
    // True classes.
    // std::vector<double> y;

    std::vector<double> gradients;

    long double loss;        // loss without regulatization added
    long double regLoss;     // keep loss with regulatization added
    long double old_regLoss; // keep loss in previous iteration for checking
                             // convergence

    std::map<string, double> features_cache;
    std::map<string, double>::iterator features_it;

    // PARAMETERS
    // Objective function. For now choice between
    // - logistic regression,
    // - l2SVM (Squared Hinge Loss)
    // - squared error loss
    SEQL::Loss loss_funct;
    // Regularizer value.
    double C = 1;
    // Weight on L1 vs L2 regularizer.
    double alpha = 0.2;
    // Max length for an ngram.
    unsigned int maxpat = 1;
    // Min length for an ngram.
    unsigned int minpat = 0;
    // // Min suport for an ngram.
    // unsigned int minsup = 1;

    // The sum of squared values of all non-zero beta_j.
    double sum_squared_betas = 0;

    // The sum of abs values of all non-zero beta_j.
    double sum_abs_betas = 0;

    std::set<string> single_node_minsup_cache;

    // Current suboptimal gradient.
    double tau = 0;

    // Total number of times the pruning condition is checked
    unsigned int total;
    // Total number of times the pruning condition is satisfied.
    unsigned int pruned;
    // Total number of times the best rule is updated.
    unsigned int rewritten;

    // Convergence threshold on aggregated change in score predictions.
    // Used to automatically set the number of optimisation iterations.
    double convergence_threshold = 0.005;

    // Verbosity level: 0 - print no information,
    //                  1 - print profiling information,
    //                  2 - print statistics on model and obj-fct-value per
    //                  iteration > 2 - print details about search for best
    //                  n-gram and pruning process
    int verbosity = 1;

    // Traversal strategy: BFS or DFS.
    bool use_bfs;

    // Profiling variables.
    struct timeval t;
    struct timeval t_origin;
    struct timeval t_start_iter;

    // Bool if csvfile loggin is turned on or not
    bool csvLog;

    // Bool if warmstart should be used. Wamrstart reevaluates the top_nodes
    // found in previous iteration befor start the search at the unigrams.
    bool warmstart{false};

    // Maximum number of iterations
    unsigned int maxitr{5000};

    int current_itr{0};

    // calculates the bound of a node
    bound_t calculate_bound(const SEQL::Data &data, SNode *space);

    // Chechs if node with given gradient can be pruned
    bool can_prune(SNode *space, bound_t bound);

    // Udates the rule if gradient is bigger than tau
    void update_rule(rule_t &rule, SNode *space, unsigned int size,
                     bound_t bound);

    // Try to grow the ngram to next level, and prune the appropriate
    // extensions. The growth is done breadth-first, e.g. grow all unigrams
    // to bi-grams, than all bi-grams to tri-grams, etc.
    void span_bfs(const SEQL::Data &data, rule_t &rule, SNode *space,
                  std::vector<SNode *> &new_space, unsigned int size);

    // Try to grow the ngram to next level, and prune the appropriate
    // extensions. The growth is done deapth-first rather than
    // breadth-first, e.g. grow each candidate to its longest unpruned
    // sequence Probably a bug inthere hence not used anymore! void span_dfs
    // (rule_t& rule, SNode *space, unsigned int size);

    /** Line search method for ngram features
     *
     * Creates a vector of 0 and 1 depending if ngram is present in
     * document. Uses this mask as feature vector in the line search
     */
    // long double line_search(const rule_t &rule,
    //                         const std::vector<double> &y_true,
    //                         const std::vector<double> &y_pred,
    //                         const SEQL::regularization_param rp);

    long double line_search(const rule_t &rule,
                            const std::vector<double> &y_true,
                            const std::vector<double> &y_pred,
                            const std::vector<double> &f_vec,
                            const SEQL::regularization_param rp);
    long double interpolate_step_size(const std::vector<double> &y_true,
                                      line_search_point n0,
                                      line_search_point n1,
                                      line_search_point n2, const rule_t &rule,
                                      const SEQL::regularization_param rp);
    rule_t findBestFeature(const SEQL::Data &data, // rule_t &rule,
                           SeedType &seed, const std::vector<double> &y_pred);

    bool pass_subgradient_crit(const double grad, const std::string name) const;

    // Searches the space of all subsequences for the ngram with the ngram
    // with the maximal abolute gradient and saves it in rule
    rule_t findBestNgram(const SEQL::Data &data, rule_t &rule, SeedType &seed);

    // Function that calculates the the excat step size for given
    // coordinate. Only used for Squared error loss.
    double excact_step_length(const rule_t &rule,
                              const std::vector<double> &y_pred,
                              const std::vector<double> &y_true);

    void warm_start(const SEQL::Data &data, rule_t &rule);
    std::vector<SNode *> top_nodes{};
    void update_loss(const std::vector<double> &predictions,
                     const std::vector<double> &y);

  public:
    Seql_trainer(SEQL::Loss loss_funct, unsigned int maxpat,
                 unsigned int minpat, unsigned int minsup, unsigned int maxgap,
                 unsigned int maxcongap, bool token_type, bool use_bfs,
                 double convergence_threshold, double regularizer_value,
                 double l1vsl2_regularizer, int verbosity, bool csvLog,
                 bool warmstart = false, unsigned int maxitr = 5000) :
        loss_funct{loss_funct},
        C{regularizer_value}, alpha{l1vsl2_regularizer}, maxpat{maxpat},
        minpat{minpat},
        convergence_threshold{convergence_threshold}, verbosity{verbosity},
        use_bfs{use_bfs}, csvLog{csvLog}, warmstart{warmstart}, maxitr{maxitr} {
        SNode::setupWildcardConstraint(maxgap, maxcongap);
        SNode::use_char_token = token_type;
        SNode::minsup         = minsup;
    };

    Seql_trainer(SEQL::Configuration c) :
        loss_funct{c.objective}, C{c.C}, alpha{c.alpha}, maxpat{c.maxpat},
        minpat{c.minpat}, convergence_threshold{c.convergence_threshold},
        verbosity{c.verbosity}, use_bfs{c.use_bfs}, csvLog{c.csv_log},
        warmstart{false}, maxitr{c.maxitr} {
        SNode::setupWildcardConstraint(c.maxgap, c.maxcongap);
        SNode::use_char_token = c.use_char_token;
        SNode::minsup         = c.minsup;
    };

    template <typename T = std::unordered_map<std::string, double>,
              typename P = std::vector<double>>
    std::tuple<T, P>
    train(const SEQL::Data &data, const std::string csvFile,
          std::function<void(T, P, int)> itr_end_callback = nullptr);

    template <typename T = std::unordered_map<std::string, double>,
              typename P = std::vector<double>>
    std::tuple<T, P>
    train(const SEQL::Data &data, const std::string csvFile, SeedType &seed,
          std::function<void(T, P, int)> itr_end_callback = nullptr);

    template <typename T = std::unordered_map<std::string, double>,
              typename P = std::vector<double>>
    std::tuple<T, P>
    slim_train(const SEQL::Data &data, CSVwriter *logger, SeedType &seed,
               std::function<void(T, P, int)> itr_end_callback = nullptr);

    long double adjust_intercept(const std::vector<double> &y_pred,
                                 const std::vector<double> &y_true);

    long double adjust_intercept(const std::vector<double> &y_pred,
                                 const std::vector<double> &y_true,
                                 const double gradient);

    std::unique_ptr<CSVwriter> setup(string csvFile);
    SEQL::Lossfunction get_objective() { return loss_funct.objective; };
};

template <typename T, typename P>
std::tuple<T, P>
Seql_trainer::train(const SEQL::Data &data, std::string csvFile,
                    std::function<void(T, P, int)> itr_end_callback) {
    // A map from unigrams to search_space.
    SeedType seed = prepareInvertedIndex(data.x);
    return train(data, csvFile, seed, itr_end_callback);
}

template <typename T, typename P>
std::tuple<T, P>
Seql_trainer::train(const SEQL::Data &data, const std::string csvFile,
                    SeedType &seed,
                    std::function<void(T, P, int)> itr_end_callback) {
    if (verbosity >= 1) {
        cout << "\nParameters used: "
             << "obective fct: " << loss_funct.objective << " T: " << maxitr
             << " minpat: " << minpat << " maxpat: " << maxpat
             << " minsup: " << SNode::minsup
             << " maxgap: " << SNode::totalWildcardLimit
             << " maxcongap: " << SNode::consecWildcardLimit
             << " use_char_token: " << SNode::use_char_token
             << " use_bfs: " << use_bfs
             << " convergence_threshold: " << convergence_threshold
             << " C (regularizer value): " << C
             << " alpha (weight on l1_vs_l2_regularizer): " << alpha
             << " verbosity: " << verbosity << endl;
    }
    gettimeofday(&t_origin, NULL);
    deleteUndersupportedUnigrams(seed);

    gettimeofday(&t, NULL);
    if (SNode::verbosity >= 1) {
        std::cout << "\n# distinct unigrams: "
                  << SNode::single_node_minsup_cache.size();
        std::cout << " ( " << (t.tv_sec - t_origin.tv_sec) << " seconds; "
                  << (t.tv_sec - t_origin.tv_sec) / 60.0 << " minutes )";
        std::cout.flush();
    }

    auto logger = setup(csvFile);
    return slim_train(data, logger.get(), seed, itr_end_callback);
}

template <typename T, typename P>
std::tuple<T, P>
Seql_trainer::slim_train(const SEQL::Data &data, CSVwriter *logger,
                         SeedType &seed,
                         std::function<void(T, P, int)> itr_end_callback) {
    // std::srand(
    //     std::time(nullptr)); // use current time as seed for random generator

    T model;
    gradients = std::vector<double>(data.x.size(), 0.0);

    // Per document, prediction with current model
    std::vector<double> y_pred_opt(data.x.size(), 0.0);
    // The scalar product obtained with the optimal beta according to the
    // line search for best step size.
    std::vector<double> y_pred;
    y_pred.reserve(y_pred_opt.size());

    // The optimal step length.
    long double step_length_opt;
    // Set the convergence threshold as in paper by Madigan et al on BBR.
    long double convergence_rate{100};

    // Current rule.
    rule_t rule;

    // Init model
    auto ret   = SEQL::calculate_init_model(data.y, loss_funct.objective);
    y_pred     = ret;
    y_pred_opt = y_pred;
    SEQL::insert_or_add(
        model, "*" + std::to_string(data.x_sf.size() - 1) + "*", y_pred[0]);
    // Compute loss with start beta vector.
    loss    = loss_funct.computeLoss(y_pred_opt, data.y);
    regLoss = loss;

    auto initial_timestamp = std::chrono::steady_clock::now();

    if (verbosity >= 1) {
        cout << "\nstart loss: " << loss << '\n';
    }

    auto csvlog = [&](auto itr) {
        if (csvLog) {
            logger->DoLog(
                itr,
                features_cache.size(),
                rewritten,
                pruned,
                total,
                step_length_opt,
                rule.ngram,
                loss,
                regLoss,
                convergence_rate,
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - initial_timestamp)
                    .count());
        }
    };
    // Main loop
    for (unsigned int itr = 0; itr < maxitr; ++itr) {
        current_itr = itr;
        y_pred      = std::move(y_pred_opt);

        if (verbosity > 0) {
            cout << "\n\n_______\nModel itr: " << itr << "\n\n";
        }

        gettimeofday(&t_start_iter, NULL);

        // generate sub seed in cyclic fashin
        // auto sn = itr % seed_length;
        // std::map<std::string, SNode *> subseed;
        // // stdit, std::next(it, 1));
        // std::cout << "Selected " << it->first << '\n';
        rule = findBestFeature(data, seed, y_pred);
        if (verbosity >= 2) {
            cout << SEQL::Color::BG_BLUE << "Search finished\n"
                 << "Feature: \t" << rule.ngram << "\n"
                 << "Gradient: \t" << rule.gradient << "\n";
            if (verbosity > 3) {
                cout << "#rewrote: \t" << rewritten << "\n"
                     << "#prone: \t" << pruned << "\n"
                     << "#total: \t" << total << '\n';
            }
            cout << SEQL::Color::BG_DEFAULT << '\n';
        }

        if (std::abs(rule.gradient) < 1e-8) {
            cout << "\nBest ngram has a gradient of 0 => Stop search\n";
            y_pred_opt = std::move(y_pred); // Move as it has to be returned
            break;
        }

        // Find best step_size for static features/intercept
        SEQL::regularization_param rp = {
            sum_abs_betas, sum_squared_betas, alpha, C};

        const auto f_vec = [&rule, &data]() {
            if (rule.f_idx >= 0) { // Static featrue
                return data.x_sf[rule.f_idx];
            } else if (rule.f_idx == -1) { // Intercept
                return data.x_sf.back();
            } else { // Ngram feature (e.g. spare vector representation)
                std::vector<double> mask(data.y.size(), 0.0);
                for (const auto id : rule.loc) {
                    mask[id] = 1;
                }
                return mask;
            }
        }();

        step_length_opt = line_search(rule, data.y, y_pred, f_vec, rp);
        y_pred_opt.clear();
        SEQL::insert_or_add(model, rule.ngram, -step_length_opt);
        std::transform(std::begin(y_pred),
                       std::end(y_pred),
                       std::begin(f_vec),
                       std::back_insert_iterator(y_pred_opt),
                       [step_length_opt](double pred, double val) {
                           return pred - (step_length_opt * val);
                       });

        if (C != 0 && rule.f_idx != -1) {
            // Inserts if not there else return reference to featrue
            auto feature_insert = features_cache.insert({rule.ngram, 0.0});
            // Adjust coeficient and the sums of coeficients.
            if (!feature_insert.second) {
                sum_squared_betas -= pow(feature_insert.first->second, 2);
                sum_abs_betas -= abs(feature_insert.first->second);
            }
            feature_insert.first->second -= step_length_opt;
            sum_squared_betas += pow(feature_insert.first->second, 2);
            sum_abs_betas += abs(feature_insert.first->second);
        }
        update_loss(y_pred_opt, data.y);
        csvlog(itr);

        if (verbosity >= 2) {
            rp.sum_abs_betas     = sum_abs_betas;
            rp.sum_squared_betas = sum_squared_betas;
            cout << SEQL::Color::FG_GREEN << "Iteration summary " << itr << '\n'
                 << "Feature: " << rule.ngram << '\n'
                 << "Update step: " << step_length_opt << '\n'
                 << "Weight: " << model.at(rule.ngram) << '\n'
                 << "Loss: " << loss << '\n'
                 << "Regularized loss: " << regLoss << '\n'
                 << "Regularizer params: \n"
                 << rp << '\n';
        }
        // stop if loss doesn't improve; a failsafe check on top of
        // conv_rate (based on predicted score residuals) reaching
        // conv_threshold
        // if (std::abs(old_regLoss - regLoss) < 1e-8) {
        //     if (verbosity >= 1) {
        //         cout << "\n\nFinish iterations due to: no change in "
        //                 "loss "
        //                 "value!";
        //         cout << "\nloss + penalty term: " << regLoss;
        //         cout << "\n# iterations: " << itr + 1 << '\n';
        //     }
        //     break;
        // }

        // The optimal step length as obtained from the line search.
        // Stop the alg if weight of best grad feature is below a given
        // threshold. Inspired by paper of Liblinear people that use a
        // thereshold on the value of the gradient to stop close to
        // optimal solution.
        if (abs(step_length_opt) <= 1e-7) {
            if (verbosity >= 1) {
                cout << "\n\nFinish iterations due to: step_length_opt "
                        "<= "
                        "1e-8 "
                        "(due to numerical precision loss doesn't "
                        "improve "
                        "for "
                        "such small weights)!";
                cout << "\n# iterations: " << itr + 1;
                cout << "\nstep_length_opt: " << step_length_opt;
                cout << "\nngram: " << rule.ngram << '\n';
            }
            break;
        }

        // Set the convergence rate as in paper by Madigan et al on BBR.
        auto convergence_rate = SEQL::update_convthreshold(y_pred, y_pred_opt);
        if (SEQL::hit_stoping_criterion(convergence_rate,
                                        convergence_threshold)) {
            if (verbosity > 0) {
                std::cout << "\nconvergence rate: " << convergence_rate;
                std::cout << "\n\nFinish training due to: convergence test "
                             "(convergence_thereshold="
                          << convergence_threshold << ")!\n";
            }
            break;
        }

        if (itr_end_callback != nullptr)
            itr_end_callback(model, y_pred_opt, itr);
    } // end optimization iterations.

    y_pred = std::move(y_pred_opt);
    gettimeofday(&t, NULL);

    return std::make_tuple(model, y_pred);

} // end run().

#endif

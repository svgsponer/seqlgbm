/*
 * Authors:
 * Georgiana Ifrim (georgiana.ifrim@gmail.com),
 * Severin Gsponer (svgsponer+seql@gmail.com)
 *
 * SEQL: Sequence Learner This library trains
 * ElasticNet-regularized Logistic Regression and L2-loss (squared-hinge-loss)
 * SVM for Classifying Sequences in the feature space of all possible
 * subsequences in the given training set. Elastic Net regularizer: alpha * L1 +
 * (1 - alpha) * L2, which combines L1 and L2 penalty effects. L1 influences the
 * sparsity of the model, L2 corrects potentially high coeficients resulting due
 * to feature correlation (see Regularization Paths for Generalized Linear
 * Models via Coordinate Descent, by Friedman et al, 2010).
 *
 *
 * License:
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation.
 *
 */

#include "seql_learn.h"

using namespace std;

// For current ngram, compute the gradient value and bounds.
// There is a possible optimization by using the fact that if the child node
// has the same support as the parent node its gradient and bound are the same
// as the parent node.
// (To use this we need to save the gradient and bound of the parent somewhere)
Seql_trainer::bound_t Seql_trainer::calculate_bound(const SEQL::Data &data,
                                                    SNode *space) {
    ++total;

    // Upper bound for the positive class.
    double upos = 0;
    // Upper bound for the negative class.
    double uneg = 0;
    // Gradient value at current ngram.
    double gradient = 0;
    // Support of current ngram.
    unsigned int support = 0;
    // string reversed_ngram;
    const auto &loc = space->loc;
    // Compute the gradient and the upper bound on gradient of extensions.

    for (unsigned int i = 0; i < loc.size(); ++i) {
        if (loc[i] >= 0)
            continue;
        ++support;
        const unsigned int j = (unsigned int)(-loc[i]) - 1;

        switch (loss_funct.objective) {

        case SEQL::SLR:
            // From differentiation we get a - in front of the sum_i_to_N
            // gradient -= y[j] * exp_fraction[j];
            gradient -= gradients[j];
            if (data.y[j] > 0) {
                upos -= gradients[j];
            } else {
                uneg -= gradients[j];
            }
            break;
        case SEQL::l1SVM:
            std::cerr << "No l1SVM gradient bound implemented. Use l1SVM only "
                         "for GBM.\n";
            std::exit(-1);

        case SEQL::l2SVM:
            gradient -= gradients[j];
            if (data.y[j] > 0) {
                upos -= gradients[j];
            } else {
                uneg -= gradients[j];
            }
            break;

        case SEQL::SqrdL:
            gradient -= gradients[j];
            if (gradients[j] > 0) {
                upos -= gradients[j];
            } else {
                uneg -= gradients[j];
            }
            break;

        case SEQL::MAE:
            gradient -= gradients[j];
            if (gradients[j] > 0) {
                upos += 1;
            } else {
                uneg -= 1;
            }
            break;
        }
    }
    gradient /= data.y.size();
    upos /= data.y.size();
    uneg /= data.y.size();
    // Correct for already selected features
    if (C != 0) {

        std::string ngram = space->getNgram();

        if (verbosity > 3) {
            cout << "\n\ncurrent ngram rule: " << ngram;
            cout << "\nlocation size: " << space->loc.size();
            cout << "\ngradient (before regularizer): " << gradient;
            cout << "\nupos (before regularizer): " << upos;
            cout << "\nuneg (before regularizer): " << uneg;
            cout << "\ntau: " << tau << '\n';
        }

        double current_upos = 0;
        double current_uneg = 0;

        // Retrieve the beta_ij coeficient of this feature. If beta_ij is
        // non-zero, update the gradient: gradient += C * [alpha*sign(beta_j) +
        // (1-alpha)*beta_j]; Fct lower_bound return an iterator to the key >=
        // given key.
        features_it = features_cache.lower_bound(ngram);

        // Check if it is a zero feature is so use shrinkage operator
        if (features_it == features_cache.end() ||
            features_it->first.compare(ngram) > 0) {
            // cout << "Zero feature found" << std::endl;
            // SUBGRADIENT CASE HERE:
            // beta_ij is zero
            auto st = C * alpha;
            upos    = upos + st;
            uneg    = uneg - st;
            // Shrinkage operator
            if (gradient > st) {
                // cout << "Case: grad > C*alpha" << std::endl;
                gradient -= st;
            } else if (gradient < -st) {
                // cout << "Case: grad < C*alpha" << std::endl;
                gradient += st;
            } else {
                // cout << "Case: -C*alpha < grad < C*alpha" << std::endl;
                gradient = 0;
            }
        }

        // If there are keys starting with this prefix (this ngram starts at
        // pos 0 in existing feature).
        if (features_it != features_cache.end() &&
            features_it->first.find(ngram) == 0) {
            // If found an exact match for the key.
            // add regularizer to gradient.
            if (features_it->first.compare(ngram) == 0) {
                int sign = abs(features_it->second) / features_it->second;
                gradient +=
                    C * (alpha * sign + (1 - alpha) * features_it->second);

                if (verbosity > 3) {
                    cout << "\ngradient after regularizer: " << gradient;
                }
            }
            // Check if current feature s_j is a prefix of any non-zero
            // features s_j'. Check exact bound for every such non-zero
            // feature.
            while (features_it != features_cache.end() &&
                   features_it->first.find(ngram) == 0) {
                int sign     = abs(features_it->second) / features_it->second;
                current_upos = upos + C * (alpha * sign +
                                           (1 - alpha) * features_it->second);
                current_uneg = uneg + C * (alpha * sign +
                                           (1 - alpha) * features_it->second);

                if (verbosity > 3) {
                    cout << "\nexisting feature starting with current ngram "
                            "rule prefix: "
                         << features_it->first << ", " << features_it->second
                         << ",  sign: " << sign;

                    cout << "\ncurrent_upos: " << current_upos;
                    cout << "\ncurrent_uneg: " << current_uneg;
                    cout << "\ntau: " << tau;
                }
                // Check bound. If any non-zero feature starting with
                // current ngram as a prefix can still qualify for selection
                // in the model, we cannot prune the search space.
                if (std::max(abs(current_upos), abs(current_uneg)) > tau) {
                    upos = current_upos;
                    uneg = current_uneg;
                    break;
                }
                ++features_it;
            }
        }
    }
    bound_t bound = {gradient, upos, uneg, support};
    return bound;
}

bool Seql_trainer::can_prune(SNode *space, bound_t bound) {
    if (std::max(std::abs(bound.upos), std::abs(bound.uneg)) <= tau) {
        ++pruned;
        if (verbosity > 3) {
            cout << "\n"
                 << space->ngram << ": Pruned due to bound!"
                 << "\n\tgradient: " << bound.gradient << "\n\ttau: " << tau
                 << "\n\tupos: " << bound.upos << "\n\tuneg: " << bound.uneg
                 << "\n";
        }
        return true;
    }
    // Check if support of ngram is below minsupport
    if (bound.support < SNode::minsup) {
        ++pruned;
        if (verbosity > 3) {
            cout << "\n"
                 << space->ngram << ": Pruned since support < minsup!\n";
        }
        return true;
    }
    return false;
}

void Seql_trainer::update_rule(rule_t &rule, SNode *node, unsigned int size,
                               bound_t bound) {
    double g = std::abs(bound.gradient);
    // If current ngram better than previous best ngram, update optimal
    // ngram. Check min length requirement.
    if (g > tau && size >= minpat) {
        // if (pass_subgradient_crit(g, node->getNgram())) {
        ++rewritten;
        top_nodes.push_back(node);
        tau           = g;
        rule.gradient = bound.gradient;
        rule.size     = size;
        rule.ngram    = node->getNgram();
        rule.f_idx    = -2;

        if (verbosity >= 3) {
            cout << "\n\nNew current best ngram rule: " << node->getNgram();
            cout << "\ngradient: " << bound.gradient << "\n";
        }

        rule.loc.clear();
        for (unsigned int i = 0; i < node->loc.size(); ++i) {
            // Keep the doc ids where the best ngram appears.
            if (node->loc[i] < 0)
                rule.loc.push_back((unsigned int)(-node->loc[i]) - 1);
        }
        // } else {
        //     cout << "Declined due to subgradient criterion\n";
        // }
    }
}

// Try to grow the ngram to next level, and prune the appropriate
// extensions. The growth is done breadth-first, e.g. grow all unigrams to
// bi-grams, than all bi-grams to tri-grams, etc.
void Seql_trainer::span_bfs(const SEQL::Data &data, rule_t &rule, SNode *space,
                            std::vector<SNode *> &new_space,
                            unsigned int size) {

    // If working with gaps.
    // Check if number of consecutive gaps exceeds the max allowed.
    if (SNode::hasWildcardConstraints) {
        if (space->violateWildcardConstraint())
            return;
    }
    if (space->has_no_childs) {
        return;
    }

    // Expand node if not done already
    if (space->next.empty()) {
        space->expand_node(data.x);
    }

    for (auto const &feature : space->next) {
        // If the last token is a gap, skip checking gradient and pruning
        // bound, since this is the same as for the prev ngram without the
        // gap token. E.g., if we checked the gradient and bounds for "a"
        // and didnt prune it, then the gradient and bounds for "a*" will be
        // the same, so we can safely skip recomputing the gradient and
        // bounds.
        auto featptr = feature.get();
        if (feature->ne.compare("*") == 0) {
            if (verbosity > 4)
                cout << "\nFeature ends in *, skipping gradient and bound "
                        "computation.";
            new_space.push_back(featptr);
        } else {
            bound_t bound = calculate_bound(data, featptr);
            if (!can_prune(featptr, bound)) {
                update_rule(rule, featptr, size, bound);
                new_space.push_back(featptr);
            }
        }
    }
}

// Line search method. Binary search for optimal step size. Calls
// find_best_range(...). y_pred keeps track of the scalar product
// beta_best^t*xi for each doc xi. Instead of working with the new weight
// vector beta_n+1 obtained as beta_n - epsilon * gradient(beta_n) we work
// directly with the scalar product. We output the y_pred_opt which contains
// the scalar poduct of the optimal beta found, by searching for the optimal
// epsilon, e.g. beta_n+1 = beta_n - epsilon_opt * gradient(beta_n)
// epsilon is the starting value
// rule contains info about the gradient at the current iteration
long double Seql_trainer::line_search(const rule_t &rule,
                                      const std::vector<double> &y_true,
                                      const std::vector<double> &y_pred,
                                      const std::vector<double> &f_vec,
                                      const SEQL::regularization_param rp) {
    // Starting value for parameter in step size search.
    // Set the initial epsilon value small enough to guaranteee
    // log-like increases in the first steps.
    double exponent = floor(log10(abs(rule.gradient)));
    double epsilon  = min(1e-5, pow(10, -exponent));

    if (verbosity > 3) {
        cout << "\nrule.ngram: " << rule.ngram;
        cout << "\nrule.gradient: " << rule.gradient;
        cout << "\nexponent of epsilon: " << -exponent;
        cout << "\nepsilon: " << epsilon << '\n';
    }

    auto start_loss   = loss_funct.computeLoss(y_pred, y_true);
    auto pre_reg_loss = start_loss;
    if (rp.C != 0 && rule.f_idx != -1) {
        start_loss = add_regularization(start_loss, rp, 0, 0);
    }
    line_search_point n0(y_pred, start_loss, 0, 0);
    line_search_point n1(y_pred, start_loss, 0, 0);
    line_search_point n2(y_pred, start_loss, 0, 0);

    double current_weight = 0.0;
    if (features_it = features_cache.find(rule.ngram);
        features_it != features_cache.end()) {
        current_weight = features_it->second;
    }

    double new_weight = 0;
    // **************************************************************************/
    // As long as the loss decreases, double the epsilon.
    // Keep track of the last three values of beta, or correspondingly
    // the last 3 values for the scalar product of beta and xi.

    int n = 0;
    do {

        n0 = std::move(n1);
        n1 = std::move(n2);

        n2.step_length = pow(2, n * 1.0) * epsilon;
        n2.update_step = n2.step_length * rule.gradient;
        new_weight     = current_weight + n2.update_step;
        n2.pred.clear();

        std::transform(std::begin(y_pred),
                       std::end(y_pred),
                       std::begin(f_vec),
                       std::back_insert_iterator(n2.pred),
                       [n2](double pred, double val) {
                           return pred - (n2.update_step * val);
                       });

        n2.loss = loss_funct.computeLoss(n2.pred, y_true);

        // Add Regularizer
        pre_reg_loss = n2.loss;
        if (rp.C != 0 && rule.f_idx != -1) {
            n2.loss =
                add_regularization(n2.loss, rp, current_weight, new_weight);
        }

        if (verbosity > 4) {
            cout << SEQL::Color::FG_RED << "Line search step: " << n << '\n'
                 << "Current weight of feature: " << current_weight << '\n'
                 << "Step size: " << n2.step_length << '\n'
                 << "Update step: " << n2.update_step << '\n'
                 << "Updated weight: " << new_weight << '\n'
                 << "loss: " << pre_reg_loss << '\n'
                 << "regloss: " << n2.loss << '\n'
                 << "regterm: " << n2.loss - pre_reg_loss << '\n'
                 << "rp: " << rp << SEQL::Color::FG_DEFAULT << "\n\n"
                 << "Current prediction for first point:\n"
                 << "y_pred_n0[0]: " << n0.pred[0] << ' ' << n0 << '\n'
                 << "y_pred_n1[0]: " << n1.pred[0] << ' ' << n1 << '\n'
                 << "y_pred_n2[0]: " << n2.pred[0] << ' ' << n2 << '\n';
        }
        ++n;
        // } while (n2.loss < n1.loss || n0.loss == n1.loss);
    } while (n2.loss < n1.loss);

    if (verbosity > 4)
        cout << "\nFinished doubling epsilon!"
                "The monotonicity loss_n+1 < loss_n is broken!";

    return interpolate_step_size(y_true, n0, n1, n2, rule, rp);
}

// Line search method. Search for step size that minimizes loss.
// Compute loss in middle point of range, beta_n1, and
// for mid of both ranges beta_n0, beta_n1 and bet_n1, beta_n2
// Compare the loss for the 3 points, and choose range of 3 points
// which contains the minimum. Repeat until the range spanned by the 3
// points is small enough, e.g. the range approximates well the vector where
// the loss function is minimized. Return the middle point of the best
// range.
long double
Seql_trainer::interpolate_step_size(const std::vector<double> &y_true,
                                    line_search_point n0, line_search_point n1,
                                    line_search_point n2, const rule_t &rule,
                                    const SEQL::regularization_param rp) {
    line_search_point mid_n0_n1(n0);
    line_search_point mid_n1_n2(n0);

    // cerr << "\nn0: " << n0;
    // cerr << "\nmid_n0_n1: " << mid_n0_n1;
    // cerr << "\nn1: " << n1;
    // cerr << "\nmid_n1_n2: " << mid_n1_n2;
    // cerr << "\nn2: " << n2 << "\n";

    double min_range_size          = 1e-3;
    double current_range_size      = 0;
    int current_interpolation_iter = 0;

    auto current_weight = 0;
    if (rp.C != 0 && rule.f_idx != -1) {
        if (features_it = features_cache.find(rule.ngram);
            features_it != features_cache.end()) {
            current_weight = features_it->second;
        }
    }

    for (auto docId = 0u; docId < y_true.size(); docId++) {
        current_range_size += abs(n2.pred[docId] - n0.pred[docId]);
    }
    if (verbosity > 4) {
        cout << "\nn0.pred[0]: " << n0.pred[0];
        cout << "\nn1.pred[0]: " << n1.pred[0];
        cout << "\nn2.pred[0]: " << n2.pred[0];
        cout << "\ncurrent range size: " << current_range_size << '\n';
    }
    auto mean = [](double a, double b) { return (a + b) / 2; };
    // Start interpolation loop.
    while (current_range_size > min_range_size) {
        if (verbosity > 4) {
            cout << "current interpolation iteration: "
                 << current_interpolation_iter << '\n';

            std::transform(std::begin(n0.pred),
                           std::end(n0.pred),
                           std::begin(n1.pred),
                           std::begin(mid_n0_n1.pred),
                           mean);
            std::transform(std::begin(n1.pred),
                           std::end(n1.pred),
                           std::begin(n2.pred),
                           std::begin(mid_n1_n2.pred),
                           mean);
        }

        n1.loss        = loss_funct.computeLoss(n1.pred, y_true);
        mid_n0_n1.loss = loss_funct.computeLoss(mid_n0_n1.pred, y_true);
        mid_n1_n2.loss = loss_funct.computeLoss(mid_n1_n2.pred, y_true);

        mid_n0_n1.update_step = mean(n0.update_step, n1.update_step);
        mid_n0_n1.step_length = mean(n0.step_length, n1.step_length);
        mid_n1_n2.update_step = mean(n1.update_step, n2.update_step);
        mid_n1_n2.step_length = mean(n1.step_length, n2.step_length);

        if (C != 0 && rule.f_idx != -1) {
            auto update_regloss = [current_weight, &rp](line_search_point &p) {
                auto new_weight = current_weight + p.update_step;
                p.loss          = SEQL::add_regularization(
                    p.loss, rp, current_weight, new_weight);
            };
            update_regloss(n1);
            update_regloss(mid_n0_n1);
            update_regloss(mid_n1_n2);
        }

        // cerr << "\nn0: " << n0;
        // cerr << "\nmid_n0_n1: " << mid_n0_n1;
        // cerr << "\nn1: " << n1;
        // cerr << "\nmid_n1_n2: " << mid_n1_n2;
        // cerr << "\nn2: " << n2 << "\n";
        // Focus on the range that contains the minimum of the loss
        // function. Compare the 3 points beta_n, and mid_beta_n-1_n and
        // mid_beta_n_n+1.
        if (n1.loss <= mid_n0_n1.loss && n1.loss <= mid_n1_n2.loss) {
            // n1 is min
            n0 = mid_n0_n1;
            n2 = mid_n1_n2;
        } else if (mid_n0_n1.loss <= n1.loss &&
                   mid_n0_n1.loss <= mid_n1_n2.loss) {
            // mid_n0_n1 is min
            n2 = n1;
            n1 = mid_n0_n1;
        } else {
            // mid_n1_n2 is min
            n0 = n1;
            n1 = mid_n1_n2;
        }

        ++current_interpolation_iter;
        current_range_size = 0;

        for (auto docId = 0u; docId < y_true.size(); docId++) {
            current_range_size += abs(n2.pred[docId] - n0.pred[docId]);
        }
        if (verbosity > 4) {
            cout << "current range size: " << current_range_size << '\n';
        }
    };
    // cerr << n0.loss << "\n";
    // cerr << n1.loss << "\n";

    // cerr << "\nn0: " << n0;
    // cerr << "\nmid_n0_n1: " << mid_n0_n1;
    // cerr << "\nn1: " << n1;
    // cerr << "\nmid_n1_n2: " << mid_n1_n2;
    // cerr << "\nn2: " << n2 << "\n";
    if (n1.loss > n0.loss) {
        // cerr << "HERE retrun " << n0.update_step << '\n';
        return n0.update_step;
    } else {
        return n1.update_step;
    }
    cerr << "\n\n";
}

Seql_trainer::rule_t
Seql_trainer::findBestFeature(const SEQL::Data &data,
                              SeedType &seed,
                              const std::vector<double> &y_pred) {
    rule_t rule;
    const std::vector<double> &y_true = data.y;
    tau                               = 0;

    loss_funct.calc_doc_gradients(y_pred, y_true, gradients);

    if (warmstart) {
        warm_start(data, rule);
    } else {
        top_nodes.clear();
    }

    // Calculate static featrue gradients
    auto gradient_sfs = loss_funct.calc_sf_gradient(y_pred, y_true, data.x_sf);
    // SEQL::print_vector(gradient_sfs, "gradient_vec");

    // Add regularization gradient
    if (C != 0) {
        // -1 so Intercept will not be regularized
        for (std::size_t id = 0; id < gradient_sfs.size() - 1; ++id) {
            std::string name = "*" + std::to_string(id) + "*";
            auto &gradient   = gradient_sfs[id];
            // cout << id << " Gradient without reg: " << gradient <<
            // std::endl;
            if (features_it = features_cache.find(name);
                features_it != features_cache.end()) {
                int sign = abs(features_it->second) / features_it->second;
                auto reg_term =
                    C * ((alpha * sign) + ((1 - alpha) * features_it->second));
                gradient += reg_term;
            } else {
                // cout << "Zero feature found" << std::endl;
                // SUBGRADIENT CASE HERE:
                // beta_ij is zero
                auto st = C * alpha;
                // Shrinkage operator
                if (gradient > st) {
                    // cout << "Case: grad > C*alpha" << std::endl;
                    gradient -= st;
                } else if (gradient < -st) {
                    // cout << "Case: grad < C*alpha" << std::endl;
                    gradient += st;
                } else {
                    // cout << "Case: -C*alpha < grad < C*alpha" <<
                    // std::endl;
                    gradient = 0;
                }
            }
            // cout << id << " Greadient with reg:" << gradient <<
            // std::endl;
        }
    }

    auto best_sf_grad = max_element(
        gradient_sfs.begin(), gradient_sfs.end(), [](double a, double b) {
            return std::abs(a) < std::abs(b);
        });
    if (std::abs(*best_sf_grad) > tau) {
        auto pos  = std::distance(gradient_sfs.begin(), best_sf_grad);
        auto name = "*" + std::to_string(pos) + "*";
        // Found new best gradient in sf
        if (verbosity > 2) {
            cout << "Current best SF:\n"
                 << "SF NR: " << pos << '\n'
                 << "Gradient: " << *best_sf_grad << '\n';
        }
        rule.gradient = *best_sf_grad;
        rule.ngram    = name;
        rule.f_idx    = pos != (gradient_sfs.size() - 1) ? pos : -1;
        tau           = std::abs(rule.gradient);
    }

    findBestNgram(data, rule, seed);
    // if (pass_subgradient_crit(rule.gradient, rule.ngram)) {
    return rule;
    // } else {
    //     cout << "No rule satifies subgradient criterion\n";
    // }
    // return rule_t();
}

// bool Seql_trainer::pass_subgradient_crit(const double grad,
//                                          const std::string name) const {
//     cout << "HERE\n";
//     if (C != 0 && alpha != 0) {
//         auto curr_weight = features_cache.find(name);
//         if (curr_weight == features_cache.end() || curr_weight->second ==
//         0.0) {
//             if (std::abs(grad) <= C) {
//                 return false;
//             }
//         }
//     }
//     return true;
// };

// Searches the space of all subsequences for the ngram with the ngram with
// the maximal abolute gradient and saves it in rule
Seql_trainer::rule_t
Seql_trainer::findBestNgram(const SEQL::Data &data, rule_t &rule,
                            SeedType &seed) {
    std::vector<SNode *> old_space;
    std::vector<SNode *> new_space;
    old_space.clear();
    new_space.clear();
    pruned = total = rewritten = 0;

    // Code for random or cyclic seed selection
    // bool random_selection {true};
    // int sn {0};
    // if(random_selection){
    //     sn = std::rand()/((RAND_MAX + 1u)/std::size(seed)); // BIASED RANDOM
    // }else{
    //     sn = current_itr % std::size(seed);
    // }

    // auto unigram = std::begin(seed);
    // std::advance(unigram, sn);
    // cout << "Selected seed: "<< unigram->first <<std::endl;

    // Iterate through unigrams.
    for (auto &unigram : seed) {
        bound_t bound = calculate_bound(data, &unigram.second);
        if (!can_prune(&unigram.second, bound)) {
            update_rule(rule, &unigram.second, 1, bound);
            // Check BFS vs DFS traversal.
            if (use_bfs) {
                new_space.push_back(&unigram.second);
            } else {
                // Traversal is DFS.
                std::cerr << "DFS not implemented!";
                std::exit(-1);
                // span_dfs (rule, &unigram.second, 2);
            }
        }
    }

    if (use_bfs) {
        unsigned int size = 2;
        while (size <= maxpat && !new_space.empty()) {
            old_space = new_space;
            new_space.clear();
            for (auto curspace : old_space) {
                span_bfs(data, rule, curspace, new_space, size);
            }
            ++size;
        }
    }
    return rule;
}

void Seql_trainer::warm_start(const SEQL::Data &data, rule_t &rule) {
    // Calculate gradient and bound of nodes that have rewritten in previous
    // iteration.
    auto top_nodes_old(top_nodes);
    top_nodes.clear();
    for (auto orule : top_nodes_old) {
        bound_t bound = calculate_bound(data, orule);
        if (can_prune(orule, bound)) {
            update_rule(rule, orule, orule->ngram.size(), bound);
        }
        // std::cout << "Revaluate Node that rewrote last time: " <<
        // orule->getNgram() << std::endl;
    }
}

// Function that calculates the the excat step size for given coordinate.
// Only used for Squared error loss.
double Seql_trainer::excact_step_length(const rule_t &rule,
                                        const std::vector<double> &y_pred,
                                        const std::vector<double> &y_true) {
    double y_btx = 0;
    for (auto docId : rule.loc) {
        y_btx += -1 * y_true[docId] + y_pred[docId];
    }
    auto stepsize = y_btx / (rule.loc.size() * rule.gradient);
    auto step     = stepsize * rule.gradient;

    return step;
}

std::unique_ptr<CSVwriter> Seql_trainer::setup(const string csvFile) {
    // if (csvLog) {
    auto logger = std::make_unique<CSVwriter>(csvFile);
    logger->DoLog(
        "Iteration,#Features,#rewritten,#pruned,total,optimalStepLength,"
        "symbol,loss,regLoss,convRate,timestamp");
    // }
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(8);
    return logger;
}

void Seql_trainer::update_loss(const std::vector<double> &predictions,
                               const std::vector<double> &y) {
    // Remember the loss from prev iteration.
    old_regLoss   = regLoss;
    auto old_loss = loss;
    loss          = loss_funct.computeLoss(predictions, y);

    // Update regLoss
    if (C != 0) {
        SEQL::regularization_param rp = {
            sum_abs_betas, sum_squared_betas, alpha, C};
        regLoss = SEQL::add_regularization(loss, rp, 0, 0);

        if (verbosity >= 2) {
            cout << "Penalty term: \t"
                 // << C * (alpha * sum_abs_betas +
                 //         (1 - alpha) * 0.5 * sum_squared_betas)
                 // << "\ndc: "
                 << rp.get_reg_term() << "\n";
        }
    } else {
        regLoss = loss;
    }

    // Report loss
    if (verbosity >= 2) {
        cout << "Old_loss: \t" << old_loss << "\n"
             << "New_loss: \t" << loss << "\n"
             << "Old_regloss: \t" << old_regLoss << "\n"
             << "New_regloss: \t" << regLoss << "\n";
    }
}

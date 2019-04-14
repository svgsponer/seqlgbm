
long double add_regularization(const double loss,
                               const regurlatization_param rp,
                               const double old_bc, const double new_bc) {
    return loss +
           rp.C * (rp.alpha * (rp.sum_abs_betas - abs(old_bc) + abs(new_bc)) +
                (1 - rp.alpha) * 0.5 *
                    (rp.sum_squared_betas - pow(old_bc, 2) + pow(new_bc, 2)));
}
// Line search method. Binary search for optimal step size. Calls
// find_best_range(...). y_pred keeps track of the scalar product
// beta_best^t*xi for each doc xi. Instead of working with the new weight vector
// beta_n+1 obtained as beta_n - epsilon * gradient(beta_n) we work directly
// with the scalar product. We output the y_pred_opt which contains the
// scalar poduct of the optimal beta found, by searching for the optimal
// epsilon, e.g. beta_n+1 = beta_n - epsilon_opt * gradient(beta_n)
// epsilon is the starting value
// rule contains info about the gradient at the current iteration
long double SEQL::optimizer::line_search(const rule_t &rule,
                                         const std::vector<double> &y_true,
                                         const std::vector<double> &y_pred,
                                         const std::vector<double>& f_vec,
                                         const regularization_param rp) {

    // Starting value for parameter in step size search.
    // Set the initial epsilon value small enough to guaranteee
    // log-like increases in the first steps.
    double exponent = ceil(log10(abs(rule.gradient)));
    double epsilon = min(1e-3, pow(10, -exponent));

    if (verbosity > 3) {
        cout << "\nrule.ngram: " << rule.ngram;
        cout << "\nrule.gradient: " << rule.gradient;
        cout << "\nexponent of epsilon: " << -exponent;
        cout << "\nepsilon: " << epsilon;
    }

    // Keep track of scalar product at points beta_n-1, beta_n and beta_n+1.
    // They are denoted with beta_n0, beta_n1, beta_n2.
    vector<double> y_pred_n0(y_pred);
    vector<double> y_pred_n1(y_pred);
    vector<double> y_pred_n2(y_pred);

    // Keep track of loss at the three points, n0, n1, n2.
    long double loss_n0 = 0;
    long double loss_n1 = 0;
    long double loss_n2 = loss_funct.computeLoss(y_pred_n2, y_true);
    // added regularization term to loss
    long double regLoss_n2 = loss_n2;
    // Binary search for epsilon. Similar to bracketing phase in which
    // we search for some range with promising epsilon.
    // The second stage finds the epsilon or corresponding weight vector with
    // smallest l2-loss value.

    // Update regLoss
    if (rp.C != 0) {
      regLoss_n2 = add_regularization(loss_n2, rp, 0, 0);
    }

    // **************************************************************************/
    // As long as the loss decreases, double the epsilon.
    // Keep track of the last three values of beta, or correspondingly
    // the last 3 values for the scalar product of beta and xi.
    int n = 0;

    if (rp.C != 0 && rp.sum_squared_betas != 0) {
        features_it = features_cache.find(rule.ngram);
    }

    double beta_coeficient_update = 0;
    double update_step = 0;
    do {
        if (verbosity > 3)
            cout << "\nn: " << n;

        update_step = pow(2, n * 1.0) * epsilon * rule.gradient;
        beta_coeficient_update -= update_step;

        std::copy(std::begin(y_pred_n1), std::end(y_pred_n1),
                  std::begin(y_pred_n0));
        std::copy(std::begin(y_pred_n2), std::end(y_pred_n2),
                  std::begin(y_pred_n1));
        std::transform(std::begin(y_pred_n1), std::end(y_pred_n1),
                       std::begin(f_vec), std::begin(y_pred_n2),
                       [update_step](double pred, double val) {
                           return pred - (update_step * val);
                       });

        if (verbosity > 4) {
            cout << "\ny_pred_n0[0]: " << y_pred_n0[0];
            cout << "\ny_pred_n1[0]: " << y_pred_n1[0];
            cout << "\ny_pred_n2[0]: " << y_pred_n2[0];
        }

        // Compute loss for all 3 values: n-1, n, n+1
        // In the first iteration compute necessary loss.
        if (n == 0) {
            loss_n0 = regLoss;
            loss_n1 = regLoss;
        } else {
            // Update just loss_n2.
            // The loss_n0 and loss_n1 are already computed.
            loss_n0 = loss_n1;
            loss_n1 = regLoss_n2;
        }
        loss_n2 = loss_funct.computeLoss(y_pred_n2, y_true);

        if (verbosity > 4) {
            cout << "\nloss_n2 before adding regularizer: " << loss_n2;
        }

        // Add Regularizer
        // If this is the first ngram selected.
        if (rp.C != 0) {
          double new_weight = 0
            if (features_it != features_cache.end())
              {
               new_weight = features_it->second + weight_update;
              }
          regLoss_n2 = add_regularization(loss_n2, rp, features_it->second, new_weight);
        } else {
            regLoss_n2 = loss_n2;
        } // end C != 0.

        if (verbosity > 4) {
            cout << "\nloss_n0: " << loss_n0;
            cout << "\nloss_n1: " << loss_n1;
            cout << "\nloss_n2: " << regLoss_n2;
        }
        ++n;
    } while (regLoss_n2 < loss_n1);
    // **************************************************************************/

    if (verbosity > 3)
        cout << "\nFinished doubling epsilon! The monotonicity loss_n+1 < "
                "loss_n is broken!";

    vector<double> y_pred_mid_n0_n1(y_pred.size());
    vector<double> y_pred_mid_n1_n2(y_pred.size());

    return find_best_range(y_true, y_pred_n0, y_pred_n1, y_pred_n2, rule,
                           y_pred, f_vec);
}

// Line search method. Search for step size that minimizes loss.
// Compute loss in middle point of range, beta_n1, and
// for mid of both ranges beta_n0, beta_n1 and bet_n1, beta_n2
// Compare the loss for the 3 points, and choose range of 3 points
// which contains the minimum. Repeat until the range spanned by the 3 points is
// small enough, e.g. the range approximates well the vector where the loss
// function is minimized. Return the middle point of the best range.
long double SEQL::optimzer::find_best_range(
    const std::vector<double> &y_true, vector<double> &y_pred_n0,
    vector<double> &y_pred_n1, vector<double> &y_pred_n2, const rule_t &rule,
    const vector<double> &y_pred, const vector<double> &f_vec) {

    vector<double> y_pred_mid_n0_n1(y_pred.size());
    vector<double> y_pred_mid_n1_n2(y_pred.size());

    double min_range_size = 1e-5;
    double current_range_size = 0;
    int current_interpolation_iter = 0;

    long double loss_mid_n0_n1 = 0;
    long double loss_mid_n1_n2 = 0;
    long double loss_n1 = 0;
    for (auto docId = 0u; docId < y_true.size(); docId++) {
      if (verbosity > 4) {
        cout << "\ny_pred_n0[docId]: " << y_pred_n0[docId];
        cout << "\ny_pred_n1[docId]: " << y_pred_n1[docId];
        cout << "\ny_pred_n2[docId]: " << y_pred_n2[docId];
      }
      current_range_size += abs(y_pred_n2[docId] - y_pred_n0[docId]);
    }
    if (verbosity > 3)
        cout << "\ncurrent range size: " << current_range_size;

    double beta_coef_n1 = 0;
    double beta_coef_mid_n0_n1 = 0;
    double beta_coef_mid_n1_n2 = 0;

    if (C != 0) {
        features_it = features_cache.find(rule.ngram);
    }
    // Start interpolation loop.
    while (current_range_size > min_range_size) {
        if (verbosity > 3)
            cout << "\ncurrent interpolation iteration: "
                 << current_interpolation_iter;

        for (unsigned int i = 0; i < y_true.size(); ++i) {
            y_pred_mid_n0_n1[i] = (y_pred_n0[i] + y_pred_n1[i]) / 2;
            y_pred_mid_n1_n2[i] = (y_pred_n1[i] + y_pred_n2[i]) / 2;

            // } else if (C != 0) {
            //     beta_coef_n1 =
            //         y_pred_n1[rule.loc[0]] - y_pred[rule.loc[0]];
            //     beta_coef_mid_n0_n1 = y_pred_mid_n0_n1[rule.loc[0]] -
            //                           y_pred[rule.loc[0]];
            //     beta_coef_mid_n1_n2 = y_pred_mid_n1_n2[rule.loc[0]] -
            //                           y_pred[rule.loc[0]];
            // }

            if (verbosity > 4) {
                cout << "\ny_pred_mid_n0_n1[i]: " << y_pred_mid_n0_n1[i];
                cout << "\ny_pred_mid_n1_n2[i]: " << y_pred_mid_n1_n2[i];
            }
            loss_n1 += loss_funct.computeLossTerm(y_pred_n1[i], y_true[i]);
            loss_mid_n0_n1 +=
                loss_funct.computeLossTerm(y_pred_mid_n0_n1[i], y_true[i]);
            loss_mid_n1_n2 +=
                loss_funct.computeLossTerm(y_pred_mid_n1_n2[i], y_true[i]);
        } // end loop through training samples.

        if (C != 0) {
            beta_coef_n1 = (y_pred_n1[0] - y_pred[0]) / f_vec[0];
            beta_coef_mid_n0_n1 = (y_pred_mid_n0_n1[0] - y_pred[0]) / f_vec[0];
            beta_coef_mid_n1_n2 = (y_pred_mid_n1_n2[0] - y_pred[0]) / f_vec[0];

            // Lambda to add regularization term 

            double new_beta_coef_n1 = 0;
            double new_beta_coef_mid_n0_n1 = 0;
            double new_beta_coef_mid_n1_n2 = 0;
            if (features_it != features_cache.end()) {
              new_beta_coef_n1 = features_it->second + beta_coef_n1;
              new_beta_coef_mid_n0_n1 =
                features_it->second + beta_coef_mid_n0_n1;
              new_beta_coef_mid_n1_n2 =
                features_it->second + beta_coef_mid_n1_n2;
            }
            loss_n1 = add_regulaization(loss_n1, 0, beta_coef_n1);
            loss_mid_n0_n1 = add_regulaization(loss_mid_n0_n1, 0, beta_coef_mid_n0_n1);
            loss_mid_n1_n2 = add_regulaization(loss_mid_n1_n2, 0, beta_coef_mid_n1_n2);
        } // end check C != 0.

        // Focus on the range that contains the minimum of the loss function.
        // Compare the 3 points beta_n, and mid_beta_n-1_n and mid_beta_n_n+1.
        if (loss_n1 <= loss_mid_n0_n1 && loss_n1 <= loss_mid_n1_n2) {
            // Min is in beta_n1.
            if (verbosity > 4) {
                cout << "\nmin is y_pred_n1";
                cout << "\nloss_mid_n0_n1: " << loss_mid_n0_n1;
                cout << "\nloss_n1: " << loss_n1;
                cout << "\nloss_mid_n1_n2: " << loss_mid_n1_n2;
            }
            // Make the beta_n0 be the beta_mid_n0_n1.
            y_pred_n0.assign(y_pred_mid_n0_n1.begin(), y_pred_mid_n0_n1.end());
            // Make the beta_n2 be the beta_mid_n1_n2.
            y_pred_n2.assign(y_pred_mid_n1_n2.begin(), y_pred_mid_n1_n2.end());
        } else {
            if (loss_mid_n0_n1 <= loss_n1 && loss_mid_n0_n1 <= loss_mid_n1_n2) {
                // Min is beta_mid_n0_n1.
                if (verbosity > 4) {
                    cout << "\nmin is y_pred_mid_n0_n1";
                    cout << "\nloss_mid_n0_n1: " << loss_mid_n0_n1;
                    cout << "\nloss_n1: " << loss_n1;
                    cout << "\nloss_mid_n1_n2: " << loss_mid_n1_n2;
                }
                // Make the beta_n2 be the beta_n1.
                y_pred_n2.assign(y_pred_n1.begin(), y_pred_n1.end());
                // Make the beta_n1 be the beta_mid_n0_n1.
                y_pred_n1.assign(y_pred_mid_n0_n1.begin(),
                                 y_pred_mid_n0_n1.end());
            } else {
                // Min is beta_mid_n1_n2.
                if (verbosity > 4) {
                    cout << "\nmin is y_pred_mid_n1_n2";
                    cout << "\nloss_mid_n0_n1: " << loss_mid_n0_n1;
                    cout << "\nloss_n1: " << loss_n1;
                    cout << "\nloss_mid_n1_n2: " << loss_mid_n1_n2;
                }
                // Make the beta_n0 be the beta_n1.
                y_pred_n0.assign(y_pred_n1.begin(), y_pred_n1.end());
                // Make the beta_n1 be the beta_mid_n1_n2
                y_pred_n1.assign(y_pred_mid_n1_n2.begin(),
                                 y_pred_mid_n1_n2.end());
            }
        }

        ++current_interpolation_iter;
        loss_mid_n0_n1 = 0;
        loss_mid_n1_n2 = 0;
        loss_n1 = 0;
        current_range_size = 0;

        for (auto docId = 0u; docId < y_true.size(); docId++) {
          if (verbosity > 4) {
            cout << "\ny_pred_n0[[docId]: " << y_pred_n0[docId];
            cout << "\ny_pred_n1[[docId]: " << y_pred_n1[docId];
            cout << "\ny_pred_n2[[docId]: " << y_pred_n2[docId];
          }
          current_range_size += abs(y_pred_n2[docId] - y_pred_n0[docId]);
        }
        if (verbosity > 4) {
            cout << "\ncurrent range size: " << current_range_size;
        }
    } // end while loop.

    // Get update step
    long double update_step;
    update_step = (y_pred[0] - y_pred_n1[0])/f_vec[0];
    return update_step;
} // end find_best_range().

/*
 * Author: Severin Gsponer (severin.gsponer@insight-centre.org)
 *
 * Gradient boosting algorithm that uses SEQL models as weak learner.
 *
 * License:
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation.
 *
 */

#include "seql_boost.h"

using namespace std;

long double Gradient_boost_trainer::line_search(
    const int N, const std::vector<double> &weak_model,
    const std::vector<double> &y_true, const std::vector<double> &y_pred) {

    // Starting value for parameter in step size search.
    // Set the initial epsilon value small enough to guaranteee
    // log-like increases in the first steps.
    double epsilon = 1e-3;

    // Keep track of scalar product at points beta_n-1, beta_n and beta_n+1.
    // They are denoted with beta_n0, beta_n1, beta_n2.
    vector<double> y_pred_n0(y_pred);
    vector<double> y_pred_n1(y_pred);
    vector<double> y_pred_n2(y_pred);

    // Keep track of loss at the three points, n0, n1, n2.
    long double loss_n0 = 0;
    long double loss_n1 = 0;
    long double loss_n2 = loss_funct.computeLoss(y_pred_n2, y_true);
    // Binary search for epsilon. Similar to bracketing phase in which
    // we search for some range with promising epsilon.
    // The second stage finds the epsilon or corresponding weight vector with
    // smallest l2-loss value.

    // **************************************************************************/
    // As long as the l2-loss decreases, double the epsilon.
    // Keep track of the last three values of beta, or correspondingly
    // the last 3 values for the scalar product of beta and xi.
    int n = 0;

    long double beta_coeficient_update = 0;
    do {
        if (verbosity > 3)
            cout << "\nn: " << n;

        // For each location (e.g. docid), update the score of the documents
        // containing best rule. E.g. update beta^t * xi.
        beta_coeficient_update -= pow(2, n * 1.0) * epsilon;
        bool print = true;
        for (auto docid = 0; docid < N; docid++) {
            y_pred_n0[docid] = y_pred_n1[docid];
            y_pred_n1[docid] = y_pred_n2[docid];
            y_pred_n2[docid] = y_pred_n1[docid] +
                               pow(2, n * 1.0) * epsilon * weak_model[docid];

            if (verbosity > 3 && print) {
                cout << "\ny_pred_n0[docid]: " << y_pred_n0[docid];
                cout << "\ny_pred_n1[docid]: " << y_pred_n1[docid];
                cout << "\ny_pred_n2[docid]: " << y_pred_n2[docid];
                print = false;
            }
        }

        // Compute loss for all 3 values: n-1, n, n+1
        // In the first iteration compute necessary loss.
        if (n == 0) {
            loss_n0 = loss_n2;
            loss_n1 = loss_n2;
        } else {
            // Update just loss_n2.
            // The loss_n0 and loss_n1 are already computed.
            loss_n0 = loss_n1;
            loss_n1 = loss_n2;
        }
        loss_n2 = loss_funct.computeLoss(y_pred_n2, y_true);

        if (verbosity > 4) {
            cout << "\nloss_n2 before adding regularizer: " << loss_n2;
        }

        if (verbosity > 3) {
            cout << "\nloss_n0: " << loss_n0;
            cout << "\nloss_n1: " << loss_n1;
            cout << "\nloss_n2: " << loss_n2;
            cout << "\nstep length: " << beta_coeficient_update << "\n";
        }
        ++n;
    } while (loss_n2 < loss_n1);
    // **************************************************************************/

    if (verbosity > 2)
        cout << "\nFinished doubling epsilon! The monotonicity loss_n+1 < "
                "loss_n is broken!";

    // cout<<"\ny_pred_n0[2]: " << y_pred_n0[2];
    // cout<<"\ny_pred_n1[2]: " << y_pred_n1[2];
    // cout<<"\ny_pred_n2[2]: " << y_pred_n2[2];
    // Search for the beta in the range beta_n-1, beta_mid_n-1_n, beta_n,
    // beta_mid_n_n+1, beta_n+1 that minimizes the objective function. It
    // suffices to compare the 3 points beta_mid_n-1_n, beta_n, beta_mid_n_n+1,
    // as the min cannot be achieved at the extrem points of the range.
    // Take the 3 point range containing the point that achieves minimum loss.
    // Repeat until the 3 point range is too small, or a fixed number of
    // iterations is achieved.

    // **************************************************************************/
    // vector<double> y_pred_mid_n0_n1(y_pred.size());
    // vector<double> y_pred_mid_n1_n2(y_pred.size());
    if (n > 1) {
        return -(beta_coeficient_update + (pow(2, (n - 1) * 1.0) * epsilon));
    } else {
        return epsilon;
    }
    // return find_best_range_new(y_true,
    //                 y_pred_n0,
    //                 y_pred_n1,
    //                 y_pred_n2,
    //                 weak_model,
    //                 y_pred);
    // **************************************************************************/
} // end binary_line)search().

long double Gradient_boost_trainer::find_best_range_new(
    const int N, const std::vector<double> &y_true, vector<double> &y_pred_n0,
    vector<double> &y_pred_n1, vector<double> &y_pred_n2,
    const std::vector<double> &y_pred) {

    vector<double> y_pred_mid_n0_n1(y_pred.size());
    vector<double> y_pred_mid_n1_n2(y_pred.size());

    double min_range_size          = 1e-5;
    double current_range_size      = 0;
    int current_interpolation_iter = 0;

    long double loss_mid_n0_n1 = 0;
    long double loss_mid_n1_n2 = 0;
    long double loss_n1        = 0;
    for (auto docId = 0; docId < N; docId++) {
        if (verbosity > 4) {
            cout << "\ny_pred_n0[docId]: " << y_pred_n0[docId];
            cout << "\ny_pred_n1[docId]: " << y_pred_n1[docId];
            cout << "\ny_pred_n2[docId]: " << y_pred_n2[docId];
        }
        current_range_size += abs(y_pred_n2[docId] - y_pred_n0[docId]);
    }
    if (verbosity > 3)
        cout << "\ncurrent range size: " << current_range_size;

    // double beta_coef_n1 = 0;
    // double beta_coef_mid_n0_n1 = 0;
    // double beta_coef_mid_n1_n2 = 0;

    // Start interpolation loop.
    while (current_range_size > min_range_size) {
        if (verbosity > 4)
            cout << "\ncurrent interpolation iteration: "
                 << current_interpolation_iter;

        for (int i = 0; i < N; ++i) { // loop through training samples
            y_pred_mid_n0_n1[i] = (y_pred_n0[i] + y_pred_n1[i]) / 2;
            y_pred_mid_n1_n2[i] = (y_pred_n1[i] + y_pred_n2[i]) / 2;

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
        loss_mid_n0_n1     = 0;
        loss_mid_n1_n2     = 0;
        loss_n1            = 0;
        current_range_size = 0;

        for (auto docId = 0; docId < N; docId++) {
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
    double myloss = loss_funct.computeLoss(y_pred_n1, y_true);
    cout << "MyLoss: " << myloss << "\n";

    double sum = 0;
    for (int i = 0u; i < y_pred_n1.size(); i++) {
        sum += y_pred[i] - y_pred_n1[i];
    }
    cout << "AVGSTEP: " << sum / y_pred.size() << "\n";
    update_step = y_pred[0] - y_pred_n1[0];
    return update_step;
}

void Gradient_boost_trainer::transform_y(const std::vector<double> &y_orig,
                                         const std::vector<double> &y_pred,
                                         std::vector<double> &y_out) {
    loss_funct.calc_doc_gradients(y_pred, y_orig, y_out);
}

void Gradient_boost_trainer::analyse(const SEQL::Data &train_data,
                                     const SEQL::Data &test_data,
                                     std::map<string, SNode> &seed) {

    std::unordered_map<std::string, double> model;
    auto logger = std::make_unique<CSVwriter>();

    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(8);

    std::vector<double> y_pred(train_data.x.size(), 0.0);
    std::vector<double> y_pred_old(train_data.x.size(), 0.0);

    deleteUndersupportedUnigrams(seed);

    auto y_weighted = train_data.y;

    double long loss = loss_funct.computeLoss(y_pred, train_data.y);
    cout << "\nstart loss: " << loss << "\n";

    std::vector<double> trainv;
    std::vector<double> testv;

    std::vector<double> loss_trainv;
    std::vector<double> loss_testv;

    // Loop for number of given optimization iterations.
    for (unsigned int itr = 0; itr < max_itr; ++itr) {

        cout << "\n\n\n#########\nGlobal iteration: \t\t " << itr << "\n";
        // set new target to negative gradient
        transform_y(train_data.y, y_pred, y_weighted);

        if (verbosity >= 2) {
            cout << "\n"
                 << "\n";
            cout << "Y original " << train_data.y[0] << " " << train_data.y[1]
                 << " " << train_data.y[2] << " " << train_data.y[3] << "\n";
            cout << "y_pred " << y_pred[0] << " " << y_pred[1] << " "
                 << y_pred[2] << " " << y_pred[3] << "\n";
            cout << "Y weighted " << y_weighted[0] << " " << y_weighted[1]
                 << " " << y_weighted[2] << " " << y_weighted[3] << "\n";
            cout << "\n";
        }
        SEQL::Data weighted_data{train_data.x, train_data.x_sf, y_weighted};
        // SEQL::print_vector(y_weighted, "weighted_Y.csv");

        // Learn submodel with Squered error loss:
        auto [local_model, local_pred] =
            local_trainer.slim_train(weighted_data, logger.get(), seed);

        long double step_length_opt{1};
        // find stepsize for new model contribution
        if (loss_funct.objective != SEQL::SqrdL) {
            step_length_opt = line_search(
                train_data.size(), local_pred, train_data.y, y_pred);
            cout << "\nSTEP SIZE:\t\t" << step_length_opt << "\n";
            cout << "LOCALBATA[0]:\t\t" << local_pred[0] << "\n";
            step_length_opt = step_length_opt * shrinkage;
        }

        loss = loss_funct.computeLoss(y_pred, train_data.y);
        cout << "OLD LOSS:\t\t" << loss << "\n";

        // Update global model
        y_pred_old.assign(y_pred.begin(), y_pred.end());
        std::transform(y_pred.begin(),
                       y_pred.end(),
                       local_pred.begin(),
                       y_pred.begin(),
                       [&](double current, double lpred) {
                           return current + step_length_opt * lpred;
                       });

        loss = loss_funct.computeLoss(y_pred, train_data.y);
        cout << "NEW LOSS:\t\t" << loss << "\n";

        // add new weights to the model
        for (auto ele : local_model) {
            SEQL::insert_or_add(model, ele.first, step_length_opt * ele.second);
        }

        auto trainstats = SEQL::Eval::eval_classifier(train_data.y, y_pred);
        // Create model and predict testset
        SEQL::LinearModel lmodel(model, true);
        SEQL::tune(train_data, lmodel);

        auto predictions = lmodel.predict(test_data.x);
        auto stats = SEQL::Eval::eval_classifier(test_data.y, predictions);

        trainv.push_back(trainstats["Accuracy"]);
        testv.push_back(stats["Accuracy"]);

        loss_trainv.push_back(loss);
        loss_testv.push_back(loss_funct.computeLoss(test_data.y, predictions));
        // if (SEQL::hit_stoping_criterion(y_pred, y_pred_old, conv_threshold))
        // {
        //     break;
        // }
    }
    SEQL::print_vector(trainv, "trainacc", "\n");
    SEQL::print_vector(testv, "testacc", "\n");
    SEQL::print_vector(loss_trainv, "trainloss", "\n");
    SEQL::print_vector(loss_testv, "testloss", "\n");
}

std::vector<double>
Gradient_boost_trainer::calculate_init_model(const std::vector<double> &y) {
    switch (loss_funct.objective) {
    case SEQL::SqrdL: {
        auto y_mean =
            std::accumulate(std::begin(y), std::end(y), 0.0) / y.size();
        std::vector<double> ret(y.size(), y_mean);
        return ret;
    }
    case SEQL::SLR: {
        auto y_mean =
            std::accumulate(std::begin(y), std::end(y), 0.0) / y.size();
        double init = 0.5 * (1 + y_mean) / (1 - y_mean);
        std::vector<double> ret(y.size(), init);
        return ret;
    }
    default:
        return std::vector<double>(y.size(), 0.0);
    }
}

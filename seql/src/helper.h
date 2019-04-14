
#ifndef SEQL_HELPER_H
#define SEQL_HELPER_H

#include "seql_learn.h"
#include <chrono>
#include <iomanip>
#include <iostream>

using nlohmann::json;
using SEQL::Configuration;
using SEQL::Data;

namespace SEQL {

template <typename T, typename P>
std::function<void(T, P, int)> save_model(const int every_n,
                                          const std::string basename) {
    return [&every_n, &basename](T list, const P pred, int itr) {
        if (itr % every_n == 0) {
            std::string filename{basename + '_' + std::to_string(itr)};
            std::ofstream ofs{filename};
            for (const auto &e : list) {
                ofs << e.first << ' ' << e.second << '\n';
            }
        }
    };
}

template <typename T, typename P>
std::function<void(T, P, int)>
val_to_stream(const SEQL::Data &train_data, const SEQL::Data &val_data,
              const bool use_char_token, const SEQL::Loss loss_fct,
              std::ostream &os) {
    return
        [&train_data, &val_data, &use_char_token, loss_fct, &os, first = true](
            T list, const P &train_predictions, int itr) mutable {
            SEQL::LinearModel model(list, val_data.x_sf.size(), use_char_token);
            SEQL::tune(train_data, train_predictions, model);
            auto predictions = model.predict(val_data);
            cout << "VALIDATION: \n";
            cout << "Y original " << val_data.y[0] << " " << val_data.y[1]
                 << " " << val_data.y[2] << " " << val_data.y[3] << "\n";
            cout << "y_pred " << predictions[0] << " " << predictions[1] << " "
                 << predictions[2] << " " << predictions[3] << "\n";
            cout << "\n";

            auto loss         = loss_fct.computeLoss(predictions, val_data.y);
            double margin_avg = std::inner_product(predictions.begin(),
                                                   predictions.end(),
                                                   val_data.y.begin(),
                                                   0.0) /
                                val_data.y.size();
            auto evaluation_stats =
                SEQL::Eval::eval_classifier(val_data.y, predictions);
            evaluation_stats.insert({"loss", loss});
            evaluation_stats.insert({"iteration", itr});
            evaluation_stats.insert({"magrin_avg", margin_avg});
            json e_j(evaluation_stats);

            // std::cout << "EVAL: " << e_j << std::endl;
            if (!first) {
                os << ",\n";
            }
            os << std::setw(4) << e_j;
            first = false;
        };
}

// Binary classification training and evaluation helper funtion.
template <typename Learner,
          typename T = std::unordered_map<std::string, double>,
          typename P = std::vector<double>>
nlohmann::json
train_eval(Learner &learner, const SEQL::Data &train_data,
           const SEQL::Data &test_data, const Configuration config,
           const bool quiet                                = false,
           std::function<void(T, P, int)> itr_end_callback = nullptr) {

    auto train_start = std::chrono::high_resolution_clock::now();
    std::map<std::string, SNode> seed = prepareInvertedIndex(train_data.x);

    auto [list, train_prediction] =
        learner.train(train_data, config.csv_file, seed, itr_end_callback);

    SEQL::LinearModel model(list, train_data.num_sf(), config.use_char_token);

    SEQL::tune(train_data, train_prediction, model);

    auto train_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> training_elapsed = train_end - train_start;

    auto pred_start  = std::chrono::high_resolution_clock::now();
    auto predictions = model.predict(test_data);
    auto pred_end    = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> pred_elapsed = pred_end - pred_start;

    auto evaluation_stats =
        SEQL::Eval::eval_classifier(test_data.y, predictions);

    if (!quiet) {
        std::ofstream train_pred_file(config.train_prediction_file,
                                      std::ios::out);
        for (decltype(train_prediction.size()) i   = 0,
                                               end = train_prediction.size();
             i < end;
             ++i) {
            train_pred_file << train_data.y[i] << ' ' << train_prediction[i]
                            << '\n';
        }

        model.print_model(config.model_file);

        std::ofstream pred_file(config.prediction_file, std::ios::out);
        for (decltype(predictions.size()) i = 0; i < predictions.size(); ++i) {
            pred_file << test_data.y[i] << ' ' << predictions[i] << '\n';
        }
        // Get classification stats save as json
        json j_stats(evaluation_stats);
        j_stats["training_time"]   = training_elapsed.count();
        j_stats["prediction_time"] = pred_elapsed.count();
        j_stats["configuation"]    = config;
        std::cout << j_stats.dump(4) << "\n";

        ofstream jsonFile;
        jsonFile.open(config.basename + ".eval.json");
        jsonFile << std::setw(4) << j_stats;
        jsonFile.close();
    }
    return evaluation_stats;
}

template <typename Learner>
nlohmann::json
train_eval_regression(Learner &learner, const SEQL::Data &train_data,
                      const SEQL::Data &test_data, const Configuration config,
                      const bool quiet = false) {
    auto train_start = std::chrono::high_resolution_clock::now();
    std::map<std::string, SNode> seed = prepareInvertedIndex(train_data.x);
    auto [list, train_prediction] =
        learner.train(train_data, config.csv_file, seed);

    SEQL::LinearModel model(list, train_data.num_sf(), config.use_char_token);
    auto train_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> training_elapsed = train_end - train_start;

    auto pred_start  = std::chrono::high_resolution_clock::now();
    auto predictions = model.predict(test_data);
    auto pred_end    = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> pred_elapsed = pred_end - pred_start;

    auto evaluation_stats =
        SEQL::Eval::eval_regressor(test_data.y, predictions);

    if (!quiet) {
        std::ofstream train_pred_file(config.train_prediction_file,
                                      std::ios::out);
        for (decltype(train_prediction.size()) i   = 0,
                                               end = train_prediction.size();
             i < end;
             ++i) {
            train_pred_file << train_data.y[i] << ' ' << train_prediction[i]
                            << '\n';
        }

        model.print_model(config.model_file);

        std::ofstream pred_file(config.prediction_file, std::ios::out);
        for (decltype(predictions.size()) i = 0; i < predictions.size(); ++i) {
            pred_file << test_data.y[i] << ' ' << predictions[i] << '\n';
        }
        // Get classification stats save as json
        json j_stats(evaluation_stats);
        j_stats["training_time"]   = training_elapsed.count();
        j_stats["prediction_time"] = pred_elapsed.count();
        j_stats["configuation"]    = config;
        std::cout << j_stats.dump(4) << "\n";

        ofstream jsonFile;
        jsonFile.open(config.basename + ".eval.json");
        jsonFile << std::setw(4) << j_stats;
        jsonFile.close();
    }
    return evaluation_stats;
}

template <typename Learner,
          typename T = std::unordered_map<std::string, double>,
          typename P = std::vector<double>>
nlohmann::json multiclass_train_eval(
    Learner &learner, const SEQL::Data &mc_train_data,
    const SEQL::Data &test_data, const Configuration config,
    const bool quiet                                = false,
    std::function<void(T, P, int)> itr_end_callback = nullptr) {

    auto seed_start = std::chrono::high_resolution_clock::now();
    std::map<std::string, SNode> seed = prepareInvertedIndex(mc_train_data.x);
    auto seed_end = std::chrono::high_resolution_clock::now();
    long long total_seed =
        std::chrono::duration_cast<std::chrono::milliseconds>(seed_end -
                                                              seed_start)
            .count();

    int num_class =
        std::set<double>(std::begin(mc_train_data.y), std::end(mc_train_data.y))
            .size();
    std::cout << "Number of classes: " << num_class << '\n';

    // Mutliclass problem trained in a one vs all fashion
    std::vector<std::vector<double>> predictions;

    long long total_train = 0;
    long long total_pred  = 0;
    for (int classnr = 0; classnr < num_class; classnr++) {
        auto train_start = std::chrono::system_clock::now();
        Configuration local_conf(config);
        local_conf.set_basename(config.basename + "_cl" +
                                std::to_string(classnr));
        Data train_data(mc_train_data);
        std::transform(mc_train_data.y.begin(),
                       mc_train_data.y.end(),
                       train_data.y.begin(),
                       [=](int label) { return label == classnr ? 1 : -1; });

        Seql_trainer seql_learner{local_conf};

        auto [list, train_prediction] = learner.train(
            train_data, local_conf.csv_file, seed, itr_end_callback);

        SEQL::LinearModel model(
            list, train_data.num_sf(), local_conf.use_char_token);

        auto train_end = std::chrono::high_resolution_clock::now();
        total_train += std::chrono::duration_cast<std::chrono::milliseconds>(
                           train_end - train_start)
                           .count();

        auto pred_start        = std::chrono::high_resolution_clock::now();
        auto local_predictions = model.predict(test_data);
        predictions.push_back(local_predictions);
        auto pred_end = std::chrono::high_resolution_clock::now();
        total_pred += std::chrono::duration_cast<std::chrono::milliseconds>(
                          pred_end - pred_start)
                          .count();

        model.print_model(local_conf.model_file);
    }

    auto pred_start = std::chrono::high_resolution_clock::now();
    auto pred       = SEQL::get_max_entry_column_matrix(predictions);
    auto pred_end   = std::chrono::high_resolution_clock::now();
    total_pred += std::chrono::duration_cast<std::chrono::milliseconds>(
                      pred_end - pred_start)
                      .count();

    auto cm = SEQL::Eval::confusion_matrix(test_data.y, pred, num_class);
    cm.print();
    auto accuracy    = SEQL::Eval::accuracy(cm);
    auto F1_macro    = SEQL::Eval::f1_score(cm, SEQL::Eval::macro);
    auto F1_weighted = SEQL::Eval::f1_score(cm, SEQL::Eval::weighted);
    std::cout << "Accuracy: " << accuracy << '\n';
    std::cout << "F1 (macro): " << F1_macro << '\n';
    std::cout << "F1 (weighted): " << F1_weighted << '\n';
    // Save conc prediciton to file
    std::ofstream pred_file(config.prediction_file, std::ios::out);
    for (decltype(pred.size()) i = 0; i < pred.size(); ++i) {
        pred_file << test_data.y[i] << ' ' << pred[i] << '\n';
    }

    json j_stats;
    j_stats["Accuracy"]           = accuracy;
    j_stats["training_time"]      = total_train / 1000;
    j_stats["seed_creation_time"] = total_seed / 1000;
    j_stats["prediction_time"]    = total_pred / 1000;
    j_stats["total_time"]      = (total_train + total_seed + total_pred) / 1000;
    j_stats["configuation"]    = config;
    j_stats["F1_macro"]        = F1_macro;
    j_stats["F1_weighted"]     = F1_weighted;
    j_stats["Confusionmatrix"] = cm; // Prints it columnwise!
    if (!quiet) {
        std::cout << j_stats.dump(4) << "\n";
    }

    ofstream jsonFile(config.basename + ".eval.json");
    jsonFile << std::setw(4) << j_stats;
    return j_stats;
}
} // namespace SEQL
#endif

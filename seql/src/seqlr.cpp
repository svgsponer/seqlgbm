/*
 * Author: Severin Gsponer (severin.gsponer@insight-centre.org)
 * Programm to start the regression pipeline
 * At least provide training file, testfile and a basename for all the output
 * files.
 */

#include "CLI11.hpp"
#include "helper.h"
#include "preprocessing.h"
#include "seql.h"
#include "seql_boost.h"
#include "seql_learn.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <thread>

using nlohmann::json;
using SEQL::Configuration;
using SEQL::Data;

void run(const Data &train_data, const Data &test_data, Configuration config,
         std::function<void(std::unordered_map<std::string, double>,
                            std::vector<double>, int)>
             f        = nullptr,
         bool run_GBM = false) {

    if (f != nullptr) {
        std::cout << "Print evaluation on validation set is ON\n";
    } else {
        std::cout << "Print evaluation on validation set is OFF\n";
    }

    json j;
    json j_config = config;

    std::ofstream configFile;
    configFile.open(config.basename + ".cfg.json");
    configFile << j_config;
    configFile.close();

    if (run_GBM) {
        std::cout << "Run gradient boosting machine\n";
        auto cfg         = config.objective;
        config.objective = SEQL::SqrdL;
        Seql_trainer seql_learner{config};
        config.objective = cfg;
        Gradient_boost_trainer gbm(seql_learner,
                                   config.objective,
                                   config.verbosity,
                                   config.max_itr_gbm,
                                   config.shrinkage,
                                   config.convergence_threshold,
                                   config.csv_log);
        SEQL::train_eval(gbm, train_data, test_data, config, false, f);
    } else {
        Seql_trainer seql_learner{config};
        SEQL::train_eval(seql_learner, train_data, test_data, config, false, f);
    }
}

int main(int argc, char **argv) {

    CLI::App app("SEQL runner");
    app.set_help_all_flag("--help-all", "Expand all help");

    bool run_GBM{false};
    bool use_validate{false};
    // auto val = app.add_subcommand("Validate", "Evalate on validation set");
    app.add_flag("--GBM", run_GBM, "Run GBM");
    app.add_flag(
        "--Validate", use_validate, "Print Validation stats in each iteration");
    // app.add_flag("--CV", mode, "Run CV");
    // app.add_flag("--ISF", mode, "Ignore static features");

    std::string train_file;
    app.add_option("-t,--trainfile", train_file, "Training file")
        ->check(CLI::ExistingFile)
        ->required();

    std::string test_file;
    app.add_option("-s,--testfile", test_file, "Test file")
        ->check(CLI::ExistingFile)
        ->required();

    std::string val_file;
    app.add_option("-v,--validationfile", val_file, "Validation file")
        ->check(CLI::ExistingFile);

    std::string config_file;
    app.add_option("-c,--config", config_file, "Config file")
        ->check(CLI::ExistingFile)
        ->required();

    std::string base_name("SEQLGBM");
    app.add_option("-n,--name", base_name, "Basename");

    CLI11_PARSE(app, argc, argv);
    if (use_validate) {
        if (val_file.empty()) {
            std::cout << "Need validation file" << std::endl;
            std::exit(-1);
        }
    }
    Configuration config;
    config.use_char_token = true;

    // Read configuration from json configuration file if a given
    std::ifstream i(config_file);
    json j;
    i >> j;
    config = j;

    config.csv_log    = true;
    config.train_file = train_file;
    config.test_file  = test_file;
    config.set_basename(base_name);

    std::cout << std::setw(4) << config << std::endl;

    std::cout << "Preprocess data: "
              << "\n";
    Data train_data_raw = SEQL::read_input(train_file);
    auto standardizer   = SEQL::Preprocessing::fit_apply_transformer<
        SEQL::Preprocessing::StandardScaler>(train_data_raw.x_sf.begin(),
                                             train_data_raw.x_sf.end() - 1);

    Data test_data_raw = SEQL::read_input(test_file);
    SEQL::Preprocessing::apply_transformer(
        test_data_raw.x_sf.begin(), test_data_raw.x_sf.end() - 1, standardizer);

    std::cout << "Train data:\n";
    print_class_stats(train_data_raw);
    std::cout << "Test data:\n";
    print_class_stats(test_data_raw);

    if (!use_validate) {
        run(train_data_raw, test_data_raw, config, nullptr, run_GBM);
    } else {
        Data val_data_raw = SEQL::read_input(val_file);
        SEQL::Preprocessing::apply_transformer(val_data_raw.x_sf.begin(),
                                               val_data_raw.x_sf.end() - 1,
                                               standardizer);
        std::cout << "Validation data:\n";
        print_class_stats(val_data_raw);

        // Prepare itr call back function
        std::ofstream os("validation.json");
        os << "[\n";
        auto f = SEQL::val_to_stream<std::unordered_map<std::string, double>,
                                     std::vector<double>>(
            train_data_raw,
            val_data_raw,
            config.use_char_token,
            SEQL::Loss{config.objective},
            os);
        run(train_data_raw, test_data_raw, config, f, run_GBM);

        os << "]";
    }
    return 0;
}

#ifndef GRADIENT_BOOST_H
#define GRADIENT_BOOST_H

#include "linear_model.h"
#include "seql.h"
#include "seql_learn.h"
#include <numeric>

class Gradient_boost_trainer {

  private:
    Seql_trainer local_trainer;
    SEQL::Loss loss_funct{};
    int verbosity{1};
    unsigned int max_itr;
    double shrinkage{1};
    double conv_threshold{0.01};
    bool csv_log{false};
    long double line_search(const int N, const std::vector<double> &weak_model,
                            const std::vector<double> &y_true,
                            const std::vector<double> &y_pred);

    long double find_best_range_new(const int N,
                                    const std::vector<double> &y_true,
                                    std::vector<double> &y_pred_n0,
                                    std::vector<double> &y_pred_n1,
                                    std::vector<double> &y_pred_n2,
                                    const std::vector<double> &y_pred);

    void transform_y(const std::vector<double> &y_orig,
                     const std::vector<double> &y_pred,
                     std::vector<double> &y_grad);

    std::vector<double> calculate_init_model(const std::vector<double> &y);

  public:
    Gradient_boost_trainer(Seql_trainer &local_trainer, SEQL::Loss loss_funct,
                           int verbosity, unsigned int max_itr,
                           double shrinkage, double conv_threshold,
                           bool csv_log) :
        local_trainer{local_trainer},
        loss_funct{loss_funct}, verbosity{verbosity}, max_itr{max_itr},
        shrinkage{shrinkage}, conv_threshold{conv_threshold}, csv_log{
                                                                  csv_log} {};

    template <typename T = std::unordered_map<std::string, double>,
              typename P = std::vector<double>>
    std::tuple<T, P>
    train(const SEQL::Data &train_data, std::string csvFile,
          std::map<string, SNode> &seed,
          std::function<void(T, P, int)> itr_end_callback = nullptr);

    void analyse(const SEQL::Data &train_data, const SEQL::Data &test_data,
                 std::map<string, SNode> &seed);
};

template <typename T, typename P>
std::tuple<T, P>
Gradient_boost_trainer::train(const SEQL::Data &train_data, std::string csvFile,
                              std::map<string, SNode> &seed,
                              std::function<void(T, P, int)> itr_end_callback) {
    if (local_trainer.get_objective() != SEQL::SqrdL) {
        std::cerr << "Local trainer does not use Squared error, are you sure "
                     "you want that?\n";
    }

    T model;
    auto logger = std::make_unique<CSVwriter>();
    if (csv_log) {
        logger = std::make_unique<CSVwriter>(csvFile + ".gbm");
        logger->DoLog("Iteration,#Features,loss,Accuracy,avg_margin");
    }
    auto local_logger = std::make_unique<CSVwriter>();
    if (csv_log) {
        local_logger = std::make_unique<CSVwriter>(csvFile);
        local_logger->DoLog(
            "Iteration,#Features,#rewritten,#pruned,total,optimalStepLength,"
            "symbol,loss,regLoss,convRate,timestamp");
    }
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(8);

    // std::vector<double> y_pred(train_data.x.size(), 0.0);
    // std::vector<double> y_pred_old(train_data.x.size(), 0.0);

    deleteUndersupportedUnigrams(seed);

    auto pseudo_response = train_data.y;

    // Setup init model
    std::vector<double> y_pred = calculate_init_model(train_data.y);
    auto y_pred_old{y_pred};
    cout << "SF in Boost: " << train_data.x_sf.size() << std::endl;
    SEQL::insert_or_add(model,
                        "*" + std::to_string(train_data.x_sf.size() - 1) + "*",
                        y_pred[0]);

    double long loss = loss_funct.computeLoss(y_pred, train_data.y);
    cout << "\nstart loss: " << loss << "\n";
    // Loop for number of given optimization iterations.
    for (unsigned int itr = 0; itr < max_itr; ++itr) {

        cout << "\n\n\n#########\nGlobal iteration: \t\t " << itr << "\n";
        // set new target to negative gradient
        transform_y(train_data.y, y_pred, pseudo_response);

        if (verbosity >= 2) {
            cout << "\n"
                 << "\n";
            cout << "Y original " << train_data.y[0] << " " << train_data.y[1]
                 << " " << train_data.y[2] << " " << train_data.y[3] << "\n";
            cout << "y_pred " << y_pred[0] << " " << y_pred[1] << " "
                 << y_pred[2] << " " << y_pred[3] << "\n";
            cout << "pseudo response " << pseudo_response[0] << " "
                 << pseudo_response[1] << " " << pseudo_response[2] << " "
                 << pseudo_response[3] << "\n";
            cout << "\n";
        }
        SEQL::Data weighted_data{
            train_data.x, train_data.x_sf, pseudo_response};
        // SEQL::print_vector(pseudo_response, "weighted_Y.csv");

        // Learn submodel with Squered error loss:
        auto [local_model, local_pred] =
            local_trainer.slim_train(weighted_data, local_logger.get(), seed);

        long double step_length_opt{1};
        // find stepsize for new model contribution
        if (loss_funct.objective != SEQL::SqrdL) {
            step_length_opt = line_search(
                train_data.size(), local_pred, train_data.y, y_pred);
            cout << "\nSTEP SIZE:\t\t" << step_length_opt << "\n";
            cout << "LOCALPRED[0]:\t\t" << local_pred[0] << "\n";
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

        if (SEQL::hit_stoping_criterion(y_pred, y_pred_old, conv_threshold)) {
            std::cout << "Reached conv_threshold!"
                      << "\n";
            break;
        }

        if (csv_log) {
            double margin_avg =
                std::inner_product(
                    y_pred.begin(), y_pred.end(), train_data.y.begin(), 0.0) /
                train_data.y.size();
            auto evs = SEQL::Eval::eval_classifier(train_data.y, y_pred);
            logger->DoLog(itr, model.size(), loss, evs["Accuracy"], margin_avg);
        }
        if (itr_end_callback != nullptr)
            itr_end_callback(model, y_pred, itr);
    }
    std::cout << "Finish boosting"
              << "\n";
    return std::make_tuple(model, y_pred);
}

// template<typename T>
// bool validation_loss(const SEQL::Dat&a valset, const T& model, const
// SEQL::Loss& loss_funct){
//         SEQL::LinearModel lmodel(model, true);
//         auto predictions = lmodel.predict(test_data.x);
//         loss_funct.computeLoss(prediction, valset.y);

// }

#endif

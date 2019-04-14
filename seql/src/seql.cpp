#include "seql.h"

using nlohmann::json;
using std::get;
namespace fs = std::filesystem;

void SEQL::Configuration::set_basename(const std::string &new_basename) {
    basename              = new_basename;
    csv_file              = new_basename + ".csv";
    model_creation_file   = new_basename + ".modelCreation";
    model_bin_file        = new_basename + ".bin";
    model_file            = new_basename + ".model";
    prediction_file       = new_basename + ".conc.pred";
    train_prediction_file = new_basename + ".train.conc.pred";
    stats_file            = new_basename + ".stats.json";
}

void SEQL::to_json(nlohmann::json &j, const SEQL::Configuration &c) {
    j = nlohmann::json{{"objective", c.objective},
                       {"use_char_token", c.use_char_token},
                       {"maxpat", c.maxpat},
                       {"minpat", c.minpat},
                       {"maxitr", c.maxitr},
                       {"minsup", c.minsup},
                       {"maxgap", c.maxgap},
                       {"maxcongap", c.maxcongap},
                       {"use_bfs", c.use_bfs},
                       {"C", c.C},
                       {"alpha", c.alpha},
                       {"convergence_threshold", c.convergence_threshold},
                       {"verbosity", c.verbosity},
                       {"csv_log", c.csv_log},
                       {"mean", c.mean},
                       {"max_itr_gbm", c.max_itr_gbm},
                       {"train_file", c.train_file},
                       {"test_file", c.test_file},
                       {"basename", c.basename},
                       {"csv_file", c.csv_file},
                       {"model_creation_file", c.model_creation_file},
                       {"model_bin_file", c.model_bin_file},
                       {"model_file", c.model_file},
                       {"prediction_file", c.prediction_file},
                       {"train_prediction_file", c.train_prediction_file},
                       {"stats_file", c.stats_file},
                       {"shrinkage", c.shrinkage}};
}

void SEQL::from_json(const nlohmann::json &j, SEQL::Configuration &c) {
    c.objective             = j.at("objective").get<int>();
    c.use_char_token        = j.at("use_char_token").get<bool>();
    c.maxpat                = j.at("maxpat").get<int>();
    c.minpat                = j.at("minpat").get<int>();
    c.maxitr                = j.at("maxitr").get<int>();
    c.minsup                = j.at("minsup").get<int>();
    c.maxgap                = j.at("maxgap").get<int>();
    c.maxcongap             = j.at("maxcongap").get<int>();
    c.use_bfs               = j.at("use_bfs").get<bool>();
    c.C                     = j.at("C").get<double>();
    c.alpha                 = j.at("alpha").get<double>();
    c.convergence_threshold = j.at("convergence_threshold").get<double>();
    c.verbosity             = j.at("verbosity").get<int>();
    c.csv_log               = j.at("csv_log").get<bool>();
    c.mean                  = j.at("mean").get<double>();
    c.max_itr_gbm           = j.at("max_itr_gbm").get<int>();
    c.shrinkage             = j.at("shrinkage").get<double>();
    // c.train_file = j.at("train_file").get<std::string>();
    // c.test_file = j.at("test_file").get<std::string>();
    // c.basename = j.at("basename").get<std::string>();
    // c.csv_file = j.at("csv_file").get<std::string>();
    // c.model_crj.ation_file =
    // .at("model_creation_file").get<std::string>(); c.model_bin_file =
    // j.at("model_bin_file").get<std::string>(); c.model_file =
    // j.at("model_file").get<st>(); c.prediction_file =
    // j.at("prediction_file").get<int>(); c.sj.ats_file =
    // .at("stats_file").get<int>();
}

std::ostream &SEQL::operator<<(std::ostream &os,
                               const SEQL::Configuration &config) {
    json config_j(config);
    os << config_j;
    return os;
}

long double SEQL::add_regularization(const double loss,
                                     const SEQL::regularization_param rp,
                                     const double old_bc, const double new_bc) {

    return loss + rp.C * ((rp.alpha * (rp.sum_abs_betas - std::abs(old_bc) +
                                       std::abs(new_bc))) +
                          ((1 - rp.alpha) * 0.5 *
                           (rp.sum_squared_betas - std::pow(old_bc, 2) +
                            std::pow(new_bc, 2))));
}

long double SEQL::Loss::computeLossTerm(const double &y_pred,
                                        const double &y_true) const {
    switch (objective) {
    case SLR:
        if (y_true * y_pred > 14) {
            // if (-y_true * y_pred > 8000) {
            // return log(LDBL_MAX);
            // Set to 0 as probably is nearly 1
            return 0;
        } else {
            return log(1 + exp(-y_true * y_pred));
        }
    case l1SVM:
        if (1 - y_true * y_pred > 0)
            return (1 - y_true * y_pred);
        else
            return 0;
    case l2SVM:
        if (1 - y_true * y_pred > 0)
            return pow(1 - y_true * y_pred, 2);
        else
            return 0;
    case SqrdL:
        return pow(y_true - y_pred, 2);
    case MAE:
        return std::abs(y_true - y_pred);
    // case EXP:
    //     if (y_true * y_pred > 14) {
    //         return 0;
    //     } else {
    //         return exp(-y_true * y_pred);
    //     }
    default:
        return 0;
    };
}

/*
long double SEQL::Loss::computeLossTerm(const double &y_pred,
                                        const double &y_true,
                                        long double &exp_fraction) {
    switch (objective) {
    case SLR:
        if (y_true * y_pred > 8000) {
            exp_fraction = 0;
        } else {
            exp_fraction = 1 / (1 + exp(y_true * y_pred));
        }
        if (-y_true * y_pred > 8000) {
            return log(LDBL_MAX);
        } else {
            return log(1 + exp(-y_true * y_pred));
        }
    case l1SVM:
        if (1 - y_true * y_pred > 0)
            return (1 - y_true * y_pred);
        else
            return 0;
    case l2SVM:
        if (1 - y_true * y_pred > 0)
            return pow(1 - y_true * y_pred, 2);
        else
            return 0;
    case SqrdL:
        return pow(y_true - y_pred, 2);
    case MAE:
        return std::abs(y_true - y_pred);
    default:
        return 0;
    };
}
*/

/* CURRENTLY NOT USED
// Updates terms of loss function that chagned. vector<> loc contains
documnets
// which loss functions changed
void SEQL::Loss::updateLoss(const std::vector<double> &y_true,
                            long double &loss,
                            const std::vector<double> &y_pred_opt,
                            const std::vector<double> &y_pred,
                            const std::vector<unsigned int> loc) {
    auto s = y_true.size();
    for (auto docid : loc) {
        loss -= computeLossTerm(y_pred[docid], y_true[docid]) / s;
        loss += computeLossTerm(y_pred_opt[docid], y_true[docid]) / s;
    }
}
*/

double SEQL::Loss::computeLoss(const std::vector<double> &y_pred,
                               const std::vector<double> &y_true) const {
    double loss = 0;
    for (unsigned int i = 0; i < y_true.size(); ++i) {
        loss += computeLossTerm(y_pred[i], y_true[i]);
    }
    return loss / y_true.size();
}

void SEQL::Loss::calc_doc_gradients(const std::vector<double> &y_pred,
                                    const std::vector<double> &y_true,
                                    std::vector<double> &gradients) const {
    switch (objective) {
    case SLR:
        std::transform(y_pred.begin(),
                       y_pred.end(),
                       y_true.begin(),
                       gradients.begin(),
                       [](const double prediction, const double y) {
                           if (y * prediction > 14) {
                               return 0.0;
                           } else {
                               return (y / (1 + exp(y * prediction)));
                           }
                       });
        break;
    case l1SVM:
        std::transform(y_pred.begin(),
                       y_pred.end(),
                       y_true.begin(),
                       gradients.begin(),
                       [](const double prediction, const double y) {
                           if (1 - y * prediction > 0) {
                               return y;
                           } else {
                               return 0.0;
                           }
                       });
        break;
    case l2SVM:
        std::transform(y_pred.begin(),
                       y_pred.end(),
                       y_true.begin(),
                       gradients.begin(),
                       [](const double prediction, const double y) {
                           if (1 - y * prediction > 0) {
                               return 2 * y * (1 - y * prediction);
                           } else {
                               return 0.0;
                           }
                       });
        break;
    case SqrdL:
        std::transform(y_pred.begin(),
                       y_pred.end(),
                       y_true.begin(),
                       gradients.begin(),
                       [](const double prediction, const double y) {
                           return (y - prediction);
                       });
        break;
    case MAE:
        std::transform(y_pred.begin(),
                       y_pred.end(),
                       y_true.begin(),
                       gradients.begin(),
                       [](const double prediction, const double y) {
                           return -((0 < (prediction - y)) -
                                    ((prediction - y) < 0));
                       });
        break;
        // case
        //     EXP:
        //     std::transform(y_pred.begin(),
        //                    y_pred.end(),
        //                    y_true.begin(),
        //                    gradients.begin(),
        //                    [](const double prediction, const double y) {
        //                        if (y * prediction > 14) {
        //                            return 0.0;
        //                        } else {
        //                            return (y * exp(-y * prediction));
        //                        }
        //                    });
        //     break;
    };
}

std::vector<double> SEQL::Loss::calc_sf_gradient(
    const std::vector<double> &y_pred, const std::vector<double> &y_true,
    const std::vector<std::vector<double>> &sfs) const {
    std::vector<double> gradients(y_pred.size());
    std::vector<double> gradient;
    calc_doc_gradients(y_pred, y_true, gradients);

    // Multiply feature vector with "gradient vector"
    for (auto const sf : sfs) {
        double grad = std::inner_product(gradients.begin(),
                                         gradients.end(),
                                         sf.begin(),
                                         0.0,
                                         std::plus<double>(),
                                         std::multiplies<double>());
        gradient.push_back((-1) * grad / y_true.size());
    }
    return gradient;
}

double
SEQL::Loss::calc_intercept_gradient(const std::vector<double> &y_pred,
                                    const std::vector<double> &y_true) const {
    // std::vector<double> gradients(y_pred.size());
    // calc_doc_gradients(y_pred, y_true, gradients);
    // double gradient = std::accumulate(
    //     gradients.begin(), gradients.end(), 0.0, std::minus<double>());
    std::vector<std::vector<double>> int_vec;
    int_vec.emplace_back(y_pred.size(), 1.0);
    return calc_sf_gradient(y_pred, y_true, int_vec)[0];
}

double SEQL::update_convthreshold(const std::vector<double> &y_pred,
                                  const std::vector<double> &y_pred_opt) {
    double convergence_rate;
    double sum_abs_linear_score        = 0;
    double sum_abs_linear_score_change = 0;
    for (auto i = 0u; i < y_pred_opt.size(); i++) {
        sum_abs_linear_score += std::abs(y_pred_opt[i]);
        sum_abs_linear_score_change += std::abs(y_pred_opt[i] - y_pred[i]);
    }
    convergence_rate = sum_abs_linear_score_change / (1 + sum_abs_linear_score);
    return convergence_rate;
}

bool SEQL::hit_stoping_criterion(const double convergence_rate,
                                 const double convergence_threshold) {
    if (convergence_rate <= convergence_threshold) {
        return true;
    }
    return false;
}

bool SEQL::hit_stoping_criterion(const std::vector<double> &y_pred,
                                 const std::vector<double> &y_pred_old,
                                 const double convergence_threshold) {
    double convergence_rate = update_convthreshold(y_pred, y_pred_old);
    return SEQL::hit_stoping_criterion(convergence_rate, convergence_threshold);
}

SEQL::SeqlFileHeader SEQL::parse_seql_file_header(const char *hlc) {
    std::stringstream hl(hlc);
    SEQL::SeqlFileHeader sfh;
    std::string header_string;
    hl >> header_string;
    hl >> sfh.version;
    hl >> sfh.num_sf;
    hl >> sfh.use_char_token;
    std::cout << "Header string: " << header_string << "\n"
              << "Header version: " << sfh.version << "\n"
              << "Number of static features: " << sfh.num_sf << "\n"
              << "Sequence uses char tokens: " << sfh.use_char_token
              << std::endl;
    return sfh;
}

SEQL::Data SEQL::read_sax(std::istream &is, int limit) {
    constexpr int kMaxLineSize = 10000000;
    char *line{new char[kMaxLineSize]};
    SEQL::Data ret;
    SEQL::SeqlFileHeader shf;
    while (is.getline(line, kMaxLineSize) && limit != 0) {
        // Skip empty lines and commented lines
        if (line[0] == '\0' || line[0] == ';')
            continue;
        // Check for header line
        if (line[0] == '#') {
            shf = parse_seql_file_header(line);
            ret.x_sf.resize(shf.num_sf);
            continue;
        }
        // Fix line endings
        if (line[strlen(line) - 1] == '\r')
            line[strlen(line) - 1] = '\0';

        auto fields = split(line, "\t ");
        if (fields.size() < 2 + shf.num_sf) {
            std::cerr << "FATAL: Format Error: " << line << '\n'
                      << "Expected at least: " << 2 + shf.num_sf << " fields\n"
                      << "Got: " << fields.size() << '\n'
                      << std::endl;
            std::exit(1);
        }
        auto s = fields.back();
        std::vector<std::string> doc;
        for (auto w = fields.begin() + shf.num_sf + 1; w != fields.end();
             std::advance(w, 1)) {
            for (auto e : *w) {
                doc.emplace_back(1, e);
            }
            doc.push_back(" ");
        }
        ret.x.push_back(doc);

        // Parse static features to double and insert in feature vectors
        if (shf.num_sf > 0) {
            auto sf_it = ret.x_sf.begin();
            for (auto it = std::begin(fields) + 1;
                 it != fields.begin() + 1 + shf.num_sf;
                 std::advance(it, 1)) {
                (*sf_it).push_back(stod(*it));
                std::advance(sf_it, 1);
            }
        }

        // Prepare class. _y is +1/-1 for classification or a double in case
        // of regression.
        double _y = atof(fields[0].c_str());
        ret.y.push_back(_y);

        limit--;
    }
    // Add 1 vector to sf for intercept
    ret.x_sf.emplace_back(ret.size(), 1.0);

    std::cout << "# total points: " << ret.size() << '\n';

    delete[] line;

    return ret;
}
SEQL::Data SEQL::read_input(std::istream &is, int limit) {
    constexpr int kMaxLineSize = 10000000;
    char *line{new char[kMaxLineSize]};
    SEQL::Data ret;
    SEQL::SeqlFileHeader shf;
    while (is.getline(line, kMaxLineSize) && limit != 0) {
        // Skip empty lines and commented lines
        if (line[0] == '\0' || line[0] == ';')
            continue;
        // Check for header line
        if (line[0] == '#') {
            shf = parse_seql_file_header(line);
            ret.x_sf.resize(shf.num_sf);
            continue;
        }
        // Fix line endings
        if (line[strlen(line) - 1] == '\r')
            line[strlen(line) - 1] = '\0';

        auto fields = split(line, "\t ");
        if (shf.use_char_token) {
            if (fields.size() != 2 + shf.num_sf) {
                std::cerr << "FATAL: Format Error: " << line << '\n'
                          << "Expected exactly: " << 2 + shf.num_sf
                          << " fields\n"
                          << "Got: " << fields.size() << '\n'
                          << "Is there a space in the charater sequence?"
                          << std::endl;
                std::exit(1);
            }
            auto s   = fields.back();
            auto doc = SEQL::tokenize(s, true);
            ret.x.push_back(doc);
        } else {
            std::vector<std::string> doc(fields.begin() + shf.num_sf + 1,
                                         fields.end());
            ret.x.push_back(doc);
        }

        // Parse static features to double and insert in feature vectors
        if (shf.num_sf > 0) {
            auto sf_it = ret.x_sf.begin();
            for (auto it = std::begin(fields) + 1;
                 it != fields.begin() + 1 + shf.num_sf;
                 std::advance(it, 1)) {
                (*sf_it).push_back(stod(*it));
                std::advance(sf_it, 1);
            }
        }

        // Prepare class. _y is +1/-1 for classification or a double in case
        // of regression.
        double _y = atof(fields[0].c_str());
        ret.y.push_back(_y);

        limit--;
    }
    // Add 1 vector to sf for intercept
    ret.x_sf.emplace_back(ret.size(), 1.0);

    std::cout << "# total points: " << ret.size() << '\n';

    delete[] line;

    return ret;
}

SEQL::Data SEQL::read_sax(fs::path filename, int limit) {
    // Set the max line size to (10Mb).
    std::cout << "Read file: " << filename << std::endl;

    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        std::cerr << "Error: " << filename << " No such file or directory"
                  << std::endl;
        std::exit(1);
    }
    return read_sax(ifs, limit);
}
SEQL::Data SEQL::read_input(fs::path filename, int limit) {
    // Set the max line size to (10Mb).
    std::cout << "Read file: " << filename << std::endl;

    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        std::cerr << "Error: " << filename << " No such file or directory"
                  << std::endl;
        std::exit(1);
    }
    return read_input(ifs, limit);
}

std::map<int, int> SEQL::print_class_stats(SEQL::Data const &data) {
    std::map<int, int> classes;
    for (const auto &c : data.y) {
        ++classes[c];
    }
    std::cout << "Number of classes: " << classes.size() << '\n';
    for (const auto &[cls, count] : classes) {
        std::cout << "Number of samples of class " << cls << ": " << count
                  << '\n';
    }
    return classes;
}

std::tuple<double, double> SEQL::print_reg_stats(SEQL::Data const &data) {
    auto mean = std::accumulate(std::begin(data.y), std::end(data.y), 0.0);
    mean      = mean / data.y.size();
    auto sqdistance = [&](double a, double b) { return a + pow(b - mean, 2); };
    auto var =
        std::accumulate(std::begin(data.y), std::end(data.y), 0.0, sqdistance);
    var = var / data.y.size();
    std::cout << "Mean: " << mean << "\n"
              << "Variance: " << var << "\n";
    return std::make_tuple(mean, var);
}

std::vector<std::string> SEQL::tokenize(std::string doc, bool use_char_token) {
    std::vector<std::string> seq;
    if (use_char_token) {
        for (auto ele : doc) {
            if (std::isspace(ele)) {
                std::cout << "\nFATAL...found space in doc: "; // << x.size()
                                                               // ;//<< ", at
                                                               // position: " <<
                                                               // pos;
                std::cout << "\nFATAL...char-tokenization assumes "
                             "contiguous tokens "
                             "(i.e., tokens are not separated by space).";
                std::cout << "\nFor space separated tokens please use "
                             "word-tokenization or remove spaces to get valid "
                             "input for char-tokenization.";
                std::cout << "\n...Exiting.....\n";
                std::exit(-1);
            } else {
                // Ugly should be fixed by using template and create vector
                // of chars rather than strings
                // seq.push_back(std::string(static_cast<char*>(&ele)));
                seq.emplace_back(1, ele);
            }
        }
    } else {
        // Word token
        size_t end_pos = 0;
        while (end_pos != std::string::npos) {
            size_t pos = doc.find_first_not_of(' ', end_pos);
            if (pos != std::string::npos) {
                end_pos = doc.find(' ', pos);
                seq.emplace_back(doc.substr(pos, end_pos - pos));
            }
        }
    }
    return seq;
}

std::vector<double> SEQL::calculate_init_model(const std::vector<double> &y,
                                               SEQL::Lossfunction objective) {
    switch (objective) {
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

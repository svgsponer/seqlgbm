/** Linear model class
    Author: Severin Gsponer (severin.gsponer@insight-centre.com)
**/
#include "linear_model.h"

namespace SEQL {
void LinearModel::print_model(std::ostream &os) const {
    os.precision(24);
    os << intercept << " INTERCEPT" << '\n';

    std::vector<std::pair<double, std::string>> rules;

    // Map to vector to be able to sort by weigth
    for (auto rule : model) {
        rules.emplace_back(rule.second, rule.first);
    }

    std::sort(std::begin(rules), std::end(rules));

    for (auto rule : rules) {
        os << rule.first << " " << rule.second << '\n';
    }
}

void LinearModel::print_model(std::string const outfile_name) const {
    std::ofstream ofs(outfile_name, std::ios::out);

    if (!ofs) {
        std::cerr << "FATAL: Cannot open outputfile: " << outfile_name
                  << std::endl;
        exit(1);
    }
    print_model(ofs);
}

void LinearModel::insert_or_add(std::string const ngram, double const weight) {
    model[ngram] += weight;
}

void LinearModel::seperate_sf() {
    for (auto rule : model) {
        auto name = rule.first;
        if (name[0] == '*') {
            // static feature
            auto idx = std::stoi(name.substr(1, name.size() - 1));
            sf[idx]  = rule.second;
        }
    }
    intercept = sf.back();
    model.erase("*" + std::to_string(sf.size() - 1) + "*");
    // sf.pop_back();
}

void LinearModel::build_tree() {

    // threshold = 0;
    // intercept = model["*INTERCEPT*"];
    // model.erase("*INTERCEPT*");

    std::vector<Darts::DoubleArray::key_type *> ary;
    for (auto rule : model) {

        // Some string foo to hande c_string correctly
        auto ngram = rule.first;
        // Static feature
        if (ngram[0] == '*')
            continue;
        char *cstr = new char[ngram.length() + 1];
        std::strcpy(cstr, ngram.c_str());
        ary.push_back((Darts::DoubleArray::key_type *)cstr);
    }

    if (ary.empty() && sf.empty()) {
        std::cerr << "FATAL: no features in the model" << std::endl;
        exit(1);
    }

    // Darts expects sorted array of keys as input
    std::sort(std::begin(ary), std::end(ary), [](const auto a, const auto b) {
        return strcmp(a, b) < 0;
    });

    auto newDa = std::make_unique<Darts::DoubleArray>();
    if (newDa->build(ary.size(), &ary[0], nullptr, nullptr, nullptr) != 0) {
        std::cerr << "Error: cannot build double array  " << std::endl;
        exit(1);
    }
    // return newDa;
    da = std::move(newDa);
    for (auto &ele : ary) {
        delete[] ele;
    }
}

double LinearModel::l2_norm() {
    auto sum2 = std::accumulate(std::begin(model),
                                std::end(model),
                                0,
                                [](int value, const Model::value_type &p) {
                                    return value + pow(p.second, 2);
                                });
    return pow(sum2, 0.5);
}

double LinearModel::l1_norm() {
    auto sum = std::accumulate(std::begin(model),
                               std::end(model),
                               0,
                               [](int value, const Model::value_type &p) {
                                   return value + std::abs(p.second);
                               });
    return sum;
}

void LinearModel::normalize_weights() {
    const auto l1_n = l1_norm();
    for (auto &rule : model) {
        rule.second = rule.second / l1_n;
    };
}

bool operator==(const LinearModel &c1, const LinearModel &c2) {
    return (c1.model == c2.model);
}

void parse_model(LinearModel &model, std::istream &is, int limit) {
    char buf[8192];
    std::vector<std::string> column;
    std::string rule{};
    int num_sf = 0;
    while (0 != limit && is.getline(buf, 8192)) {
        if (buf[strlen(buf) - 1] == '\r') {
            buf[strlen(buf) - 1] = '\0';
        }
        column = SEQL::split(buf, "\t ");
        if (column.size() != 2) {
            std::cerr << "FATAL: Format Error: " << buf << std::endl;
            exit(1);
        }
        double curbeta = atof(column[0].c_str());
        rule.assign(column[1]);
        if (rule == "INTERCEPT") {
            model.intercept = curbeta;
        } else if (rule[0] == '*') {
            ++num_sf;
        }
        model.insert_or_add(rule, -curbeta);
        --limit;
    }

    model.seperate_sf();
    model.build_tree();
}

LinearModel parse_model(std::istream &is, int limit) {
    std::cerr << "METHOD DOES NOT SUPPORT STATIC FEATURES YET!\n";
    // LinearModel need to know how many static features there will be
    // use parse_model(LinearModel&, std::istream, in) instead where
    // LinearModel has already correct size

    LinearModel model;
    parse_model(model, is, limit);
    return model;
}

LinearModel parse_model(std::string file, int limit) {
    std::ifstream is{file.c_str()};
    if (!is.is_open()) {
        std::cerr << "FATAL: Cannot open inputfile: " << file << std::endl;
        exit(1);
    }
    return parse_model(is, limit);
}

void LinearModel::project(std::string prefix,
                          const std::vector<stx::string_symbol> &doc,
                          unsigned int pos, size_t trie_pos, size_t str_pos,
                          matches &matches) const {
    if (pos == doc.size() - 1)
        return;

    // Check traversal with both the next actual unigram in the doc and the
    // wildcard *.
    string next_unigrams[2];
    next_unigrams[0] = doc[pos + 1].key();
    next_unigrams[1] = "*";

    for (int i = 0; i < 2; ++i) {

        string next_unigram = next_unigrams[i];
        std::string item;
        if (use_char_token) { // char-level token
            item = prefix + next_unigram;
        } else { // word-level token
            item = prefix + " " + next_unigram;
        }
        size_t new_trie_pos = trie_pos;
        size_t new_str_pos  = str_pos;
        int id = da->traverse(item.c_str(), new_trie_pos, new_str_pos);

        if (id == -2) {
            if (i == 0)
                continue;
            else
                return;
        }
        if (id >= 0) {
            matches.emplace(item);
        }
        project(item, doc, pos + 1, new_trie_pos, new_str_pos, matches);
    }
}

matches LinearModel::find_matches(std::vector<std::string> const &str) const {

    matches matches;
    std::vector<stx::string_symbol> doc;

    std::copy(std::cbegin(str), std::cend(str), std::back_inserter(doc));

    for (unsigned int i = 0; i < doc.size(); ++i) {
        std::string item = doc[i].key();
        int id;
        da->exactMatchSearch(item.c_str(), id);
        if (id == -2)
            continue;
        if (id >= 0) {
            matches.emplace(item);
        }
        project(doc[i].key(), doc, i, 0, 0, matches);
    }

    return matches;
}

double LinearModel::predict(const std::vector<double> &sfl,
                            std::vector<std::string> const &str) const {

    auto sf_score =
        std::inner_product(std::begin(sfl), std::end(sfl), std::begin(sf), 0.0);
    sf_score += -threshold;
    return sf_score + predict(str);
}

double LinearModel::predict(std::vector<std::string> const &str) const {
    // Find all features that are present in str
    auto matches = find_matches(str);

    int oov_docs{0};
    if (matches.size() == 0) {
        // if (userule)
        //     cout << "\n Test doc out of vocabulary\n";
        oov_docs++;
    }
    // if (oov_docs != 0) {
    //     cerr << "OOV!" << std::endl;
    // }

    // Add up wheight of all found features as well as intercept
    // double score = -threshold;
    // score += intercept;
    double score = 0.0;
    for (const auto ngram : matches) {
        // cout << "Found ngram: " << ngram << std::endl;
        score += model.at(ngram);
    }

    return score;
}

std::vector<double> LinearModel::predict(const SEQL::Data d) const {
    if (d.x_sf.size() != 0) {
        return predict(d.x_sf, d.x);
    } else {
        return predict(d.x);
    }
}

std::vector<double>
LinearModel::predict(const std::vector<std::vector<double>> &sfv,
                     const std::vector<std::vector<std::string>> &x) const {
    std::vector<double> predictions;
    std::vector<std::vector<double>> sfv_transpose = transpose(sfv);
    std::transform(
        std::begin(x),
        std::end(x),
        std::begin(sfv_transpose),
        std::back_inserter(predictions),
        [this](std::vector<std::string> seq, std::vector<double> sfl) {
            return predict(sfl, seq);
        });
    return predictions;
}

std::vector<double>
LinearModel::predict(const std::vector<std::vector<std::string>> &x) const {
    std::vector<std::vector<double>> sfv(x.size(), std::vector<double>(1, 1.0));
    return predict(sfv, x);
}

double tune(const SEQL::Data &data, std::vector<double> predictions,
            LinearModel &model) {
    SEQL::Eval::ConfusionMatrix cm(data.y, predictions);

    auto p      = SEQL::sort_permutation(predictions);
    predictions = SEQL::apply_permutation(predictions, p);
    auto y      = SEQL::apply_permutation(data.y, p);

    // double AUC = SEQL::Eval::calcROC(y, predictions, cm.P, cm.N);

    // Choose the threshold that minimized the errors on training data.
    // Same as Madigan et al BBR.

    // Start by retrieving all, e.g. predict all as positives.
    // Compute the error as FP + FN.
    unsigned int all           = y.size();
    unsigned int TP            = cm.TP + cm.FN;
    unsigned int FP            = all - cm.TP + cm.FN;
    unsigned int FN            = 0;
    unsigned int TN            = 0;
    unsigned int min_error     = FP + FN;
    unsigned int current_error = 0;
    double best_threshold      = -std::numeric_limits<double>::max();

    for (unsigned int i = 0; i < all; ++i) {
        // Take only 1st in a string of equal values
        if (i != 0 && predictions[i] > predictions[i - 1]) {
            current_error = FP + FN; // sum of errors, e.g # training errors
            if (current_error < min_error) {
                min_error      = current_error;
                best_threshold = (predictions[i - 1] + predictions[i]) / 2;
            }
        }
        if (y[i] > 0) {
            FN++;
            TP--;
        } else {
            FP--;
            TN++;
        }
    }

    // Finally, check the "retrieve none" situation
    current_error = FP + FN;
    if (current_error < min_error) {
        min_error      = current_error;
        best_threshold = predictions[all - 1] + 1;
    }

    model.threshold = best_threshold;
    return best_threshold;
}
double tune(const SEQL::Data &data, LinearModel &model) {
    auto predictions = model.predict(data.x_sf, data.x);
    return tune(data, predictions, model);
}

double tune(std::string filename, LinearModel &model) {
    auto data = SEQL::read_input(filename);
    return tune(data, model);
}
} // namespace SEQL

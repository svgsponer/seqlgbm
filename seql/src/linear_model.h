/** Linear model class
    Author: Severin Gsponer (severin.gsponer@insight-centre.com)
**/

#ifndef LINEARMODEL_H
#define LINEARMODEL_H

#include "basic_symbol.h"
#include "common.h"
#include "darts.h"
#include "evaluation.h"
#include "unistd.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

using Model   = std::unordered_map<std::string, double>;
using matches = std::set<std::string>;

namespace SEQL {
class LinearModel {
  public:
    LinearModel() : da(new Darts::DoubleArray){};
    LinearModel(Model m) : model(m), da(new Darts::DoubleArray) {
        build_tree();
    };
    LinearModel(Model m, size_t num_sf) :
        model(m), sf(num_sf, 0.0), da(new Darts::DoubleArray) {
        seperate_sf();
        build_tree();
    };
    LinearModel(size_t num_sf) : sf(num_sf, 0.0), da(new Darts::DoubleArray){};

    LinearModel(Model m, size_t num_sf, bool use_char_token) :
        use_char_token(use_char_token), model(m), sf(num_sf, 0.0),
        da(new Darts::DoubleArray) {
        seperate_sf();
        build_tree();
    };

    long double intercept = 0.0;
    double threshold      = 0.0;

    bool use_char_token{true};

    void insert_or_add(std::string const ngram, double const weight);
    void build_tree();
    void seperate_sf();

    void normalize_weights();

    // Prediction
    std::vector<double> predict(const SEQL::Data d) const;
    std::vector<double>
    predict(const std::vector<std::vector<std::string>> &x) const;
    double predict(std::vector<std::string> const &str) const;
    virtual matches find_matches(std::vector<std::string> const &str) const;
    std::vector<double>
    predict(const std::vector<std::vector<double>> &sfv,
            const std::vector<std::vector<std::string>> &x) const;

    double predict(const std::vector<double> &sfl,
                   std::vector<std::string> const &str) const;

    void print_model(std::ostream &os) const;
    void print_model(std::string const outfile_name) const;

    auto &operator[](std::string ngram) { return model[ngram]; };
    auto operator[](std::string ngram) const { return model.at(ngram); };

    auto begin() { return model.begin(); };
    auto end() { return model.end(); };

    friend bool operator==(const LinearModel &c1, const LinearModel &c2);

  protected:
    Model model;
    // weights of static featrues
    std::vector<double> sf;
    // Tree
    std::unique_ptr<Darts::DoubleArray> da;

  private:
    double l2_norm();
    double l1_norm();

    void project(std::string prefix, const std::vector<stx::string_symbol> &doc,
                 unsigned int pos, size_t trie_pos, size_t str_pos,
                 matches &matches) const;
};

void parse_model(LinearModel &model, std::istream &is, int limit = -1);
LinearModel parse_model(std::istream &is, int limit = -1);
LinearModel parse_model(std::string file, int limit = -1);

double tune(const SEQL::Data &data, std::vector<double> predictions,
            LinearModel &model);
double tune(const SEQL::Data &d, LinearModel &model);
double tune(std::string filename, LinearModel &model);

} // namespace SEQL

#endif

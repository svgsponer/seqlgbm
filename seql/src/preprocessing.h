#include "common.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <numeric>
#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace SEQL {
namespace Preprocessing {

class StandardScaler {
  private:
    double epsilon = 1e-8;

  public:
    double mean{0};
    double var{0};
    double std{0};

    std::tuple<double, double> fit(std::vector<double> &v);
    void transform(std::vector<double> &v) const;
    void fit_transform(std::vector<double> &v);
};

class NormScaler {
  public:
    double norm{0};

    double fit(std::vector<double> &v);
    void transform(std::vector<double> &v) const;
    void fit_transform(std::vector<double> &v);
};

class LabelNormalizer {
  public:
    std::unordered_map<int, int> mapping;

    auto fit(std::vector<double> &v);
    void transform(std::vector<double> &v) const;
    void fit_transform(std::vector<double> &v);

    friend std::ostream &operator<<(std::ostream &os,
                                    const LabelNormalizer &dt);
};

template <class Transformer, class Iter>
std::vector<Transformer> fit_apply_transformer(Iter first, Iter last) {
    std::vector<Transformer> transformers;
    while (first != last) {
        Transformer t;
        t.fit_transform(*first);
        transformers.push_back(t);
        std::advance(first, 1);
    }
    return transformers;
}

template <class Transformer, class Iter>
void apply_transformer(Iter first, Iter last,
                       const std::vector<Transformer> &t) {

    if (std::distance(first, last) != t.size()) {
        std::cerr << "Error: Unequal size of data transformer and colums to "
                     "transform. "
                  << std::distance(first, last) << " data columns and "
                  << t.size() << " transformer given." << std::endl;
        std::exit(1);
    }

    auto n_it = std::begin(t);
    while (first != last) {
        n_it->transform(*first);
        std::advance(first, 1);
        std::advance(n_it, 1);
    }
}
} // namespace Preprocessing
} // namespace SEQL

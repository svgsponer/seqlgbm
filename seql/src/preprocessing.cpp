#include "preprocessing.h"

// StandardScaler
std::tuple<double, double>
SEQL::Preprocessing::StandardScaler::fit(std::vector<double> &v) {
    mean = std::accumulate(v.begin(), v.end(), 0.0);
    mean = mean / v.size();
    var  = std::accumulate(v.begin(), v.end(), 0.0, [](double a, double b) {
        return a + std::pow(b, 2);
    });
    var  = (var / std::size(v)) - std::pow(mean, 2);
    std  = std::sqrt(var);
    return std::make_tuple(mean, var);
}

void SEQL::Preprocessing::StandardScaler::transform(
    std::vector<double> &v) const {

    if (std::abs(var) < epsilon) {
        std::fill(std::begin(v), std::end(v), 0.0);
    } else {
        std::transform(
            std::begin(v),
            std::end(v),
            std::begin(v),
            [mean = mean, std = std](double a) { return (a - mean) / std; });
    }
}

void SEQL::Preprocessing::StandardScaler::fit_transform(
    std::vector<double> &v) {
    fit(v);
    transform(v);
}

// NormScaler
double SEQL::Preprocessing::NormScaler::fit(std::vector<double> &v) {
    norm = SEQL::norm2(v);
    return norm;
}

void SEQL::Preprocessing::NormScaler::transform(std::vector<double> &v) const {
    std::transform(std::begin(v),
                   std::end(v),
                   std::begin(v),
                   [norm = norm](auto x) { return x / norm; });
}

void SEQL::Preprocessing::NormScaler::fit_transform(std::vector<double> &v) {
    fit(v);
    transform(v);
}

// LabelNormalizer
auto SEQL::Preprocessing::LabelNormalizer::fit(std::vector<double> &v) {
    // Create set
    std::set<int> s(v.begin(), v.end());
    for (const auto l : s) {
        mapping.insert({l, mapping.size()});
    }
    return mapping;
}

void SEQL::Preprocessing::LabelNormalizer::transform(
    std::vector<double> &v) const {
    std::transform(std::begin(v),
                   std::end(v),
                   std::begin(v),
                   [mapping = mapping](auto x) { return mapping.at(x); });
}

void SEQL::Preprocessing::LabelNormalizer::fit_transform(
    std::vector<double> &v) {
    fit(v);
    transform(v);
}

std::ostream &SEQL::Preprocessing::
operator<<(std::ostream &os, const SEQL::Preprocessing::LabelNormalizer &ln) {
    os << "Label mapping:\n";
    os << "Orginal label -> new label:\n";
    for (const auto &m : ln.mapping) {
        os << m.first << " -> " << m.second << '\n';
    }
    os << '\n';
    return os;
}

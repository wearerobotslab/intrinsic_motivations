// Minimal stand-in for AREFramework's misc utilities used by the controller.
#ifndef ARE_MISC_UTILITIES_STUB_H
#define ARE_MISC_UTILITIES_STUB_H
#include <vector>
#include <eigen3/Eigen/Core>

namespace are { namespace misc {
inline void stdvect_to_eigenmat(const std::vector<double>& v, Eigen::MatrixXd& m) {
    m.resize(static_cast<Eigen::Index>(v.size()), 1);
    for (size_t i = 0; i < v.size(); i++) m(static_cast<Eigen::Index>(i), 0) = v[i];
}
inline void eigenvect_to_stdvect(const Eigen::MatrixXd& m, std::vector<double>& v) {
    v.resize(static_cast<size_t>(m.size()));
    for (Eigen::Index i = 0; i < m.size(); i++) v[static_cast<size_t>(i)] = m(i);
}
}} // namespace are::misc
#endif

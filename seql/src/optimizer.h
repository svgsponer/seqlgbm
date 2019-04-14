namespace SEQL{
  
  struct regularization_param{
    const double sum_abs_betas = 0;
    const double sum_squared_betas = 0;
    const double alpha = 0;
    const double C = 0;
  };

  namespace optimizer{

    long double add_regularization(const double loss,
                                   const regurlatization_param rp,
                                   const double old_bc, const double new_bc);
  
    long double line_search(const rule_t &rule,
                            const std::vector<double> &y_true,
                            const std::vector<double> &y_pred,
                            const std::vector<double>& f_vec,
                            const bool is_intercept,
                            const regularization_param rp);
}

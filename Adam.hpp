#pragma once

#include "functions.hpp"
#include "named_types.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <valarray>

namespace Optimize
{
  ///////////////////////////
  // Dedicated Named_Types //
  ///////////////////////////
  //
  // alpha <- std::func(size_t iteration)
  //
  // By example logistic regression αt <- α0/sqrt(t)
  //
  using Adam_Alpha_Schedule =
      OptionalArgument::Named_Std_Function<
          struct Adam_Alpha_Schedule_Tag,
          double,
          const size_t>;
  static constexpr auto _Adam_alpha_schedule_ =
      typename Adam_Alpha_Schedule::
          argument_syntactic_sugar();
  static constexpr auto Adam_alpha_constant_schedule =
      [](const double alpha) {
        return [alpha](const size_t) -> double {
          return alpha;
        };
      };

  using Adam_Beta_1 = OptionalArgument::Named_Assert_Type<
      struct Adam_Beta_1_Tag,
      Assert_In_01_Strict<double>,
      double>;
  static constexpr auto _Adam_beta_1_ =
      typename Adam_Beta_1::argument_syntactic_sugar();

  using Adam_Beta_2 = OptionalArgument::Named_Assert_Type<
      struct Adam_Beta_2_Tag,
      Assert_In_01_Strict<double>,
      double>;
  static constexpr auto _Adam_beta_2_ =
      typename Adam_Beta_2::argument_syntactic_sugar();

  using Adam_Internal_Epsilon =
      OptionalArgument::Named_Assert_Type<
          struct Adam_Internal_Epsilon_Tag,
          Assert_In_01_Strict<double>,
          double>;
  static constexpr auto _Adam_internal_epsilon_ =
      typename Adam_Internal_Epsilon::
          argument_syntactic_sugar();

  ///////////////////
  // Configuration //
  ///////////////////
  //
  template <typename SCALAR_TYPE>
  struct Adam_Configuration
  {
    Maximum_Iterations maximimum_iterations = 100;

    Adam_Alpha_Schedule alpha_schedule =
        Adam_alpha_constant_schedule(0.01);

    Adam_Beta_1 beta_1 = 0.9;
    Adam_Beta_2 beta_2 = 0.999;

    Absolute_Epsilon absolute_epsilon = 1e-6;

    Verbose verbose = false;

    Adam_Internal_Epsilon internal_epsilon = std::sqrt(
        std::numeric_limits<SCALAR_TYPE>::epsilon());
  };

  template <typename SCALAR_TYPE,
            typename VECTOR_TYPE,
            typename... USER_ARGS>
  Adam_Configuration<SCALAR_TYPE>
  configure_Adam(const Differentiable_Function<
                     VECTOR_TYPE,
                     SCALAR_TYPE,
                     VECTOR_TYPE>& /*objective_function*/,
                 const VECTOR_TYPE& /*x_init*/,
                 USER_ARGS... user_args)
  {
    Adam_Configuration<SCALAR_TYPE> configuration;

    auto options = take_optional_argument_ref(
        configuration.alpha_schedule,
        configuration.beta_1,
        configuration.beta_2,
        configuration.maximimum_iterations,
        configuration.absolute_epsilon,
        configuration.verbose);
    optional_argument(options, user_args...);

    return configuration;
  }

  // Here specialization
  template <typename SCALAR_TYPE>
  auto
  squared_norm_2(const std::valarray<SCALAR_TYPE>& v)
  {
    using std::abs;
    using std::pow;

    using real_type =
        decltype(std::abs(std::declval<SCALAR_TYPE>()));

    real_type sum = 0;

    const size_t n = v.size();
    for (size_t i = 0; i < n; i++)
    {
      sum += pow(abs(v[i]), 2);
    }
    return sum;
  }
  template <typename SCALAR_TYPE>
  auto
  norm_2(const std::valarray<SCALAR_TYPE>& v)
  {
    return std::sqrt(squared_norm_2(v));
  }

  // Returns true if has converged
  template <typename SCALAR_TYPE,
            typename CONFIGURATION_TYPE>
  bool
  Adam_optimize(
      const CONFIGURATION_TYPE& configuration,
      const Differentiable_Function<
          std::valarray<SCALAR_TYPE>,
          SCALAR_TYPE,
          std::valarray<SCALAR_TYPE>>& objective_function,
      std::valarray<SCALAR_TYPE>& x_init)
  {
    using vector_type = std::valarray<SCALAR_TYPE>;

    const size_t domain_size = x_init.size();

    vector_type m_k(domain_size);
    m_k = 0;
    vector_type hat_m_k(domain_size);
    vector_type v_k(domain_size);
    v_k = 0;
    vector_type hat_v_k(domain_size);

    vector_type grad(domain_size);

    // shortcut
    const auto& alpha_schedule =
        configuration.alpha_schedule;

    const auto& beta_1 = configuration.beta_1.value();
    const auto& beta_2 = configuration.beta_2.value();

    const auto& internal_epsilon =
        configuration.internal_epsilon.value();

    bool has_converged = false;
    for (size_t k = 1;
         k < configuration.maximimum_iterations.value();
         ++k)
    {
      objective_function.df(x_init, grad);

      const auto grad_norm = norm_2(grad);
      has_converged =
          grad_norm <
          configuration.absolute_epsilon.value();

      if (has_converged or
          (configuration.verbose.value() and (k % 10 == 1)))
      {
        std::cerr << std::setw(5) << k << " "
                  << std::setw(15) << std::setprecision(10)
                  << eval_f(objective_function, x_init)
                  << " " << std::setw(15)
                  << std::setprecision(10) << grad_norm
                  << std::endl;
      }
      if (has_converged) break;

      m_k     = beta_1 * m_k + (1 - beta_1) * grad;
      v_k     = beta_2 * v_k + (1 - beta_2) * grad * grad;
      hat_m_k = (1 / (1 - std::pow(beta_1, k))) * m_k;
      hat_v_k = (1 / (1 - std::pow(beta_2, k))) * v_k;
      x_init =
          x_init - alpha_schedule(k) * hat_m_k /
                       (sqrt(hat_v_k) + internal_epsilon);
    }

    return has_converged;
  }

  template <typename SCALAR_TYPE,
            typename VECTOR_TYPE,
            typename... USER_ARGS>
  bool
  Adam_optimize(const Differentiable_Function<VECTOR_TYPE,
                                              SCALAR_TYPE,
                                              VECTOR_TYPE>&
                    objective_function,
                VECTOR_TYPE& x_init,
                USER_ARGS... user_args)
  {
    auto configuration = configure_Adam(
        objective_function,
        x_init,
        std::forward<USER_ARGS>(user_args)...);

    return Adam_optimize(
        configuration, objective_function, x_init);
  }

}

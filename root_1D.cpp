#include "functions.hpp"

#include <iomanip>
#include <iostream>

using namespace Optimize;

template <typename T>
void
square_root(const T& x, T* f, T* df, T c)
{
  if (f)
  {
    (*f) = x * x - c;
  }
  if (df)
  {
    (*df) = 2 * x;
  }
}

template <typename T>
void
show_iteration(size_t iter, T x, T f)
{
  constexpr auto max_digits =
      std::numeric_limits<T>::max_digits10;

  std::cerr << std::setw(4) << iter
            << " x = " << std::setw(max_digits + 5)
            << std::setprecision(max_digits) << x
            << " f = " << std::setw(max_digits + 5)
            << std::setprecision(max_digits) << f
            << std::endl;
}

template <typename T>
bool
Newton(const Differentiable_Function<T, T, T>& f_obj,
       T& x,
       double epsilon  = 1e-10,
       size_t max_iter = 20)
{
  T f, df;

  bool has_converged = false;

  for (size_t iter = 1; iter <= max_iter; ++iter)
  {
    f_obj.f_df(x, f, df);

    auto delta_x = -f / df;

    has_converged = std::abs(delta_x) < epsilon;

    x = x + delta_x;

    show_iteration(iter, x, f);

    if (has_converged) break;
  }
  return has_converged;
}

template <typename T>
bool
Steffensen(const Function<T, T>& f_obj,
           T& x,
           double epsilon  = 1e-10,
           size_t max_iter = 20)
{
  T f, g;

  bool has_converged = false;

  for (size_t iter = 1; iter <= max_iter; ++iter)
  {
    f_obj.f(x, f);
    f_obj.f(x + f, g);

    auto delta_x = -f * f / (g - f);

    has_converged = std::abs(delta_x) < epsilon;

    x = x + delta_x;

    show_iteration(iter, x, f);

    if (has_converged) break;
  }
  return has_converged;
}

int
main()
{
  Differentiable_Function f(square_root<double>, 2);
  bool has_converged;
  const double x_init = 2;
  double x;

  ////////////////

  std::cerr << std::endl << "Newton" << std::endl;
  f.initialize_counter();
  x = x_init;

  has_converged = Newton(f, x);

  std::cerr << "has converged: " << std::boolalpha
            << has_converged << std::endl;
  std::cerr << "f counter:  " << f.f_counter() << std::endl;
  std::cerr << "df counter: " << f.df_counter()
            << std::endl;

  ////////////////

  std::cerr << std::endl << "Steffensen" << std::endl;
  f.initialize_counter();
  x = x_init;

  has_converged = Steffensen(f.as_function(), x);

  std::cerr << "has converged: " << std::boolalpha
            << has_converged << std::endl;
  std::cerr << "f counter:  " << f.f_counter() << std::endl;
  std::cerr << "df counter: " << f.df_counter()
            << std::endl;
}

#include "Adam.hpp"

using namespace Optimize;

void
Rosenbrock(const std::valarray<double>& x, double* f, std::valarray<double>* df, double c)
{
  assert(x.size() == 2);
  assert(df == nullptr or df->size() == 2);

  if (f)
  {
    *f = (1 - x[0]) * (1 - x[0]) + c * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]);
  }

  if (df)
  {
    (*df)[0] = 2 * (-1 + x[0] + 2 * c * x[0] * x[0] * x[0] - 2 * c * x[0] * x[1]);
    (*df)[1] = 2 * c * (x[1] - x[0] * x[0]);
  }
}

int
main()
{
  Differentiable_Function f(Rosenbrock, 10);

  std::valarray<double> x(2, 2);
  double y;
  std::valarray<double> grad(2);

  f.initialize_counter();

  bool has_converged = Adam_optimize(
      f,
      x,
      y,
      grad,

      _Adam_beta_1_         = 0.6,
      _Adam_beta_2_         = 0.6,
      _Adam_alpha_schedule_ = [](const size_t t) -> double { return 1 / sqrt(t); },
      _absolute_epsilon_    = 0.01,
      _verbose_             = true);

  std::cerr << "has converged: " << std::boolalpha << has_converged << std::endl;
  std::cerr << "f counter:  " << f.f_counter() << std::endl;
  std::cerr << "df counter: " << f.df_counter() << std::endl;
}

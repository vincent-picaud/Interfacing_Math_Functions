#pragma once

#include "identity.hpp"

#include <cassert>
#include <memory>

namespace Optimize
{
  //////////////
  // Function //
  //////////////
  //
  template <typename DOMAIN_TYPE, typename CODOMAIN_TYPE>
  class Function
  {
   public:
    using domain_type   = DOMAIN_TYPE;
    using codomain_type = CODOMAIN_TYPE;

    struct Interface
    {
      virtual void f(const domain_type& x, codomain_type& f) const = 0;

      virtual ~Interface() = default;
    };

   protected:
    using pimpl_type = std::shared_ptr<const Interface>;
    pimpl_type _pimpl;
    std::shared_ptr<size_t> _f_counter;

   public:
    Function() : _pimpl{} {}
    Function(pimpl_type&& pimpl, std::shared_ptr<size_t> f_counter = {})
        : _pimpl{std::move(pimpl)}, _f_counter{f_counter}
    {
    }

    void
    f(const domain_type& x, codomain_type& y) const
    {
      if (_f_counter) ++(*_f_counter);

      (*_pimpl).f(x, y);
    }

    void
    initialize_counter()
    {
      _f_counter = std::make_shared<size_t>(0);
    }

    size_t
    f_counter() const
    {
      assert(_f_counter);
      return *_f_counter;
    }
  };

  // Helper
  //
  template <typename DOMAIN_TYPE, typename CODOMAIN_TYPE>
  CODOMAIN_TYPE
  eval_f(const Function<DOMAIN_TYPE, CODOMAIN_TYPE>& func, const DOMAIN_TYPE& x)
  {
    CODOMAIN_TYPE y;
    func.f(x, y);
    return y;
  }

  //
  // Build from:
  // #+begin_src cpp
  // void foo(const domain_type& x,codomain_type& y, extra_args... ) { ... }
  // #+end_src
  //
  template <typename DOMAIN_TYPE, typename CODOMAIN_TYPE, typename... EXTRA_ARGS>
  Function<DOMAIN_TYPE, CODOMAIN_TYPE>
  create_function(void(f)(const DOMAIN_TYPE&, CODOMAIN_TYPE&, EXTRA_ARGS...),
                  const Identity_t<EXTRA_ARGS>&... args)
  {
    using objective_function_type = decltype(create_function(f, args...));

    auto lambda = [=](const DOMAIN_TYPE& x, CODOMAIN_TYPE& y) { return f(x, y, args...); };

    struct Impl final : public objective_function_type::Interface
    {
      using F = decltype(lambda);
      F _f;

      Impl(F f) : _f(f) {}

      void
      f(const DOMAIN_TYPE& x, CODOMAIN_TYPE& y) const
      {
        _f(x, y);
      }
    };
    return objective_function_type{std::make_shared<Impl>(lambda)};
  }

  /////////////////
  // C1 Function //
  /////////////////
  //
  template <typename DOMAIN_TYPE, typename CODOMAIN_TYPE, typename DIFFERENTIAL_TYPE>
  class Differentiable_Function
  {
   public:
    using objective_function_type = Function<DOMAIN_TYPE, CODOMAIN_TYPE>;
    using domain_type             = typename objective_function_type::domain_type;
    using codomain_type           = typename objective_function_type::codomain_type;
    using differential_type       = DIFFERENTIAL_TYPE;

    struct Interface_C1 : public objective_function_type::Interface
    {
      virtual void f_df(const domain_type& x,
                        codomain_type& y,
                        differential_type& differential) const           = 0;
      virtual void df(const domain_type& x, differential_type& df) const = 0;
    };

   protected:
    using pimpl_type = std::shared_ptr<const Interface_C1>;

   public:
    pimpl_type _pimpl;
    std::shared_ptr<size_t> _f_counter;
    std::shared_ptr<size_t> _df_counter;

   public:
    Differentiable_Function(pimpl_type&& pimpl,
                            std::shared_ptr<size_t> f_counter  = {},
                            std::shared_ptr<size_t> df_counter = {})
        : _pimpl(std::move(pimpl)), _f_counter{f_counter}, _df_counter{df_counter}
    {
    }

    objective_function_type
    as_function() const
    {
      return objective_function_type(_pimpl, _f_counter);
    }

    void
    f(const domain_type& x, codomain_type& y) const
    {
      if (_f_counter) ++(*_f_counter);

      (*_pimpl).f(x, y);
    }

    void
    f_df(const domain_type& x, codomain_type& y, differential_type& df) const
    {
      if (_f_counter) ++(*_f_counter);
      if (_df_counter) ++(*_df_counter);

      (*_pimpl).f_df(x, y, df);
    }

    void
    df(const domain_type& x, differential_type& df) const
    {
      if (_df_counter) ++(*_df_counter);

      (*_pimpl).df(x, df);
    }

    void
    initialize_counter()
    {
      _f_counter  = std::make_shared<size_t>(0);
      _df_counter = std::make_shared<size_t>(0);
    }

    size_t
    f_counter() const
    {
      assert(_f_counter);
      return *_f_counter;
    }

    size_t
    df_counter() const
    {
      assert(_df_counter);
      return *_df_counter;
    }
  };

  // Helper
  //
  template <typename DOMAIN_TYPE, typename CODOMAIN_TYPE, typename DIFFERENTIAL_TYPE>
  CODOMAIN_TYPE
  eval_f(const Differentiable_Function<DOMAIN_TYPE, CODOMAIN_TYPE, DIFFERENTIAL_TYPE>& func,
         const DOMAIN_TYPE& x)
  {
    CODOMAIN_TYPE y;
    func.f(x, y);
    return y;
  }

  // Build from:
  // #+begin_src
  // void foo(const std::vector<double>& x, double* f, std::vector<double>* df, extra_args...)
  // #+end_src
  // Use the convention = ptr = 0 -> do not compute
  //
  // ATTENTION: for scalar function, df means gradient
  //
  template <typename DOMAIN_TYPE,
            typename CODOMAIN_TYPE,
            typename DIFFERENTIAL_TYPE,
            typename... EXTRA_ARGS>
  Differentiable_Function<DOMAIN_TYPE, CODOMAIN_TYPE, DIFFERENTIAL_TYPE>
  create_differentiable_function(
      void(f)(const DOMAIN_TYPE&, CODOMAIN_TYPE*, DIFFERENTIAL_TYPE*, EXTRA_ARGS...),
      const Identity_t<EXTRA_ARGS>&... args)
  {
    using differentiable_function_type = decltype(create_differentiable_function(f, args...));

    auto lambda = [=](const DOMAIN_TYPE& x, CODOMAIN_TYPE* y, DIFFERENTIAL_TYPE* df) {
      return f(x, y, df, args...);
    };

    struct Impl final : public differentiable_function_type::Interface_C1
    {
      using F = decltype(lambda);
      F _f;

      Impl(const F& f) : _f(f) {}

      void
      f(const DOMAIN_TYPE& x, CODOMAIN_TYPE& y) const
      {
        _f(x, &y, nullptr);
      }

      void
      f_df(const DOMAIN_TYPE& x, CODOMAIN_TYPE& y, DIFFERENTIAL_TYPE& df) const
      {
        _f(x, &y, &df);
      }
      void
      df(const DOMAIN_TYPE& x, DIFFERENTIAL_TYPE& df) const
      {
        _f(x, nullptr, &df);
      }
    };

    return differentiable_function_type{std::make_shared<Impl>(lambda)};
  }

}  // namespace Optimize

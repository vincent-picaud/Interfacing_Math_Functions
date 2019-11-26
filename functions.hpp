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
      virtual void f(const domain_type& x,
                     codomain_type& f) const = 0;

      virtual ~Interface() = default;
    };

   protected:
    using pimpl_type = std::shared_ptr<const Interface>;
    pimpl_type _pimpl;
    std::shared_ptr<size_t> _f_counter;

   public:
    Function() : _pimpl{} {}
    Function(pimpl_type&& pimpl,
             std::shared_ptr<size_t> f_counter = {})
        : _pimpl{std::move(pimpl)}, _f_counter{f_counter}
    {
    }

    // lambda(domain)->codomain
    template <typename F,
              std::enable_if_t<std::is_invocable_r_v<
                  CODOMAIN_TYPE,
                  std::decay_t<F>,
                  DOMAIN_TYPE>>* = nullptr>
    Function(F&& f)
    {
      struct Impl final : public Interface
      {
        std::decay_t<F> _f;

        Impl(F&& f) : _f(std::forward<F>(f)) {}

        void
        f(const DOMAIN_TYPE& x, CODOMAIN_TYPE& y) const
        {
          y = _f(x);
        }
      };
      _pimpl = std::make_shared<Impl>(std::forward<F>(f));
    }
    // lambda(domain,codomain&)
    template <typename F,
              std::enable_if_t<std::is_invocable_r_v<
                  void,
                  std::decay_t<F>,
                  DOMAIN_TYPE,
                  CODOMAIN_TYPE&>>* = nullptr>
    Function(F&& f)
    {
      struct Impl final : public Interface
      {
        std::decay_t<F> _f;

        Impl(F&& f) : _f(std::forward<F>(f)) {}

        void
        f(const DOMAIN_TYPE& x, CODOMAIN_TYPE& y) const
        {
          _f(x, y);
        }
      };
      _pimpl = std::make_shared<Impl>(std::forward<F>(f));
    }
    // codomain_type foo(const domain_type& x, extra_args... )
    template <typename... EXTRA_ARGS>
    Function(CODOMAIN_TYPE(f)(const DOMAIN_TYPE&,
                              EXTRA_ARGS...),
             const Identity_t<EXTRA_ARGS>&... args)
        : Function{
              [=](const DOMAIN_TYPE& x, CODOMAIN_TYPE& y) {
                y = f(x, args...);
              }}
    {
    }

    // void foo(const domain_type& x,codomain_type& y, extra_args... )
    template <typename... EXTRA_ARGS>
    Function(void(f)(const DOMAIN_TYPE&,
                     CODOMAIN_TYPE&,
                     EXTRA_ARGS...),
             const Identity_t<EXTRA_ARGS>&... args)
        : Function{
              [=](const DOMAIN_TYPE& x, CODOMAIN_TYPE& y) {
                f(x, y, args...);
              }}
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
  eval_f(const Function<DOMAIN_TYPE, CODOMAIN_TYPE>& func,
         const Identity_t<DOMAIN_TYPE>& x)
  {
    CODOMAIN_TYPE y;
    func.f(x, y);
    return y;
  }

  /////////////////////////////
  // Differentiable Function //
  /////////////////////////////
  //
  template <typename DOMAIN_TYPE,
            typename CODOMAIN_TYPE,
            typename DIFFERENTIAL_TYPE>
  class Differentiable_Function
  {
   public:
    using function_type =
        Function<DOMAIN_TYPE, CODOMAIN_TYPE>;
    using domain_type = typename function_type::domain_type;
    using codomain_type =
        typename function_type::codomain_type;
    using differential_type = DIFFERENTIAL_TYPE;

    struct Diff_Interface : public function_type::Interface
    {
      virtual void f_df(
          const domain_type& x,
          codomain_type& y,
          differential_type& differential) const   = 0;
      virtual void df(const domain_type& x,
                      differential_type& df) const = 0;
    };

   protected:
    using pimpl_type =
        std::shared_ptr<const Diff_Interface>;

   public:
    pimpl_type _pimpl;
    std::shared_ptr<size_t> _f_counter;
    std::shared_ptr<size_t> _df_counter;

   public:
    operator function_type() const
    {
      return {_pimpl, _f_counter};
    }

    Differentiable_Function(
        pimpl_type&& pimpl,
        std::shared_ptr<size_t> f_counter  = {},
        std::shared_ptr<size_t> df_counter = {})
        : _pimpl(std::move(pimpl)),
          _f_counter{f_counter},
          _df_counter{df_counter}
    {
    }
    // lambda(domain,codomain*,differential*)
    template <typename F,
              std::enable_if_t<std::is_invocable_r_v<
                  void,
                  std::decay_t<F>,
                  DOMAIN_TYPE,
                  CODOMAIN_TYPE*,
                  DIFFERENTIAL_TYPE*>>* = nullptr>
    Differentiable_Function(F&& f)
    {
      struct Impl final : public Diff_Interface
      {
        std::decay_t<F> _f;

        Impl(F&& f) : _f(std::forward<F>(f)) {}

        void
        f(const DOMAIN_TYPE& x, CODOMAIN_TYPE& y) const
        {
          _f(x, &y, nullptr);
        }
        void
        f_df(const DOMAIN_TYPE& x,
             CODOMAIN_TYPE& y,
             DIFFERENTIAL_TYPE& df) const
        {
          _f(x, &y, &df);
        }
        void
        df(const DOMAIN_TYPE& x,
           DIFFERENTIAL_TYPE& df) const
        {
          _f(x, nullptr, &df);
        }
      };
      _pimpl = std::make_shared<Impl>(std::forward<F>(f));
    }
    // void foo(const std::vector<double>& x, double* f, std::vector<double>* df, extra_args...)
    template <typename... EXTRA_ARGS>
    Differentiable_Function(
        void(f)(const DOMAIN_TYPE&,
                CODOMAIN_TYPE*,
                DIFFERENTIAL_TYPE*,
                EXTRA_ARGS...),
        const Identity_t<EXTRA_ARGS>&... args)
        : Differentiable_Function{
              [=](const DOMAIN_TYPE& x,
                  CODOMAIN_TYPE* y,
                  DIFFERENTIAL_TYPE* df) {
                f(x, y, df, args...);
              }}
    {
    }

    function_type
    as_function() const
    {
      return static_cast<function_type>(*this);
    }

    void
    f(const domain_type& x, codomain_type& y) const
    {
      if (_f_counter) ++(*_f_counter);

      (*_pimpl).f(x, y);
    }

    void
    f_df(const domain_type& x,
         codomain_type& y,
         differential_type& df) const
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
  template <typename DOMAIN_TYPE,
            typename CODOMAIN_TYPE,
            typename DIFFERENTIAL_TYPE>
  CODOMAIN_TYPE
  eval_f(const Differentiable_Function<DOMAIN_TYPE,
                                       CODOMAIN_TYPE,
                                       DIFFERENTIAL_TYPE>&
             func,
         const Identity_t<DOMAIN_TYPE>& x)
  {
    CODOMAIN_TYPE y;
    func.f(x, y);
    return y;
  }

}  // namespace Optimize

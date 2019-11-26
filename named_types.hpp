#pragma once

#include "optional_argument.hpp"

////////////////
// Named_Type //
////////////////
//

//////////////// Maximimum iterations ////////////////
//
using Maximum_Iterations = Named_Type<struct Maximum_Iterations_Tag, size_t>;
static constexpr auto _maximimum_iterations_ =
    typename Maximum_Iterations::argument_syntactic_sugar();

//////////////// Verbose ////////////////
//
using Verbose                   = Named_Type<struct Verbose_Tag, bool>;
static constexpr auto _verbose_ = typename Verbose::argument_syntactic_sugar();

///////////////////////
// Named_Assert_Type //
///////////////////////

//////////////// Asserts ////////////////
//
template <typename T>
struct Assert_Positive
{
  void
  operator()(const T& t) const
  {
    assert(t > 0);
  }
};

template <typename T>
struct Assert_In_01_Strict
{
  void
  operator()(const T& t) const
  {
    assert(t > 0);
    assert(t < 1);
  }
};

//////////////// Relative_Epsilon (stopping criterion) ////////////////
//
using Relative_Epsilon =
    Named_Assert_Type<struct Relative_Epsilon_Tag, Assert_Positive<double>, double>;
static constexpr auto _relative_epsilon_ = typename Relative_Epsilon::argument_syntactic_sugar();

//////////////// Absolute_Epsilon (stopping criterion) ////////////////
//
using Absolute_Epsilon =
    Named_Assert_Type<struct Absolute_Epsilon_Tag, Assert_Positive<double>, double>;
static constexpr auto _absolute_epsilon_ = typename Absolute_Epsilon::argument_syntactic_sugar();


#pragma once

namespace Optimize
{

  template <typename T>
  struct Identity
  {
    using type = T;
  };
  template <typename T>
  using Identity_t = typename Identity<T>::type;

}  // namespace Optimize

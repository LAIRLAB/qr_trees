
#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator> // for ostream_iterator

template <typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
  if ( !v.empty() ) {
    out << '[';
    std::copy (v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
    out << "\b\b]" << std::endl;;
  }
  return out;
}


template <typename T>
void print_vec(const std::vector<T>& v)
{
    std::cout << v;
}

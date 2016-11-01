#pragma once


#include <iostream>
#include <stdexcept>
#include <sstream>

// Macros and other utilties

#define PRINT(x) std::cout << x << std::endl;
#define DEBUG(x) std::cout << "\033[36m" << x << "\033[0m" << std::endl;
#define ERROR(x) std::cout << "\033[31m" << x << "\033[0m" << std::endl;
#define SUCCESS(x) std::cout << "\033[32m" << x << "\033[0m" << std::endl;
#define WARN(x) std::cout << "\033[33m" << x << "\033[0m" << std::endl;

#define FILE_LINE __FILE__ << ":" << __LINE__ << " in " << __func__  << "(): "


#define IS_EQUAL(X,Y) {\
    if (X != Y) {std::ostringstream ss; ss << FILE_LINE << "\n\t\tExpected value (" << X << ") EQUAL to (" << Y << ")."; throw std::logic_error(ss.str()); } \
}

#define IS_ALMOST_EQUAL(X,Y,EPS) {\
    if (std::abs(X - Y) <= EPS) {std::ostringstream ss; ss << FILE_LINE << "\n\t\tExpected value (" << X << ") ALMOST EQUAL to (" << Y << ") with Eps " << EPS; throw std::logic_error(ss.str()); } \
}

#define IS_NOT_EQUAL(X,Y) {\
    if (X == Y) {std::ostringstream ss; ss << FILE_LINE << "\n\t\tExpected value (" << X << ") NOT EQUAL to (" << Y << ")."; throw std::logic_error(ss.str()); } \
}

#define IS_TRUE(X) {\
    if (!X) {std::ostringstream ss; ss << FILE_LINE << "\n\t\tExpected TRUE (" << X << ")"; throw std::logic_error(ss.str()); } \
}

#define IS_FALSE(X) {\
    if (X) {std::ostringstream ss; ss << FILE_LINE << "\n\t\tExpected FALSE (" << X << ")"; throw std::logic_error(ss.str()); } \
}

#define IS_GREATER(X,Y) {\
    if (X <= Y) {std::ostringstream ss; ss << FILE_LINE << "\n\t\tExpected value (" << X << ") GREATER than (" << Y << ")."; throw std::logic_error(ss.str()); } \
}

#define IS_GREATER_EQUAL(X,Y) {\
    if (X < Y) {std::ostringstream ss; ss << FILE_LINE << "\n\t\tExpected value (" << X << ") GREATER EQUAL than (" << Y << ")."; throw std::logic_error(ss.str()); } \
}

#define IS_BETWEEN(X, L, U) {\
    if (X <= L || X >= U) {std::ostringstream ss; ss << FILE_LINE << "\n\t\tExpected value (" << X << ") in range (" << L << ", " << U << ")."; throw std::logic_error(ss.str()); } \
}

#define IS_BETWEEN_INCLUSIVE(X, L, U) {\
    if (X < L || X > U) {std::ostringstream ss; ss << FILE_LINE << "\n\t\tExpected value (" << X << ") in inclusive range [" << L << ", " << U << "]."; throw std::logic_error(ss.str()); } \
}

#define IS_BETWEEN_UPPER_INCLUSIVE(X, L, U) {\
    if (X <= L || X > U) {std::ostringstream ss; ss << FILE_LINE << "\n\t\tExpected value (" << X << ") in upper-inclusive range (" << L << ", " << U << "]."; throw std::logic_error(ss.str()); } \
}

#define IS_BETWEEN_LOWER_INCLUSIVE(X, L, U) {\
    if (X < L || X >= U) {std::ostringstream ss; ss << FILE_LINE << "\n\t\tExpected value (" << X << ") in upper-inclusive range [" << L << ", " << U << ")."; throw std::logic_error(ss.str()); } \
}

#define IS_LESS(X,Y) {\
    if (X >= Y) {std::ostringstream ss; ss << FILE_LINE << "\n\t\tExpected value (" << X << ") LESS than (" << Y << ")."; throw std::logic_error(ss.str()); } \
}

#define IS_LESS_EQUAL(X,Y) {\
    if (X > Y) {std::ostringstream ss; ss << FILE_LINE << "\n\t\tExpected value (" << X << ") LESS EQUAL than (" << Y << ")."; throw std::logic_error(ss.str()); } \
}

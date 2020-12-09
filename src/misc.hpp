#pragma once

/*
 * This file is part of RFPKOG
 *
 * Copyright (C) 2020 Gard Spreemann <gspr@nonempty.org>
 *
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * RFPKOG is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 3.
 *
 * RFPKOG is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace rfpkog
{
  std::vector<std::string> split(const std::string & s, char delim);

  inline constexpr double PI = 4*std::atan(1);
  
  template <typename T> inline void reverse_endianness(T & x)
  {
    uint8_t * p = reinterpret_cast<uint8_t *>(&x);
    std::reverse(p, p + sizeof(T));
  }

  template <typename T> inline T read_le(std::istream & s)
  {
    T result;
    s.read(reinterpret_cast<char *>(&result), sizeof(T));
    #ifdef BIGENDIAN
    reverse_endianness(result);
    #endif
    return result;
  }

  template <typename T> inline void write_le(std::ostream & s, T value)
  {
    #ifdef BIGENDIAN
    reverse_endianness(value);
    #endif
    s.write(reinterpret_cast<char *>(&value), sizeof(T));
  }

  template <typename T> inline T read_be(std::istream & s)
  {
    T result;
    s.read(reinterpret_cast<char *>(&result), sizeof(T));
    #ifndef BIGENDIAN
    reverse_endianness(result);
    #endif
    return result;
  }

  template <typename T> inline void write_be(std::ostream & s, T value)
  {
    #ifndef BIGENDIAN
    reverse_endianness(value);
    #endif
    s.write(reinterpret_cast<char *>(&value), sizeof(T));
  }
}

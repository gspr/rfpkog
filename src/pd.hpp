#pragma once

/*
 * This file is part of RFPKOG
 *
 * Copyright (C) 2020-2021 Gard Spreemann <gspr@nonempty.org>
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

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "dtype.hpp"
#include "misc.hpp"

namespace rfpkog
{
  constexpr int64_t DIPHA_MAGIC = 8067171840;
  constexpr int64_t DIPHA_DISTANCE_MATRIX = 7;
  constexpr int64_t DIPHA_PERSISTENCE_DIAGRAM = 2;
  constexpr int64_t DIPHA_WEIGHTED_BOUNDARY_MATRIX = 0;

  template <typename T>
  int read_dipha_degree(const std::string & fname, unsigned int degree, typename T::scalar_type finitization, std::vector<typename T::vector_type> & pd)
  {
    std::ifstream f;
    f.open(fname, std::ios::in | std::ios::binary);
    if (!f.is_open())
    {
      //std::cerr << "Failed to open " << fname << "." << std::endl;
      return 1;
    }

    if (read_le<int64_t>(f) != DIPHA_MAGIC)
    {
      //std::cerr << "File " << fname << " is not DIPHA file." << std::endl;
      f.close();
      return 1;
    }
    
    if (read_le<int64_t>(f) != DIPHA_PERSISTENCE_DIAGRAM)
    {
      //std::cerr << "File " << fname << " is not persistence diagram. Bailing." << std::endl;
      f.close();
      return 1;
    }
  
    int64_t n = read_le<int64_t>(f);

    for (int64_t i = 0; i < n; ++i)
    {
      int64_t d = read_le<int64_t>(f);
      double birth = read_le<double>(f);
      double death = read_le<double>(f);
      if (d == degree && birth < death)
      {
        typename T::vector_type tmp;
        tmp.s[0] = birth;
        tmp.s[1] = death;
        pd.push_back(tmp);
      }
      else if (-d - 1 == degree)
      {
        typename T::vector_type tmp;
        tmp.s[0] = birth;
        tmp.s[1] = finitization;
        pd.push_back(tmp);
      }
    }
  
    f.close();
    return 0;
  }
}


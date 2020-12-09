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

#include <sstream>

#include "misc.hpp"

namespace rfpkog
{
  std::vector<std::string> split(const std::string & s, char delim)
  {
    std::vector<std::string> ret;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim))
    {
      ret.push_back(item);
    }
    return ret;
  }
}

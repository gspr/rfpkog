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

#ifndef KERNEL_FILENAME
#define KERNEL_FILENAME "kernels.cl"
#endif

#ifndef RFPKOG_VERSION
#define RFPKOG_VERSION "undefined"
#endif


#include <array>
#include <limits>
#include <set>
#include <string>
#include <vector>

namespace rfpkog
{
  enum class Kernel_choice { pssk, pwgk };
  
  class Options
  {
  public:
    Options() : help(false), list_devices(false), use_double(false), symmetric(false), print_version(false),
                sigma(std::numeric_limits<double>::quiet_NaN()),
                p(std::numeric_limits<double>::quiet_NaN()), c(std::numeric_limits<double>::quiet_NaN()),
                finitization(std::numeric_limits<double>::quiet_NaN()),
                degree(std::numeric_limits<unsigned int>::max()), verbosity(0), local_work_shape({0,0}),
                platform_id(std::numeric_limits<std::size_t>::max()), output_fname(""), kernel_fname(KERNEL_FILENAME), kernel_choice(Kernel_choice::pssk)
    {
      copyright = std::string(R"(Copyright (C) 2020-2021 Gard Spreemann
License LGPLv3: GNU LGPL version 3 <https://gnu.org/licenses/lgpl.html>.
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.)");
    }

    int parse(const int argc, const char * const * argv);
    int validate();
    std::string get_help() const;

    bool help;
    bool list_devices;
    bool use_double;
    bool symmetric;
    bool print_version;
    double sigma;
    double p;
    double c;
    double finitization;
    unsigned int degree;
    unsigned int verbosity;
    std::array<std::size_t, 2> local_work_shape;
    std::size_t platform_id;
    std::set<std::size_t> device_ids;
    std::array<std::string, 2> fname_lists;
    std::array<std::vector<std::string>, 2> fnames;
    std::string output_fname;
    std::string kernel_fname;
    std::string invocation;
    std::string copyright;
    Kernel_choice kernel_choice;
  };
}

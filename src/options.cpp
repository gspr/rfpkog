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

#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include "misc.hpp"
#include "options.hpp"

namespace rfpkog
{
  int Options::parse(const int argc, const char * const * argv)
  {
    for (int i = 1; i < argc; ++i)
    {
      const std::string arg(argv[i]);
      
      if (arg == "--help" || arg == "-h")
      {
        help = true;
      }
      else if (arg == "--list" || arg == "-l")
      {
        list_devices = true;
      }
      else if (arg == "--double" || arg == "--64")
      {
        use_double = true;
      }
      else if (arg == "--sigma" || arg == "-s")
      {
        if (i + 1 < argc)
          sigma = std::stod(argv[++i]);
        else
        {
          std::cerr << "Missing argument for --sigma." << std::endl;
          return 1;
        }
      }
      else if (arg == "--finitization" || arg == "-f")
      {
        if (i + 1 < argc)
          finitization = std::stod(argv[++i]);
        else
        {
          std::cerr << "Missing argument for --finitization." << std::endl;
          return 1;
        }
      }
      else if (arg == "--degree" || arg == "-d")
      {
        if (i + 1 < argc)
          degree = std::atoi(argv[++i]);
        else
        {
          std::cerr << "Missing argument for --degree." << std::endl;
          return 1;
        }
      }
      else if (arg == "--verbose" || arg == "-v")
      {
        ++verbosity;
      }
      else if (arg == "--workshape")
      {
        if (i + 1 < argc && (std::string(argv[i+1]) == "max" || std::string(argv[i+1]) == "auto")) // --max and --auto are synonyms for now.
        {
          local_work_shape = {0, 0};
          ++i;
        }
        else if (i + 2 < argc)
        {
          local_work_shape = {std::stoul(argv[i+1]), std::stoul(argv[i+2])};
          i += 2;
        }
        else
        {
          std::cerr << "Missing argument(s) for --workshape." << std::endl;
          return 1;
        }  
      }
      else if (arg == "--platform" || arg == "-p")
      {
        if (i + 1 < argc)
          platform_id = std::stoul(argv[++i]);
        else
        {
          std::cerr << "Missing argument for --platform." << std::endl;
          return 1;
        }
      }
      else if (arg == "--devices")
      {
        if (i + 1 < argc)
        {
          if (std::string(argv[i+1]) == "all")
          {
            device_ids.clear();
          }
          else
          {
            const std::vector<std::string> splitargs_0 = split(argv[i+1], ',');
            for (const std::string splitarg : splitargs_0)
            {
              const std::vector<std::string> splitargs_1 = split(splitarg, '-');
              if (splitargs_1.size() == 1)
              {
                device_ids.insert(std::stoul(splitargs_1[0]));
              }
              else if (splitargs_1.size() == 2)
              {
                for (std::size_t a = std::stoul(splitargs_1[0]); a <= std::stoul(splitargs_1[1]); ++a)
                {
                  device_ids.insert(a);
                }
              }
              else
              {
                std::cerr << "Bad arguments for --devices." << std::endl;
                return 1;
              }
            }
          }
          ++i;
        }
        else
        {
          std::cerr << "Missing argument for --devices." << std::endl;
          return 1;
        }
      }
      else if (arg == "--output" || arg == "-o")
      {
        if (i + 1 < argc)
        {
          output_fname = std::string(argv[++i]);
        }
        else
        {
          std::cerr << "Mising argument for --output." << std::endl;
          return 1;
        }
      }
      else
      {
        if (fname_lists[0].empty())
          fname_lists[0] = arg;
        else if (fname_lists[1].empty())
          fname_lists[1] = arg;
        else
        {
          std::cerr << "Got unrecognized argument " << arg << "." << std::endl;
          return 1;
        }
      }
      
    }

    const char * tmp = std::getenv("RFPKOG_KERNEL_FILE_NAME");
    if (tmp)
    {
      kernel_fname = std::string(tmp);
    }
    
    return 0;
  }

  int Options::validate()
  {
    if (!list_devices)
    {
      if (sigma <= 0 || std::isinf(sigma) || std::isnan(sigma))
      {
        std::cerr << "σ must be positive and finite." << std::endl;
        return 1;
      }
      
      if (std::isinf(finitization) || std::isnan(finitization))
      {
        std::cerr << "The finitization must be finite." << std::endl;
        return 1;
      }

      if (degree == std::numeric_limits<unsigned int>::max())
      {
        std::cerr << "You must specify a degree for DIPHA input files." << std::endl;
        return 1;
      }

      if (platform_id == std::numeric_limits<std::size_t>::max())
      {
        std::cerr << "You must specify an OpenCL platform. Use --list to see those available." << std::endl;
        return 1;
      }
      
      if (fname_lists[0].empty() || fname_lists[1].empty())
      {
        std::cerr << "You must specify two files (or one repeated twice) containing the persistence diagram file nmaes to consider." << std::endl;
        return 1;
      }
    
      symmetric = fname_lists[0] == fname_lists[1];

      for (std::size_t a : {0, 1})
      {
        std::ifstream stream(fname_lists[a]);
        if (!stream.is_open())
        {
          std::cerr << "Failed to read input file " << fname_lists[a] << "." << std::endl;
          return 1;
        }
        std::string line;
        while (std::getline(stream, line))
        {
          fnames[a].push_back(line);
        }
      }

      if (symmetric)
        fnames[1] = fnames[0];
      
      if (fnames[0].empty() || fnames[1].empty())
      {
        std::cerr << "The file lists cannot be empty." << std::endl;
        return 1;
      }

      if (output_fname == "-")
        output_fname.clear();

      if (kernel_fname == "")
      {
        std::cerr << "The kernel file name, defined at compile-time or overriden through the RFPKOG_KERNEL_FILENAME environment variable, cannot be empty." << std::endl;
        return 1;
      }
    }

    return 0;
  }
}

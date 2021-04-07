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

#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "misc.hpp"
#include "options.hpp"

namespace rfpkog
{
  int Options::parse(const int argc, const char * const * argv)
  {
    invocation = std::string(argv[0]);
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
      else if (arg == "--power")
      {
        if (i + 1 < argc)
          p = std::stod(argv[++i]);
        else
        {
          std::cerr << "Missing argument for --power." << std::endl;
          return 1;
        }
      }
      else if (arg == "--weight")
      {
        if (i + 1 < argc)
          c = std::stod(argv[++i]);
        else
        {
          std::cerr << "Missing argument for --weight." << std::endl;
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
      else if (arg == "--kernel" || arg == "-k")
      {
        if (i + 1 < argc)
        {
          std::string kernel_string(argv[++i]);
          if (kernel_string == "pssk" || kernel_string == "PSSK" || kernel_string == "heat")
            kernel_choice = Kernel_choice::pssk;
          else if (kernel_string == "pwgk" || kernel_string == "PWGK")
            kernel_choice = Kernel_choice::pwgk;
          else
          {
            std::cerr << "Invalid argument for --kernel." << std::endl;
            return 1;
          }
        }
        else
        {
          std::cerr << "Missing argument for --kernel." << std::endl;
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
      else if (arg == "--version")
      {
        print_version = true;
      }
      else if (arg == "--workshape")
      {
        if (i + 1 < argc) 
        {
          if (std::string(argv[i+1]) == "max" || std::string(argv[i+1]) == "auto") // --max and --auto are synonyms for now.
          {
            local_work_shape = {0, 0};
          }
          else
          {
            const std::vector<std::string> splitargs = split(argv[i+1], ',');
            if (splitargs.size() == 2)
            {
              local_work_shape = {std::stoul(splitargs[0]), std::stoul(splitargs[1])};
            }
            else
            {
              std::cerr << "Malformed argument for --workshape." << std::endl;
              return 1;
            }
          }
          ++i;
        }
        else
        {
          std::cerr << "Missing argument for --workshape." << std::endl;
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
    if (list_devices || help || print_version)
    {
      return 0;
    }
    
    if (sigma <= 0 || std::isinf(sigma) || std::isnan(sigma))
    {
      std::cerr << "sigma must be positive and finite." << std::endl;
      return 1;
    }

    if (kernel_choice == Kernel_choice::pssk)
    {
      if (!std::isnan(p) || !std::isnan(c))
      {
        std::cerr << "Power and weight parameters make no sense for the PSSK." << std::endl;
        return 1;
      }
    }
    else if (kernel_choice == Kernel_choice::pwgk)
    {
      if (p <= 0 || std::isinf(p) || std::isnan(p))
      {
        std::cerr << "Power must be positive and finite." << std::endl;
        return 1;
      }
      if (c <= 0 || std::isinf(c) || std::isnan(c))
      {
        std::cerr << "Weight must be positive and finite." << std::endl;
        return 1;
      }
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
      std::cerr << "You must specify two files (or one repeated twice) containing the persistence diagram file names to consider." << std::endl;
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

    return 0;
  }

  std::string Options::get_help() const
  {
    std::ostringstream s;

    s << "Usage: " << invocation << " [options] <file_1> <file_2>" << std::endl;
    s << std::endl;
    s << "-d, --degree <d>         Read this degree from DIPHA persistence diagram files. Mandatory if reading DIPHA persistence file." << std::endl;
    s << "--devices <list>         Use these GPU devices on the selected platform. Use --list to see a list. Mandatory." << std::endl;
    s << "                         Comma-separated non-negative integers. Inclusive ranges can be specified by m-n, and can be part of the list. Mandatory." << std::endl;
    s << "--64, --double           Do GPU computations with double precision. Default is single precision." << std::endl;
    s << "-f, --finitization <f>   Make infinite intervals die at f. Mandatory." << std::endl;
    s << "-k, --kernel <k>         Kernel to compute. Optional. Valid options are:" << std::endl;
    s << "                          - pssk: (Default) Persistence Scale Space Kernel" << std::endl;
    s << "                          - heat: Synonym for \"pssk\"" << std::endl;
    s << "                          - pwgk: Persistence Weighted Gaussian Kernel" << std::endl;
    s << "-h, --help               Print this help message." << std::endl;
    s << "-l, --list               List all OpenCL platforms and devices available on the system." << std::endl;
    s << "-o, --output <f>         Use f as the output file name. Use - for standard output. [Default: -]" << std::endl;
    s << "-p, --platform <p>       Select platform p. Use --list to see a list. Mandatory." << std::endl;
    s << "--power <p>              Parameter p. Positive real. Mandatory for PWGK." << std::endl;
    s << "-s, --sigma <s>          Parameter sigma. Meaning depends on kernel. Positive real. Mandatory." << std::endl;
    s << "-v, --verbosity          Increase verbosity level. Can be repeated." << std::endl;
    s << "--version                Print version and exit." << std::endl;
    s << "--weight <c>             Parameter c. Positive real. Mandatory for PWGK." << std::endl;
    s << "--workshape <w>          Use this local work shape on the GPUs. [Default: max]" << std::endl;
    s << "                         Comma-separated pair of positive integers, or \"max\" for automatic." << std::endl;
    s << "<file_1> <file_2>        These should refer to two text files, each containing a list of persistence diagram files to process, one per line. Mandatory." << std::endl;
    s << "                         If file_list_1 contains M files and file_list_2 contains N files, then the output is an MxN matrix of the corresponding kernel values." << std::endl;
    s << "                         If file_list_1 and file_list_2 are the same file, then the computation will exploit symmetry and do half the work. The output will be the same." << std::endl;

    s << std::endl;

    return s.str();
  }
}

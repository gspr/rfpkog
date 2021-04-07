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

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

#include "cl_headers.hpp"
#include "dtype.hpp"
#include "options.hpp"
#include "rfpkog.hpp"

int main(int argc, char ** argv)
{
  rfpkog::Options opts;
  int opterr = opts.parse(argc, argv);
  if (opterr)
  {
    std::cerr << opts.get_help();
    return 1;
  }
  opterr = opts.validate();
  if (opterr)
  {
    std::cerr << opts.get_help();
    return 1;
  }

  if (opts.help)
  {
    std::cout << opts.get_help();
    return 0;
  }

  if (opts.print_version)
  {
    std::cout << "RFPKOG version " << RFPKOG_VERSION << std::endl;
    std::cout << std::endl;
    std::cout << opts.copyright << std::endl;
    return 0;
  }
  
  cl_int status;
  
  std::vector<cl::Platform> platforms_available;
  status = cl::Platform::get(&platforms_available);
  if (status != CL_SUCCESS)
  {
    std::cerr << "Failed to get platform list. OpenCL error code " << status << "." << std::endl;
    return 1;
  }
  if (opts.verbosity >= 2)
  {
    std::cerr << "Found " << platforms_available.size() << " platforms:" << std::endl;
    for (std::size_t i = 0; i < platforms_available.size(); ++i)
    {
      std::cerr << " * " << i << ": " << platforms_available[i].getInfo<CL_PLATFORM_NAME>() << std::endl;
    }
  }

  if (opts.list_devices)
  {
    std::cout << "Platforms and devices:" << std::endl;
    for (std::size_t i = 0; i < platforms_available.size(); ++i)
    {
      std::cout << " * ";
      std::cout << std::setw(static_cast<int>(std::log10(platforms_available.size())) + 1) << i;
      std::cout << ": " << platforms_available[i].getInfo<CL_PLATFORM_NAME>() << std::endl;
      std::vector<cl::Device> devices_available;
      status = platforms_available[i].getDevices(CL_DEVICE_TYPE_ALL, &devices_available);
      if (status != CL_SUCCESS)
      {
        std::cerr << "Failed to get device list for platform " << i << ". OpenCL error code " << status << "." << std::endl;
        return 1;
      }
      for (std::size_t j = 0; j < devices_available.size(); ++j)
      {
        std::cout << "   - ";
        std::cout << std::setw(static_cast<int>(std::log10(devices_available.size())) + 1) << j;
        std::cout << ": " << devices_available[j].getInfo<CL_DEVICE_NAME>() << std::endl;
      }     
    }
    return 0;
  }

  if (opts.platform_id >= platforms_available.size())
  {
    std::cerr << "Selected platform (ID " << opts.platform_id << ") not available." << std::endl;
    return 1;
  }

  cl::Platform platform = platforms_available[opts.platform_id];

  if (opts.verbosity >= 2)
    std::cerr << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << "." << std::endl;

  std::vector<cl::Device> devices_available;
  status = platform.getDevices(CL_DEVICE_TYPE_ALL, &devices_available);
  if (status != CL_SUCCESS)
  {
    std::cerr << "Failed to get device list for platform " << opts.platform_id << ". OpenCL error code " << status << "." << std::endl;
    return 1;
  }

  if (opts.verbosity >= 2)
  {
    std::cerr << "Found " << devices_available.size() << " devices:" << std::endl;
    for (std::size_t i = 0; i < devices_available.size(); ++i)
    {
      std::cerr << " * " << i << ": " << devices_available[i].getInfo<CL_DEVICE_NAME>() << std::endl;
    }
  }

  std::vector<cl::Device> devices;
  for (std::size_t i = 0; i < devices_available.size(); ++i)
  {
    if (opts.device_ids.empty() || opts.device_ids.find(i) != std::end(opts.device_ids))
      devices.push_back(devices_available[i]);
  }

  if (devices.empty())
  {
    std::cerr << "Unable to select any devices." << std::endl;
    return 1;
  }

  if (opts.verbosity >= 2)
  {
    std::cerr << "Using these devices:" << std::endl;
    for (auto d : devices)
      std::cerr << " * " << d.getInfo<CL_DEVICE_NAME>() << std::endl;
  }

  for (std::size_t i = 0; i < devices.size(); ++i)
  {
    if (!devices[i].getInfo<CL_DEVICE_ENDIAN_LITTLE>())
    {
      std::cerr << "Device with ID " << i << " is not little endian. This is currently unsupported." << std::endl;
      return 1;
    }
  }
  

  cl::Context context(devices, NULL, NULL, NULL, &status);
  if (status != CL_SUCCESS)
  {
    std::cerr << "Failed to create OpenCL context. OpenCL error code " << status << "." << std::endl;
    return 1;
  }

  std::ifstream kernel_stream(opts.kernel_fname);
  if (!kernel_stream.is_open())
  {
    std::cerr << "Failed to open kernel file " << opts.kernel_fname << "." << std::endl;
    return 1;
  }
  std::stringstream buffer;
  buffer << kernel_stream.rdbuf();
  kernel_stream.close();
  std::string kernel_src(buffer.str());

  
  cl::Program kernel_program(context, kernel_src, false, &status);
  if (status != CL_SUCCESS)
  {
    std::cerr << "Error creating OpenCL program. OpenCL error code " << status << "." << std::endl;
    return 1;
  }
  if (opts.verbosity >= 1)
    std::cerr << "Building OpenCL program." << std::endl;
  std::string build_options(opts.use_double ? "-DDTYPE_DOUBLE" : "");
  status = kernel_program.build(devices, build_options.c_str(), NULL, NULL);
  auto build_status = kernel_program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>();
  for (std::size_t i = 0; i < build_status.size(); ++i)
  {
    if (build_status[i].second != CL_BUILD_SUCCESS)
    {
      std::cerr << "Failed to build program for device ID " << i << "." << std::endl;
      auto build_log = kernel_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(build_status[i].first, &status);
      if (status != CL_SUCCESS)
      {
        std::cerr << "Failed to get build log. OpenCL error code " << status << "." << std::endl;
        return 1;
      }
      std::string logfname("rfpkog_build_device_id_" + std::to_string(i) + std::string(".log"));
      
      std::ofstream log_stream(logfname);
      if (!log_stream.is_open())
      {
        std::cerr << "Failed to open log file to record build failure." << std::endl;
        return 1;
      }
      log_stream << build_log;
      log_stream.close();
      std::cerr << "Build log available in " << logfname << "." << std::endl;
      
      if (opts.verbosity >= 2)
      {
        std::cerr << "=== BEGIN BUILD LOG ===" << std::endl;
        std::cerr << build_log << std::endl;
        std::cerr << "=== END BUILD LOG ===" << std::endl;
      }
    }
  }

  
  std::vector<cl::CommandQueue> cmd_qs;
  for (std::size_t i = 0; i < devices.size(); ++i)
  {
    cmd_qs.emplace_back(context, devices[i], 0, &status);
    if (status != CL_SUCCESS)
    {
      std::cerr << "Failed to make command queue for device " << i << ". OpenCL error code " << status << "." << std::endl;
      return 1;
    }
  }

  
  if (opts.verbosity >= 1)
  {
    std::cerr << "Initiating computations." << std::endl;
  }

  std::vector<double> results;
  int rfpkog_status = 0;
  if (opts.use_double)
  {
    rfpkog::RFPKOG<rfpkog::DoubleType> rfpkog(context, cmd_qs, devices, kernel_program, opts);
    if (!rfpkog.setup_complete())
    {
      std::cerr << "Encountered error in setup." << std::endl;
      return 1;
    }
    rfpkog_status = rfpkog.run();
    results = rfpkog.results();
  }
  else
  {
    rfpkog::RFPKOG<rfpkog::FloatType> rfpkog(context, cmd_qs, devices, kernel_program, opts);
    if (!rfpkog.setup_complete())
    {
      std::cerr << "Encountered error in setup." << std::endl;
      return 1;
    }
    rfpkog_status = rfpkog.run();
    results = rfpkog.results();
  }
  if (rfpkog_status)
  {
    std::cerr << "Main loop encountered an error. Not writing any results." << std::endl;
    return 1;
  }
  

  std::ofstream output_file;
  if (!opts.output_fname.empty())
  {
    output_file.open(opts.output_fname, std::ios::out);
    if (!output_file.is_open())
    {
      std::cerr << "Failed to open output file " << opts.output_fname << "." << std::endl;
      return 1;
    }
  }
  std::ostream & output = (opts.output_fname.empty() ? std::cout : output_file);

  output << std::setprecision(15);
  for (std::size_t i = 0; i < opts.fnames[0].size(); ++i)
  {
    for (std::size_t j = 0; j < opts.fnames[1].size(); ++j)
    {
      output << results[i*opts.fnames[0].size() + j] << " ";
    }
    output << std::endl;
  }

  if (!opts.output_fname.empty())
    output_file.close();
  
  return 0;
}

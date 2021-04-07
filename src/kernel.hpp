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

#include <array>
#include <limits>
#include <string>

#include "cl_headers.hpp"
#include "dtype.hpp"
#include "pd.hpp"

namespace rfpkog
{
  template <typename T>
  class Kernel
  {
  public:
    Kernel(cl::Context & context,
           const cl::Device & device,
           cl::CommandQueue & cmd_q,
           const cl::Program & program, 
           const std::array<std::size_t, 2> & local_work_shape) :
      context(context), device(device), cmd_q(cmd_q), program(program), local_work_shape(local_work_shape),
      N({0, 0}), global_prefactor(std::numeric_limits<double>::quiet_NaN()), status(CL_SUCCESS)
    {
    }

    virtual ~Kernel()
    {
    }

    int init()
    {
      int err;
      
      if (local_work_shape[0] == 0 || local_work_shape[1] == 0)
      {
        err = autodetermine_local_work_shape();
        if (err)
          return 1;
      }

      err = init_kernel_args();
      if (err)
        return 1;
      
      return 0;
    }

    double sum() const
    {
      double ret = 0.0;
      for (auto it = partial_sums.cbegin(); it != partial_sums.cend(); ++it)
        ret += static_cast<double>(*it);
      return ret*global_prefactor;
    }

    inline cl_int get_status() const { return status; }
    inline bool is_ok() const { return get_status() == CL_SUCCESS; }
    inline std::array<std::size_t, 2> get_local_work_shape() const { return local_work_shape; }
    
    virtual int autodetermine_local_work_shape() = 0;
    virtual int init_kernel_args() = 0;
    virtual int prepare_new_pd(std::size_t a, const std::vector<typename T::vector_type> & pd) = 0;
    virtual int compute_partial_sums() = 0;

  protected:
    cl::Context & context;
    const cl::Device & device;
    cl::CommandQueue & cmd_q;
    const cl::Program & program;
    std::array<std::size_t, 2> local_work_shape;
    std::array<cl::Buffer, 2> pd_bufs;
    std::array<std::size_t, 2> N;
    std::vector<typename T::scalar_type> partial_sums;
    cl::Buffer partial_sums_buf;
    double global_prefactor;

    cl_int status;
  };
}

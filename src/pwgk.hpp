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
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

#include "cl_headers.hpp"
#include "dtype.hpp"
#include "kernel.hpp"
#include "misc.hpp"
#include "pd.hpp"

namespace rfpkog
{
  template <typename T>
  class PWGK : public Kernel<T>
  {
  public:
    PWGK(cl::Context & context,
         cl::CommandQueue & cmd_q,
         const cl::Device & device,
         const cl::Program & program, 
         const std::array<std::size_t, 2> & local_work_shape,
         typename T::scalar_type sigma, typename T::scalar_type p, typename T::scalar_type c) :
      Kernel<T>(context, device, cmd_q, program, local_work_shape),
      sigmasq(std::pow(sigma, 2)), p(p), c(c)
    {
      kernel = cl::Kernel(program, "rfpkog_pwgk", &(Kernel<T>::status));
      if (Kernel<T>::status != CL_SUCCESS)
      {
        std::cerr << "Failed to create kernel rfpkog_pwgk. OpenCL error code " << Kernel<T>::status << "." << std::endl;
      }
      Kernel<T>::global_prefactor = 1.0;
    }

    int autodetermine_local_work_shape()
    {
      int & status = Kernel<T>::status; // For convenience.
      std::size_t pwgsm = kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(Kernel<T>::device, &status);
      if (status != CL_SUCCESS)
      {
        std::cerr << "Failed to automatically determine max good workgroup shape. OpenCL error code " << status << "." << std::endl;
        return 1;
      }
      std::size_t wgs = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(Kernel<T>::device, &status);
      if (status != CL_SUCCESS)
      {
        std::cerr << "Failed to automatically determine max good workgroup shape. OpenCL error code " << status << "." << std::endl;
        return 1;
      }
      Kernel<T>::local_work_shape = {1, (wgs/pwgsm)*pwgsm};
      if (Kernel<T>::local_work_shape[1] == 0)
      {
        std::cerr << "Failed to automatically determine max good workgroup shape. OpenCL error code " << status << "." << std::endl;
        return 1;
      }

      return 0;
    }

    int init_kernel_args()
    {
      // For convenience.
      int & status = Kernel<T>::status;
      const std::array<std::size_t, 2> & local_work_shape = Kernel<T>::local_work_shape;
      
      status = kernel.setArg(0, sigmasq);
      if (status != CL_SUCCESS)
      {
        std::cerr << "Failed to set argument 0 of PWGK. OpenCL error code " << status << "." << std::endl;
        return 1;
      }
      status = kernel.setArg(1, p);
      if (status != CL_SUCCESS)
      {
        std::cerr << "Failed to set argument 1 of PWGK. OpenCL error code " << status << "." << std::endl;
        return 1;
      }
      status = kernel.setArg(2, c);
      if (status != CL_SUCCESS)
      {
        std::cerr << "Failed to set argument 2 of PWGK. OpenCL error code " << status << "." << std::endl;
        return 1;
      }
      status = kernel.setArg(3, cl::Local(local_work_shape[0]*sizeof(typename T::vector_type)));
      if (status != CL_SUCCESS)
      {
        std::cerr << "Failed to set argument 3 of PWGK. OpenCL error code " << status << "." << std::endl;
        return 1;
      }
      status = kernel.setArg(4, cl::Local(local_work_shape[1]*sizeof(typename T::vector_type)));
      if (status != CL_SUCCESS)
      {
        std::cerr << "Failed to set argument 4 of PWGK. OpenCL error code " << status << "." << std::endl;
        return 1;
      }
      status = kernel.setArg(7, cl::Local(local_work_shape[0]*local_work_shape[1]*sizeof(typename T::scalar_type)));
      if (status != CL_SUCCESS)
      {
        std::cerr << "Failed to set argument 7 of PWGK. OpenCL error code " << status << "." << std::endl;
        return 1;
      }
      return 0;
    }
    

    int prepare_new_pd(std::size_t a, const std::vector<typename T::vector_type> & pd)
    {
      int & status = Kernel<T>::status; // For convenience.
      
      Kernel<T>::N[a] = pd.size();
      
      Kernel<T>::pd_bufs[a] = cl::Buffer(Kernel<T>::context, CL_MEM_READ_ONLY, Kernel<T>::N[a]*sizeof(typename T::vector_type), NULL, &status);
      if (status != CL_SUCCESS)
        return 1;

      status = Kernel<T>::cmd_q.enqueueWriteBuffer(Kernel<T>::pd_bufs[a], true, 0, Kernel<T>::N[a]*sizeof(typename T::vector_type), pd.data(), NULL, NULL);
      if (status != CL_SUCCESS)
        return 1;

      status = kernel.setArg(a == 0 ? 5 : 6, Kernel<T>::pd_bufs[a]);
      if (status != CL_SUCCESS)
        return 1;

      return 0;
    }

    int compute_partial_sums()
    {
      int & status = Kernel<T>::status; // For convenience.
      const std::array<std::size_t, 2> & local_work_shape = Kernel<T>::local_work_shape;
      const std::array<std::size_t, 2> & N = Kernel<T>::N;
        
      Kernel<T>::partial_sums.resize((N[0]/local_work_shape[0])*(N[1]/local_work_shape[1]), std::numeric_limits<double>::quiet_NaN());

      Kernel<T>::partial_sums_buf = cl::Buffer(Kernel<T>::context, CL_MEM_READ_WRITE, Kernel<T>::partial_sums.size()*sizeof(typename T::scalar_type), NULL, &status);
      if (status != CL_SUCCESS)
        return 1;

      status = kernel.setArg(8, Kernel<T>::partial_sums_buf);
      if (status != CL_SUCCESS)
        return 1;

      cl::Event event;
      status = Kernel<T>::cmd_q.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(N[0], N[1]), cl::NDRange(local_work_shape[0], local_work_shape[1]), NULL, &event);
      if (status != CL_SUCCESS)
        return 1;
      event.wait();

      status = Kernel<T>::cmd_q.enqueueReadBuffer(Kernel<T>::partial_sums_buf, true, 0, Kernel<T>::partial_sums.size()*sizeof(typename T::scalar_type), Kernel<T>::partial_sums.data(), NULL, NULL);
      if (status != CL_SUCCESS)
        return 1;
      
      return 0;
    }
    
  private:
    typename T::scalar_type sigmasq;
    typename T::scalar_type p;
    typename T::scalar_type c;
    cl::Kernel kernel;
  };
}

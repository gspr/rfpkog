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
#include <array>
#include <chrono>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <CL/cl2.hpp>

#include "dtype.hpp"
#include "pd.hpp"

namespace rfpkog
{
  template <typename T>
  class RFPKOG
  {
  public:
    RFPKOG(cl::Context & context,
           std::vector<cl::CommandQueue> & cmd_qs,
           std::vector<cl::Kernel> & kernels,
           const std::vector<std::array<std::size_t, 2> > & local_work_shapes,
           typename T::scalar_type sigma,
           typename T::scalar_type finitization,
           unsigned int degree,
           unsigned int verbosity,
           bool symmetric,
           const std::array<std::vector<std::string >, 2> & fnames) :
      context(context), cmd_qs(cmd_qs), kernels(kernels), local_work_shapes(local_work_shapes),
      sigma8(8*sigma), finitization(finitization), degree(degree), verbosity(verbosity), symmetric(symmetric),
      fnames(fnames), idxs({0,0}), done(false), mutex()
    {
      results = new double[fnames[0].size()*fnames[1].size()];
      std::fill(results, results + fnames[0].size()*fnames[1].size(), std::numeric_limits<double>::quiet_NaN());

      statuses = new cl_int[cmd_qs.size()];
      std::fill(statuses, statuses + cmd_qs.size(), CL_SUCCESS);
    }

    ~RFPKOG()
    {
      delete[] statuses;
      delete[] results;
    }

    RFPKOG(const RFPKOG &) = delete;
    RFPKOG & operator=(const RFPKOG &) = delete;

    std::vector<double> get_results() const
    {
      std::vector<double> ret;
      ret.reserve(fnames[0].size()*fnames[1].size());
      std::copy(results, results + fnames[0].size()*fnames[1].size(), std::back_inserter(ret));
      return ret;
    }
    
    int run()
    {
      std::vector<std::thread> workers;
      for (std::size_t w = 0; w < cmd_qs.size(); ++w)
        workers.emplace_back(&RFPKOG::worker, this, w);

      for (std::size_t w = 0; w < workers.size(); ++w)
      {
        workers[w].join();
      }

      bool success = true;
      for(std::size_t w = 0; w < workers.size(); ++w)
      {
        if (statuses[w] == 1337) // FIXME
        {
          std::cerr << "Worker thread " << w << " failed to load a persistence diagram." << std::endl;
          success = false;
        }
        else if (statuses[w] != CL_SUCCESS)
        {
          std::cerr << "Worker thread " << w << " encountered OpenCL error code " << statuses[w] << "." << std::endl;
          success = false;
        }
      }

      if (!success)
        return 1;
      
      return 0;
    }

  private:
    cl::Context & context;
    std::vector<cl::CommandQueue> & cmd_qs;
    std::vector<cl::Kernel> & kernels;
    const std::vector<std::array<std::size_t, 2> > & local_work_shapes;
    const typename T::scalar_type sigma8;
    const typename T::scalar_type finitization;
    const unsigned int degree;
    const unsigned int verbosity;
    const bool symmetric;
    const std::array<std::vector<std::string >, 2> & fnames;
    std::array<std::size_t, 2> idxs;
    bool done;
    double * results; // I believe that in theory, concurrent writing to different elements of an std::vector can be UB. We'll play it safe and do it raw.
    cl_int * statuses;
    std::mutex mutex;

    inline void advance()
    {
      ++idxs[1];
      if (idxs[1] >= fnames[1].size())
      {
        ++idxs[0];
        idxs[1] = (symmetric ? idxs[0] : 0);
      }
      if (idxs[0] >= fnames[0].size())
        done = true;
    }


    // ATTENTION: This method contains a lot of variables with names that shadow members.
    void worker(const std::size_t w)
    {
      bool done = false;
      std::array<std::size_t, 2> idxs = {std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::size_t>::max()};
      std::array<std::size_t, 2> prev_idxs = {std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::size_t>::max()};
      
      std::array<std::vector<typename T::vector_type>, 2> pds;
      cl::Buffer result_buf;
      std::array<cl::Buffer, 2> pd_bufs;

      cl::Kernel & kernel = kernels[w];
      
      std::unique_lock lock(this->mutex, std::defer_lock);

      std::chrono::time_point<std::chrono::steady_clock> t_0;
      std::chrono::time_point<std::chrono::steady_clock> t_1;

      cl_int status = CL_SUCCESS;
      
      status = kernel.setArg(0, sigma8);
      if (status != CL_SUCCESS)
      {
        lock.lock();
        statuses[w] = status;
        lock.unlock();
      }
      status = kernel.setArg(1, cl::Local(local_work_shapes[w][0]*sizeof(typename T::vector_type)));
      if (status != CL_SUCCESS)
      {
        lock.lock();
        statuses[w] = status;
        lock.unlock();
      }
      status = kernel.setArg(2, cl::Local(local_work_shapes[w][0]*sizeof(typename T::vector_type)));
      if (status != CL_SUCCESS)
      {
        lock.lock();
        statuses[w] = status;
        lock.unlock();
      }
      status = kernel.setArg(3, cl::Local(local_work_shapes[w][1]*sizeof(typename T::vector_type)));
      if (status != CL_SUCCESS)
      {
        lock.lock();
        statuses[w] = status;
        lock.unlock();
      }
      status = kernel.setArg(6, cl::Local(local_work_shapes[w][0]*local_work_shapes[w][1]*sizeof(typename T::scalar_type)));
      if (status != CL_SUCCESS)
      {
        lock.lock();
        statuses[w] = status;
        lock.unlock();
      }
      
      while (true)
      {
        if (verbosity >= 3)
        {
          lock.lock();
          std::cerr << "Worker thread " << w << " entering main loop." << std::endl;
          lock.unlock();
        }

        lock.lock();
        done = this->done;
        prev_idxs = idxs;
        idxs = this->idxs;
        advance();
        lock.unlock();

        if (done)
          break;

        if (verbosity >= 3)
        {
          lock.lock();
          std::cerr << "Worker thread " << w << " has a valid work unit (" << idxs[0] << "," << idxs[1] << ")." <<  std::endl;
          lock.unlock();
        }
        
        for (auto a : {0, 1})
        {
          if (idxs[a] != prev_idxs[a])
          {
            if (verbosity >= 3)
            {
              lock.lock();
              std::cerr << "Worker thread " << w << " is reloading PD_" << a << "." << std::endl;
              lock.unlock();
            }
            pds[a].clear();
            int pd_err = read_dipha_degree<T>(fnames[a][idxs[a]], degree, finitization, pds[a]);
            if (pd_err)
            {
              lock.lock();
              statuses[w] = 1337; // FIXME.
              lock.unlock();
              return;
            }
            if (pds[a].size() % local_work_shapes[w][a] != 0)
            {
              typename T::vector_type zero;
              zero.s[0] = 0;
              zero.s[1] = 0;
              if (verbosity >= 4)
              {
                lock.lock();
                std::cerr << "Worker thread " << w << " is padding PD_" << a << " from size " << pds[a].size() << " to size " << (pds[a].size() + local_work_shapes[w][a] - (pds[a].size() % local_work_shapes[w][a])) << " to keep it a multiple of " << local_work_shapes[w][a] << "." << std::endl;
                lock.unlock();
              }
              pds[a].resize(pds[a].size() + local_work_shapes[w][a] - (pds[a].size() % local_work_shapes[w][a]), zero);
            }
            
            pd_bufs[a] = cl::Buffer(context, CL_MEM_READ_ONLY, pds[a].size()*sizeof(typename T::vector_type), NULL, &status);
            if (status != CL_SUCCESS)
            {
              lock.lock();
              statuses[w] = status;
              lock.unlock();
              return;
            }
            
            status = cmd_qs[w].enqueueWriteBuffer(pd_bufs[a], true, 0, pds[a].size()*sizeof(typename T::vector_type), pds[a].data(), NULL, NULL);
            if (status != CL_SUCCESS)
            {
              lock.lock();
              statuses[w] = status;
              lock.unlock();
              return;
            }
            
          }
        }

        std::size_t thread_result_size = (pds[0].size()/local_work_shapes[w][0])*(pds[1].size()/local_work_shapes[w][1]);
        typename T::scalar_type * thread_result = new typename T::scalar_type[thread_result_size];
        result_buf = cl::Buffer(context, CL_MEM_READ_WRITE, thread_result_size*sizeof(typename T::scalar_type), NULL, &status);
        if (status != CL_SUCCESS)
        {
          lock.lock();
          statuses[w] = status;
          lock.unlock();
          delete[] thread_result;
          return;
        }
          
        status = kernel.setArg(7, result_buf);
        if (status != CL_SUCCESS)
        {
          lock.lock();
          statuses[w] = status;
          lock.unlock();
          delete[] thread_result;
          return;
        }
        
        status = kernel.setArg(4, pd_bufs[0]);
        if (status != CL_SUCCESS)
        {
          lock.lock();
          statuses[w] = status;
          lock.unlock();
          delete[] thread_result;
          return;
        }
        
        status = kernel.setArg(5, pd_bufs[1]);
        if (status != CL_SUCCESS)
        {
          lock.lock();
          statuses[w] = status;
          lock.unlock();
          delete[] thread_result;
          return;
        }
        
        if (verbosity >= 3)
        {
          lock.lock();
          std::cerr << "Worker thread " << w << " will now run kernel." << std::endl;
          lock.unlock();
        }
        if (verbosity >= 1)
        {
          t_0 = std::chrono::steady_clock::now();
        }

        cl::Event event;
        status = cmd_qs[w].enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(pds[0].size(), pds[1].size()), cl::NDRange(local_work_shapes[w][0], local_work_shapes[w][1]), NULL, &event);
        if (status != CL_SUCCESS)
        {
          lock.lock();
          statuses[w] = status;
          lock.unlock();
          delete[] thread_result;
          return;
        }
        event.wait();

        status = cmd_qs[w].enqueueReadBuffer(result_buf, true, 0, thread_result_size*sizeof(typename T::scalar_type), thread_result, NULL, NULL);
        if (status != CL_SUCCESS)
        {
          lock.lock();
          statuses[w] = status;
          lock.unlock();
          delete[] thread_result;
          return;
        }

        double sum = 0;
        for (std::size_t i = 0; i < thread_result_size; ++i)
          sum += thread_result[i];
        delete[] thread_result;
        
        this->results[idxs[0]*fnames[1].size() + idxs[1]] = sum / (sigma8*PI);
        if (this->symmetric && idxs[1] < idxs[0])
        {
          this->results[idxs[1]*fnames[1].size() + idxs[0]] = this->results[idxs[0]*fnames[1].size() + idxs[1]];
        }

        if (verbosity >= 1)
        {
          t_1 = std::chrono::steady_clock::now();
          std::chrono::duration<double> elapsed = t_1 - t_0;
          lock.lock();
          std::cerr << "Worker thread " << w << " computed result (" << idxs[0] << "," << idxs[1] << ") in " << elapsed.count() << " s." << std::endl;
          lock.unlock();
        }
      } // End main loop.

      if (verbosity >= 3)
      {
        lock.lock();
        std::cerr << "Worker thread " << w << " is ending." << std::endl;
        lock.unlock();
      }
    }

    
  };

}

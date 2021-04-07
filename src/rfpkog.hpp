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

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "cl_headers.hpp"
#include "dtype.hpp"
#include "kernel.hpp"
#include "heat_kernel.hpp"
#include "options.hpp"
#include "pd.hpp"

namespace rfpkog
{
  template <typename T>
  class RFPKOG
  {
  public:
    RFPKOG(cl::Context & context,
           std::vector<cl::CommandQueue> & cmd_qs,
           const std::vector<cl::Device> & devices, 
           const cl::Program & program,
           const Options & opts) :
      context_(context), cmd_qs_(cmd_qs), devices_(devices), program_(program), opts_(opts),
      statuses_(devices.size(), CL_SUCCESS),
      errors_(devices.size(), 0),
      idxs_({0,0}), results_(opts.fnames[0].size()*opts.fnames[1].size(), std::numeric_limits<double>::quiet_NaN()),
      mutex_(), done_(false), setup_complete_(false)
    {
      for (std::size_t i = 0; i < cmd_qs.size(); ++i)
      {
        switch (opts_.kernel_choice)
        {
        case Kernel_choice::pssk:
          kernels_.push_back(std::make_unique<Heat_kernel<T>>(context_, cmd_qs_[i], devices_[i], program_, opts_.local_work_shape, opts_.sigma));
          break;
        }

        if (!kernels_.back()->is_ok())
        {
          statuses_[i] = kernels_.back()->get_status();
          return;
        }

        if (kernels_.back()->init() != 0)
        {
          statuses_[i] = kernels_.back()->get_status();
          return;
        }
      }
      setup_complete_ = true;
    }

    inline bool setup_complete() const { return setup_complete_; }
    inline const std::vector<double> & results() const { return results_; }

    int run()
    {
      std::vector<std::thread> workers;
      for (std::size_t w = 0; w < cmd_qs_.size(); ++w)
        workers.emplace_back(&RFPKOG::worker, this, w);

      for (std::size_t w = 0; w < workers.size(); ++w)
      {
        workers[w].join();
      }

      bool success = true;
      for(std::size_t w = 0; w < workers.size(); ++w)
      {
        if (errors_[w] != 0)
        {
          std::cerr << "Worker thread " << w << " encountered an error." << std::endl;
          success = false;
        }
        if (statuses_[w] != CL_SUCCESS)
        {
          std::cerr << "Worker thread " << w << " encountered OpenCL error code " << statuses_[w] << "." << std::endl;
          success = false;
        }
      }

      if (!success)
        return 1;
      
      return 0;
    }

  private:
    cl::Context & context_;
    std::vector<cl::CommandQueue> & cmd_qs_;
    const std::vector<cl::Device> & devices_;
    const cl::Program & program_;
    const Options & opts_;
    
    std::vector<cl_int> statuses_;
    std::vector<int> errors_;

    std::array<std::size_t, 2> idxs_;
    std::vector<double> results_;
    std::mutex mutex_;
    bool done_;
    bool setup_complete_;
    std::vector<std::unique_ptr<Kernel<T> > > kernels_;

    
    inline void advance()
    {
      ++idxs_[1];
      if (idxs_[1] >= opts_.fnames[1].size())
      {
        ++idxs_[0];
        idxs_[1] = (opts_.symmetric ? idxs_[0] : 0);
      }
      if (idxs_[0] >= opts_.fnames[0].size())
        done_ = true;
    }
    
    
    void worker(const std::size_t w)
    {
      bool done = false;
      std::array<std::size_t, 2> idxs = {std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::size_t>::max()};
      std::array<std::size_t, 2> prev_idxs = idxs;
      std::array<std::vector<typename T::vector_type>, 2> pds;
      std::unique_lock lock(mutex_, std::defer_lock);
      const std::array<std::size_t, 2> local_work_shape = kernels_[w]->get_local_work_shape();
      int err = 0;

      std::chrono::time_point<std::chrono::steady_clock> t_0;
      std::chrono::time_point<std::chrono::steady_clock> t_1;

      while (true)
      {
        lock.lock();
        done = done_;
        prev_idxs = idxs;
        idxs = idxs_;
        advance();
        lock.unlock();

        if (done)
          break;

        if (opts_.verbosity >= 3)
        {
          lock.lock();
          std::cerr << "Worker thread " << w << " has a valid work unit (" << idxs[0] << "," << idxs[1] << ")." <<  std::endl;
          lock.unlock();
        }

        for (auto a : {0, 1})
        {
          if (idxs[a] != prev_idxs[a])
          {
            if (opts_.verbosity >= 3)
            {
              lock.lock();
              std::cerr << "Worker thread " << w << " is reloading PD_" << a << "." << std::endl;
              lock.unlock();
            }
            pds[a].clear();
            err = read_dipha_degree<T>(opts_.fnames[a][idxs[a]], opts_.degree, opts_.finitization, pds[a]);
            if (err)
            {
              lock.lock();
              std::cerr << "Thread " << w << " failed to read persistence diagram." << std::endl;
              errors_[w] = err;
              lock.unlock();
            }

            if (pds[a].size() % local_work_shape[a] != 0)
            {
              typename T::vector_type zero;
              zero.s[0] = 0;
              zero.s[1] = 0;
              if (opts_.verbosity >= 4)
              {
                lock.lock();
                std::cerr << "Worker thread " << w << " is padding PD_" << a << " from size " << pds[a].size() << " to size " << (pds[a].size() + local_work_shape[a] - (pds[a].size() % local_work_shape[a])) << " to keep it a multiple of " << local_work_shape[a] << "." << std::endl;
                lock.unlock();
              }
              pds[a].resize(pds[a].size() + local_work_shape[a] - (pds[a].size() % local_work_shape[a]), zero);
            }
            
            err = kernels_[w]->prepare_new_pd(a, pds[a]);
            if (err)
            {
              lock.lock();
              statuses_[w] = kernels_[w]->get_status();
              std::cerr << "Kernel in worker thread " << w << " encountered error. OpenCL error code " << statuses_[w] << "." << std::endl;
              lock.unlock();
              return;
            }
          }
        }

        if (opts_.verbosity >= 3)
        {
          lock.lock();
          std::cerr << "Worker thread " << w << " will now run kernel." << std::endl;
          lock.unlock();
        }
        if (opts_.verbosity >= 1)
        {
          t_0 = std::chrono::steady_clock::now();
        }

        err = kernels_[w]->compute_partial_sums();
        results_[idxs[0]*opts_.fnames[1].size() + idxs[1]] = kernels_[w]->sum();
        if (opts_.symmetric)
        {
          results_[idxs[1]*opts_.fnames[1].size() + idxs[0]] = results_[idxs[0]*opts_.fnames[1].size() + idxs[1]];
        }

        if (opts_.verbosity >= 1)
        {
          t_1 = std::chrono::steady_clock::now();
          std::chrono::duration<double> elapsed = t_1 - t_0;
          lock.lock();
          std::cerr << "Worker thread " << w << " computed result (" << idxs[0] << "," << idxs[1] << ") in " << elapsed.count() << " s." << std::endl;
          lock.unlock();
        }
        
      } // End main loop.

      if (opts_.verbosity >= 3)
      {
        lock.lock();
        std::cerr << "Worker thread " << w << " is ending." << std::endl;
        lock.unlock();
      }
    }
    
  };
}

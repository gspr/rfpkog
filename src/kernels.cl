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

#ifdef DTYPE_DOUBLE
typedef double dtype;
typedef double2 vdtype;
#else
typedef float dtype;
typedef float2 vdtype;
#endif

__kernel void rfpkog_heat(const dtype sigma8,
                          __local vdtype * lpd0, __local vdtype * lpd1,
                          const __global vdtype * pd0, const __global vdtype * pd1,
                          __local dtype * partials, __global dtype * out)
{
  const size_t i[2]  = {get_global_id(0)   , get_global_id(1)   };
  const size_t N[2]  = {get_global_size(0) , get_global_size(1) };
  const size_t wi[2] = {get_group_id(0)    , get_group_id(1)    };
  const size_t li[2] = {get_local_id(0)    , get_local_id(1)    };
  const size_t lN[2] = {get_local_size(0)  , get_local_size(1)  };

  const size_t wN[2] = {N[0]/lN[0], N[1]/lN[1]};

  if (li[1] == 0)
  {
    lpd0[li[0]] = pd0[i[0]];
  } 
  if (li[0] == 0)
  {
    lpd1[li[1]] = pd1[i[1]];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  const vdtype x = lpd0[li[0]];
  const vdtype xbar = {x.s1, x.s0};
  const vdtype y = lpd1[li[1]];

  const vdtype diff = x - y;
  const vdtype diffbar = xbar - y;

  partials[li[0]*lN[1] + li[1]] = exp(-dot(diff, diff)/sigma8) - exp(-dot(diffbar, diffbar)/sigma8);
  barrier(CLK_LOCAL_MEM_FENCE);

  for (size_t k = lN[1] / 2; k > 0; k /= 2)
  {
    if (li[1] < k)
    {
      partials[li[0]*lN[1] + li[1]] += partials[li[0]*lN[1] + li[1] + k];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  for (size_t k = lN[0] / 2; k > 0; k /= 2)
  {
    if (li[1] == 0 && li[0] < k)
    {
      partials[li[0]*lN[1]] += partials[(li[0] + k)*lN[1]];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  
  if (li[0] == 0 && li[1] == 0)
  {
    out[wi[0]*wN[1] + wi[1]] = partials[0];
  }
}

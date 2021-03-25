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

__kernel void rfpkog_heat(const dtype sigma8, __local vdtype * lpd1, __local vdtype * lpd2, const __global vdtype * pd1, const __global vdtype * pd2,  __local dtype * partials, __global dtype * out)
{
  const size_t i = get_global_id(0);
  const size_t j = get_global_id(1);
  //const size_t M = get_global_size(0);
  const size_t N = get_global_size(1);

  const size_t wi = get_group_id(0);
  const size_t wj = get_group_id(1);
  
  const size_t li = get_local_id(0);
  const size_t lj = get_local_id(1);
  const size_t lM = get_local_size(0);
  const size_t lN = get_local_size(1);

  //const size_t wM = M/lM;
  const size_t wN = N/lN;

  if (lj == 0)
  {
    lpd1[li] = pd1[i];
  } 
  
  if (li == 0)
  {
    lpd2[lj] = pd2[j];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  const uint2 perm = (uint2)(1, 0);

  const vdtype x = lpd1[li];
  const vdtype xbar = shuffle(x, perm);
  const vdtype y = lpd2[lj];

  const vdtype diff = x - y;
  const vdtype diffbar = xbar - y;

  partials[li*lN + lj] = exp(-dot(diff, diff)/sigma8) - exp(-dot(diffbar, diffbar)/sigma8);
  barrier(CLK_LOCAL_MEM_FENCE);

  for (size_t ljj = lN / 2; ljj > 0; ljj /= 2)
  {
    if (lj < ljj)
    {
      partials[li*lN + lj] += partials[li*lN + lj + ljj];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  for (size_t lii = lM / 2; lii > 0; lii /= 2)
  {
    if (lj == 0 && li < lii)
    {
      partials[li*lN] += partials[(li + lii)*lN];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  
  if (lj == 0 && li == 0)
  {
    out[wi*wN + wj] = partials[0];
  }
}

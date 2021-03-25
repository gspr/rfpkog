project('rfpkog', 'cpp',
  version : '0.2.0',
  default_options : ['cpp_std=c++17'])

dep_threads = dependency('threads')
dep_opencl = dependency('OpenCL', version : '>=1.2.0')

prog_pandoc = find_program('pandoc', get_option('pandoc'), required : false)
if prog_pandoc.found()
  message(prog_pandoc.full_path())
endif

if get_option('legacy_cl_headers')
  add_project_arguments('-DLEGACY_CL_HEADERS', language : 'cpp')
endif

kernel_path = get_option('kernel_path')
if kernel_path == ''
  kernel_path = get_option('prefix') / get_option('datadir') / 'rfpkog'
endif
add_project_arguments('-DKERNEL_FILENAME="' + kernel_path / 'kernels.cl"', language : 'cpp')

add_project_arguments('-DCL_HPP_MINIMUM_OPENCL_VERSION=120',
                      '-DCL_HPP_TARGET_OPENCL_VERSION=120',
                      '-DCL_HPP_CL_1_2_DEFAULT_BUILD',
                      language : 'cpp')

add_project_arguments('-DRFPKOG_VERSION="' + meson.project_version() + '"', language : 'cpp')

executable('rfpkog',
           'src/cl_headers.hpp',
           'src/dtype.hpp',
           'src/main.cpp',
           'src/misc.cpp', 'src/misc.hpp',
           'src/options.cpp', 'src/options.hpp',
           'src/pd.hpp',
           'src/rfpkog.hpp',
           install : true,
           dependencies : [dep_threads, dep_opencl])

install_data('src/kernels.cl')

# This will get less ugly when Meson's GH issue #1550 is fixed.
if prog_pandoc.found()
  manpage = custom_target('manpage',
                          input : ['man/rfpkog.1.md'],
                          output : ['rfpkog.1'],
                          command : [prog_pandoc, '--standalone', '--to', 'man', '-o', '@OUTPUT@', '@INPUT@'],
                          install : true, install_dir : get_option('mandir') / 'man1')
endif

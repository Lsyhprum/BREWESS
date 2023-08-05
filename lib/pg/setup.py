# System imports
from distutils.core import *
from distutils      import sysconfig

# Third-party modules - we depend on numpy for everything
import numpy
import os

os.environ['CC'] = 'g++ -fpic -std=c++17 -O3 -march=native'

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# extension module
module = Extension("_pg",
                   ["pg.i","pg.cpp"],
                   include_dirs=[numpy_include, '/home/zjlab/ANNS/yq/RPQ/lib/pg/nsg/include/efanna2e/', './'],
                   extra_compile_args=["-fopenmp"],
                   extra_link_args=['-lgomp'],
                   extra_objects=['/home/zjlab/ANNS/yq/RPQ/lib/pg/nsg/build/src/CMakeFiles/efanna2e_s.dir/index.cpp.o', '/home/zjlab/ANNS/yq/RPQ/lib/pg/nsg/build/src/CMakeFiles/efanna2e_s.dir/index_nsg.cpp.o'],
                   swig_opts=['-c++']
                   )

# setup
setup(  name        = "pg",
        author      = "HDU",
        version     = "1.0",
        ext_modules = [module]
)
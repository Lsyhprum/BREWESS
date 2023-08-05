"""
Does all sorts of dark magic in order to build/import c++ bfs
"""
import os
import os.path as osp
import setuptools.sandbox


package_abspath = osp.join(*osp.split(osp.abspath(__file__))[:-1])

try:
    import pg
except Exception as e:
    # try build
    workdir = os.getcwd()
    try:
        os.chdir(package_abspath)
        setuptools.sandbox.run_setup(osp.join(package_abspath, 'setup.py'), ['clean', 'build'])
        os.system('cp {}/build/lib*/*.so {}/.'.format(package_abspath, package_abspath))
    except Exception as e:
        raise ImportError("Failed to import pg, please see error log or compile manually")
    finally:
        os.chdir(workdir)

import pg
PG = pg.PG
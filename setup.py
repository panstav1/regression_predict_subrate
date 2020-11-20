
# Libraries
from setuptools import setup, find_packages
import os.path as path
#import numpy as np

# Read requirements.txt file
curDir = path.dirname(__file__)
with open(path.join(curDir, 'packages.txt')) as file:
    requirements = file.read().splitlines()

# Description of service
desc = """
test
"""

setup(name='ex_ty',
      version='0.1',
      url='',
      author='Panagiotis Stavrianos',
      author_email='pstav@unipi.gr',
      long_description=desc,
      package_dir={'': 'src'},
      packages=find_packages('src'),  # dependency resolution
      #include_dirs=[np.get_include()],
      namespace_packages=['model_regr', ],
      include_package_data=True,
      install_requires=requirements,
      zip_safe=False,
      setup_requires=[],
      tests_require=['pytest'])

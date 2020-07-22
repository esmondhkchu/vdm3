# from distutils.core import setup
from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(
  name = 'vdm3',
  packages = ['vdm3'],
  version = '0.0.1',
  license='MIT',
  description = 'Use Value Difference Metric to find distance between categorical features',
  long_description=long_description,
  long_description_content_type="text/markdown",
  author = 'Esmond Chu',
  author_email = 'chuhke@gmail.com',
  url = 'https://github.com/esmondhkchu/vdm3',
  download_url = 'https://github.com/esmondhkchu/vdm3/archive/v0.0.1.tar.gz',
  keywords = ['statistics', 'machine learning', 'distance'],
  test_suite = 'tests',
  install_requires=['numpy','pandas'],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8'
  ],
)

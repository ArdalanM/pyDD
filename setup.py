from setuptools import setup
from setuptools import find_packages

setup(name='pyDD',
      version='0.1.0',
      description='Python binding for DeepDetect',
      author='Ardalan MEHRANI',
      author_email='ardalan.mehrani@iosquare.com',
      url='https://github.com/ArdalanM/pyDD',
      download_url='https://github.com/ArdalanM/pyDD',
      license='MIT',
      classifiers=['License :: MIT License',
                   'Programming Language :: Python',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX',
                   'Operating System :: Unix',
                   'Operating System :: MacOS',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.4',
                   'Programming Language :: Python :: 3.5',
                   ],
      install_requires=[],
      extras_require={},
      packages=find_packages())

import sys
from setuptools.command.test import test as TestCommand
from setuptools import setup


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)


setup(
    name='acc',
    version='0.1.1',
    url='http://github.com/udacity/self-driving-car',
    license='MIT',
    author='Udacity ND013 members',
    description='Adaptive Cruise Control',
    long_description=open('README.md').read() + '\n',
    tests_require=['pytest>=3.0,<3.1', 'coveralls', 'flake8'],
    cmdclass={'test': PyTest},
    py_modules=['acc'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
)

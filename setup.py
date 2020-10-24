from setuptools import setup, find_packages

setup(name='phase_reconstruct',
      version='1.0.0',
      description="reconstruct phase from data",
      author='Yaopeng Ma',
      author_email="yaopeng.ma@biu.ac.il",
      packages=find_packages(),
      install_requires=['numpy', 'matplotlib', 'numba', 'mkl_fft'])

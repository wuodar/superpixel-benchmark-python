wheel:
    python3 setup.py sdist bdist_wheel 

build cython:
    python3 setup.py build_ext --inplace

deploy package:
    https://levelup.gitconnected.com/how-to-deploy-a-cython-package-to-pypi-8217a6581f09

manylinux:
   https://vinayak.io/2020/10/05/day-39-manylinux-is-awesome/ 
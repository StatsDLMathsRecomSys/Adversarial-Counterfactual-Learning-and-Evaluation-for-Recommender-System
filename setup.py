from setuptools import setup, find_packages

print(find_packages())
setup(name='acgan',
      version='1.0',
      packages=['acgan'],
      package_data={"acgan": ["py.typed"]})
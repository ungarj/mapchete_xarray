import os
from setuptools import setup, find_packages

# get version number
with open('mapchete_xarray/__init__.py') as f:
    for line in f:
        if line.find("__version__") >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"')
            version = version.strip("'")
            continue

# use README for project long_description
with open('README.rst') as f:
    readme = f.read()


def parse_requirements(file):
    return sorted(set(
        line.partition('#')[0].strip()
        for line in open(os.path.join(os.path.dirname(__file__), file))
    ) - set(''))

setup(
    name='mapchete_xarray',
    version=version,
    description='Mapchete xarray output driver',
    long_description=readme,
    long_description_content_type='text/x-rst',
    author='Joachim Ungar',
    author_email='joachim.ungar@gmail.com',
    url='https://github.com/ungarj/mapchete_xarray',
    license='MIT',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    entry_points={
        'mapchete.formats.drivers': [
            'xarray=mapchete_xarray',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: GIS',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)

from setuptools import setup

setup(
    name='tf-parallel-map',
    version='0.1.0',
    author='István Sárándi',
    author_email='istvan.sarandi@uni-tuebingen.de',
    packages=['tf_parallel_map'],
    license='LICENSE',
    description='',
    python_requires='>=3.8',
    install_requires=['tensorflow', 'more-itertools', 'numpy'],
)

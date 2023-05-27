from setuptools import setup, find_packages

setup(
    name="circuit_toolkit",
    version="0.0",
    author='Your Name',
    author_email='binxu_wang@hms.harvard.edu',
    description='A short description of your project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/PonceLab/circuit_toolkit',
    packages=find_packages(),
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    install_requires=[
        # 'dependency1',
        # 'dependency2',
    ],
)
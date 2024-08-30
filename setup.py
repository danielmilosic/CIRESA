from setuptools import setup, find_packages

setup(
    name='CIRESA',
    version='0.0.0',
    description='CIR Evolution Statistics Architecture',
    author='Daniel Milosic',
    author_email='danielmilosic@live.com',
    url='https://github.com/danielmilosic/CIRESA',
    packages=find_packages(),
    install_requires=[
        'astropy'
        'sunpy'
        'pyspedas'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.6',
)

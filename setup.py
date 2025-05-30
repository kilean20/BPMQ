from setuptools import setup, find_packages

    
setup(name='BPMQ',
    version=0.0,
    description='BPMQ models and BAL for CS reconstruction',
    classifiers=[
    'Development Status :: 2 - Pre-Alpha',
    'Programming Language :: Python :: 3.6',
    'Topic :: FRIB beam tuning'
    ],
    keywords = ['BPMQ','machine-learning', 'optimization', 'FRIB beam tuning'],
    author='Kilean Hwang',
    author_email='hwang@frib.msu.edu',
#     license='',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'pytorch'
    ],
    zip_safe=False)

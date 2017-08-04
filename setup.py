from setuptools import setup, find_packages

setup(name             = 'NNFlow',
      version          = '0.6.0',
      description      = 'NNFlow framework to convert ROOT files to Tensorflow models',
      url              = 'https://github.com/kit-cn-cms/NNFlow',
      author           = 'KIT CN CMS team: Lukas Hilser, Marco A. Harrendorf, Maximilian Welsch, Martin Lang',
      author_email     = 'iekp-cn-analysis@lists.kit.edu',
      packages         = find_packages(),
      install_requires = ['numpy',
                          'scipy',
                          'matplotlib',
                          'scikit-learn>=0.18.2',
                          'pandas>=0.20.2',
                          'tables>=3.4.2'
                          ],
      zip_safe         = True)

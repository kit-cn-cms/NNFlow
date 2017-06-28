from setuptools import setup

setup(name='NNFlow',
      version='0.5.0',
      description='NNFlow framework to convert ROOT files to Tensorflow models',
      url='https://github.com/kit-cn-cms/NNFlow',
      author='KIT CN CMS team: Lukas Hilser, Marco A. Harrendorf, Maximilian Welsch',
      author_email='iekp-cn-analysis@lists.kit.edu',
      packages=['NNFlow'],
      install_requires=[
          'matplotlib',
          'root-numpy',
          'tensorflow-gpu>=1.1.0*',
          'queuelib',
          'scikit-learn>=0.18.2',
          'graphviz>=0.7.1'
      ],
zip_safe=True)

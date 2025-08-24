from setuptools import setup, find_packages

setup(
    name='EffiED-LLM',
    version='0.1.0',
    description='高效表格错误检测，基于大语言模型（LLMs）',
    author='Yifeng Zhao',
    author_email='yifeng.zhao2@student.uva.nl',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'transformers',
        'openai',
        'pandas',
        'numpy',
        'pyyaml'
    ],
    python_requires='>=3.10',
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)


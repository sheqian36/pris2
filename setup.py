from setuptools import setup, find_packages

setup(
    name='pris2',
    version='0.1',
    description='A simple image colorization testing package',
    # long_description=open('README.md').read(),  # 可选，如果有 README 文件
    long_description_content_type='text/markdown',  # 如果使用 Markdown 格式
    author='Huan Ouyang and Zheng Chang',
    url='https://github.com/sheqian/pris2.git',  # 如果有 GitHub 仓库链接
    packages=find_packages(),  # 自动发现所有包
    install_requires=[
        'torch==1.13.1',
        'torchvision==0.14.1',
        'numpy',
        'pillow',
        'timm==0.4.9'
    ],
    dependency_links=[
        'https://download.pytorch.org/whl/cu117',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',  # 根据你的环境选择合适的版本
)
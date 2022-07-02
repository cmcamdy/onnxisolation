
from setuptools import setup, find_packages
from os import path


pwd = path.abspath(path.dirname(__file__))
print(pwd)



with open("README.md", encoding="utf-8") as fp:
    long_description = fp.read()

with open("/home/qcraft/code/onnxisolation/requirements.txt", encoding="utf-8") as fp:
    install_requires = fp.read()


print(long_description)
print(install_requires)


setup(
    # 名称
    name='onnxisolation',

    # 版本号
    version='1.0.0',

    # 基本描述
    description='To isolate onnx.',

    # 项目的详细介绍，我这填充的是README.md的内容
    long_description=long_description,

    # README的格式，支持markdown，应该算是标准了
    long_description_content_type='text/markdown',

    # 项目的地址
    url='-',

    # 项目的作者
    author='chenhuan',

    # 作者的邮箱地址
    author_email='chenhuan21s@ict.ac.cn',

    # Classifiers，
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ],

    # 项目的关键字
    keywords='onnx isolate shape',

    # 打包时需要加入的模块，调用find_packages方法实现，简单方便
    packages=find_packages(exclude=['contrib', 'docs', 'tests', 'build', 'dist']),
    # packages=find_packages("onnxisolation"),
    
    # 项目的依赖库，读取的requirements.txt内容
    install_requires=install_requires,

    # 数据文件都写在了MANIFEST.in文件中
    include_package_data=True,

    # # entry_points 指明了工程的入口，在本项目中就是facialattendancerecord模块下的main.py中的main方法
    # # 我这是命令行工具，安装成功后就是执行的这个命令

    # entry_points={
    #     'console_scripts': [
    #         'FacialAttendanceRecord=facialattendancerecord.main:main',
    #     ],
    # },

)
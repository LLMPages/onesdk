from setuptools import setup, find_packages

setup(
    name="llm_onesdk",
    version="0.0.7",
    author="anycodes",
    author_email="liuyu@xmail.tech",
    description="OneSDK is a Python library that provides a unified interface for interacting with various Large Language Model (LLM) providers.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://onesdk.llmpages.cn/",
    packages=find_packages(exclude=["tests*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "requests>=2.25.1,<3",
        "python-dotenv>=0.19.0,<1",
        "typing-extensions>=3.7.4,<5",
    ],
    extras_require={
        "test": ["unittest2>=1.1.0"],
    },
    project_urls={
        "GitHub": "https://github.com/LLMPages/onesdk",
    },
    license="MIT",
    keywords="LLM API SDK NLP AI",
    include_package_data=True,
)

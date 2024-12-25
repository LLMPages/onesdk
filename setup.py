from setuptools import setup, find_packages

setup(
    name="llm_onesdk",
    version="0.0.3",
    author="anycodes",
    author_email="liuyu@xmail.tech",
    description="OneSDK is a Python library that provides a unified interface for interacting with various Large Language Model (LLM) providers.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://onesdk.llmpages.cn/",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "requests>=2.25.1",
        "python-dotenv>=0.19.0",
        "typing-extensions>=3.7.4",
        "unittest2>=1.1.0",
    ],
    project_urls={
        "GitHub": "https://github.com/LLMPages/onesdk",
    },
)
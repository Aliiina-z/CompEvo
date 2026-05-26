# File: setup.py

from setuptools import setup, find_packages

setup(
    name="llm_egt_forecaster",
    version="0.1.0",
    packages=find_packages(),
    author="[YY&YX/bigwind]", 
    description="An implementation of competition-driven evolution for LLM agents in time series forecasting.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/your-username/llm-egt-forecaster", # Replace with your repo URL
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.38.0",
        "peft>=0.9.0",
        "accelerate>=0.28.0",
        "bitsandbytes>=0.42.0",
        "numpy>=1.23.5,<2.0",
        "tqdm>=4.65.0",
        "sentence-transformers>=2.2.2",
        "openai>=1.0.0",
    ],
    python_requires='>=3.8',
)

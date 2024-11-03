from setuptools import setup, find_packages

# Read the long description from your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the LICENSE file
with open("LICENSE", "r", encoding="utf-8") as fh:
    license_text = fh.read()

# Function to read requirements.txt
def parse_requirements(filename):
    """Read dependencies from requirements.txt."""
    with open(filename, 'r') as file:
        return file.read().splitlines()

# Load requirements from requirements.txt
requirements = parse_requirements('requirements.txt')

setup(
    name="scGenAI",  
    version="1.0.0",  
    author="Ruijia Wang",  
    author_email="help.qbio@vorbio.com",  
    description="A package for single-cell gene NGS data prediction analysis with large language models (LLMs)",  # Short description
    long_description=long_description,  
    long_description_content_type="text/markdown",  
    url="https://bitbucket.org/vor-compbio/scgenai/",  
    packages=find_packages(),  # Automatically find packages in the directory
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Free for non-commercial use",  # Custom license description
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  # Minimum Python version requirement
    install_requires=requirements,  
    entry_points={
        "console_scripts": [
            "scgenai=scGenAI.cli:main",  # Command to run the CLI: 'scgenai'
        ]
    },
    zip_safe=False,
    license="License for non-commercial use",  # Specify the license name
    license_files=('LICENSE',),  # Include the LICENSE file
)

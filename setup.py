from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="image_encoder",
    version="0.1.0",
    author="Talon Bernard van Vuuren",
    description="A multi-task image encoder trained with PyTorch Lightning.",
    long_description=open("README.md").read() if "README.md" in os.listdir(".") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/image-encoder",  # Placeholder URL
    packages=find_packages(),
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

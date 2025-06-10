from setuptools import setup, find_packages

setup(
    name='dust3r',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
        # e.g., 'numpy', 'torch', 'scipy', etc.
    ],
    include_package_data=True,
    description='Description of your package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/dust3r',  # Update this to the repo URL
    author='Your Name',
    author_email='your.email@example.com',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Update the license as per your repo
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
from setuptools import setup, find_packages

setup(
    name='dreamdrive',
    version='0.0.0',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
        # e.g., 'numpy', 'torch', 'scipy', etc.
    ],
    include_package_data=True,
    description='DreamDrive for Image-to-4D Street View Scene Generation',
    long_description='README.md',
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername',  # Update this to the repo URL
    author='Jiageng Mao',
    author_email='maojiageng@gmail.com',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Update the license as per your repo
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
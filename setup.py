from distutils.core import setup
from os import path
import site

site_dir = site.getsitepackages()[0]


with open('requirements.txt', 'r') as f:
    requirements = list(map(str.strip, f))

if path.exists('README.md'):
    with open('README.md', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = None

setup_kwargs = dict(
    name='sk-torch',
    version='0.1dev',
    packages=['sktorch'],
    provides=['sktorch'],
    url='git@github.com:mattHawthorn/sk-torch.git',
    license='MIT license',
    author='Matt Hawthorn',
    maintainer='Matt Hawthorn',
    author_email='hawthorn.matthew@gmail.com',
    description='A wrapper around pytorch module objects with a sklearn-like interface, allowing boilerplate-free '
                'training of complex neural nets.',
    long_description=long_description,
    requires=requirements
)


if __name__ == "__main__":
    try:
        setup(**setup_kwargs)
    except Exception as e:
        print(e)
        print("Failed to execute setup()")
        exit(1)

    exit(0)

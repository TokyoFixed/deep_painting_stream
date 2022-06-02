# from setuptools import find_packages
# from setuptools import setup

# with open('requirements.txt') as f:
#     content = f.readlines()
# requirements = [x.strip() for x in content if 'git+' not in x]

# setup(name='deep_painting_stream',
#       version="1.0",
#       description="Project Description",
#       packages=find_packages(),
#       install_requires=requirements,
#       test_suite='tests',
#       # include_package_data: to install data from MANIFEST.in
#       include_package_data=True,
#       scripts=['scripts/deep_painting_stream-run'],
#       zip_safe=False)
mkdir -p ~/.streamlit

echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml

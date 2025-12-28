from setuptools import setup, find_packages

with open("LICENSE") as f:
    license_text = f.read()

setup(
    name="uisbp",
    version="0.1.0",
    description="Identify nerves in mobile ultrasound",
    author="Infinia ML",
    license=license_text,
    author_email="engineering@infiniaml.com",
    python_requires="==3.*",
    packages=find_packages(exclude=("tests", "tests.*", "docs", "script")),
    include_package_data=True,
    package_data={'uisbp': ['graphs/*/*']}
)

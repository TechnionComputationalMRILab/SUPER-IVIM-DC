from setuptools import setup, find_packages

setup(
    name="super-ivim-dc",
    version="1.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy==1.16.0",
        "matplotlib",
        "pandas",
        "SimpleITK",
        "torch==2.7.1",
        "tqdm",
        "joblib"
    ],
    entry_points={
        'console_scripts': [
            'super-ivim-dc = super_ivim_dc.main:main',
            'super-ivim-dc-train = super_ivim_dc.train:train_entry',
            'super-ivim-dc-infer = super_ivim_dc.infer:infer_entry',
        ],
    },
)

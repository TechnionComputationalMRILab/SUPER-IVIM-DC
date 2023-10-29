from setuptools import setup, find_packages

setup(
    name="super-ivim-dc",
    version="0.3.0",
    packages=find_packages(),
    install_requires=[
        "numpy==1.25.2",
        "scipy==1.11.1",
        "matplotlib==3.7.2",
        "pandas==2.0.3",
        "SimpleITK==2.3.0",
        "torch==2.0.1",
        "tqdm==4.65.0",
        "joblib==1.3.2"
    ],
    entry_points={
        'console_scripts': [
            'super-ivim-dc = super_ivim_dc.main:main',
            'super-ivim-dc-train = super_ivim_dc.train:train_entry',
            'super-ivim-dc-infer = super_ivim_dc.infer:infer_entry',
        ],
    },
)

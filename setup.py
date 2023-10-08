from setuptools import setup, find_packages

setup(
    name="super-ivim-dc",
    version="0.1",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'super-ivim-dc = super_ivim_dc.main:main',
            'super-ivim-dc-sim-infer = super_ivim_dc.infer:infer_entry',
            'super-ivim-dc-sim = super_ivim_dc.simulate:simulate_entry',
        ],
    },
)

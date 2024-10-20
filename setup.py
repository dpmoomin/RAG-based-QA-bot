from setuptools import setup, find_packages

setup(
    name='faq_chatbot',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'openai',
        'tqdm',
        'scikit-learn',
        'python-dotenv',
    ],
    entry_points={
        'console_scripts': [
            'faq-chatbot=main:main',
        ],
    },
    author='Your Name',
    author_email='your.email@example.com',
    description='A chatbot that answers questions based on FAQ data using OpenAI API',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/faq_chatbot',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)

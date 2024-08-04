# LocalPrompt
A simple AI powered Chat window that works with documents in a local directory.

Use it to process the rulebooks of a game you like, or the documentation of a project that you're working on. And then use AI to ask that knowledgebase some questions

## Setup
### To get the right PyTorch

```bash
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
replace the cu124 with the version you need

https://pytorch.org/get-started/locally/

### If you have an NVIDIA Graphics Card, then check for CUDA stuff
Install CUDA Toolkit:

* Download and install the CUDA Toolkit from the CUDA Downloads page.
* Download and install cuDNN from the cuDNN Download page.
* Ensure that your environment variables are set correctly to include the CUDA Toolkit and cuDNN directories. This is typically done during the installation process, but you can verify by checking the PATH, CUDA_HOME, and LD_LIBRARY_PATH (on Linux) or PATH (on Windows) environment variables.

### Install the other requirements

```bash
pip install -r requirements.txt
```

## Usage

1. Adjust settings in the 'config.ini' file to fit your needs
2. Use the 'build_knowledgebase.py' script to pre-process all the files in a specified directory
2. Start the session using 'chart.py'
3. Ask some questions!

## AI Models

When running Chat for the first time, you might be happier seeing the progress of the model download.

```bash
(venv): cd src
(venv): python chat.py
```
# LLM Eng

## Anaconda route:
Install Anaconda via [Conda Docs](https://www.anaconda.com/docs/getting-started/anaconda/install#macos-2).
Run after:
- Run `open ~/.zshrc` and add `export PATH=$PATH:$HOME/anaconda3/bin`.
- `cd` to the project dir.
- Run `conda env create -f environment.yml` to create the environment.
- Run `conda init && conda activate && conda activate llms` to kickstart the virtual environment.

## Alternative route (venv):
- Run `python3.11 -m venv llms`. Python 3.11 is the only compartible one at the moment (August 2025).
- Run `source llms/bin/activate` instead of `conda init && conda activate && conda activate llms`.
- Run `python3.11 -m pip install --upgrade pip` to upgrade pip.
- Run `pip install -r requirements.txt` to install requirements.
- Run `jupyter lab`.
- Run `cat > .env` to create an environments file and paste in your Open AI key `OPENAI_API_KEY=sk-proj-xyz`. Get the key from [OpenAI Org Billing page](https://platform.openai.com/settings/organization/billing/overview) - invest $5.
    It will look like this.
    ```sh
    OPENAI_API_KEY=sk-proj-xyz
    GOOGLE_API_KEY=xyz
    ANTHROPIC_API_KEY=xyz
    DEEPSEEK_API_KEY=xyz
    HF_TOKEN=xyz
    ```
- Run these:
    ```sh
    pip install selenium
    pip install playwright
    playwright install # to install browser binaries after installing playwright
    pip install nest_asyncio
    ```
- Run `xattr -d com.apple.quarantine drivers/chromedriver-mac-x64/chromedriver` for Mac/Apple to authorize chromedriver.

## Notes:
- Summarization is one of the most common AI use-cases.
- OpenAI is a frontier model
- The thre dimensions of LLM Engineering are Models, Tools and Techniques.
    - Models (open source, closed multi-modal, Architecture, Selecting).
    - Tools (HuggingFace, LangChain, Gradio, Weights, biases, Modal).
    - Techniques (APIs, Multip-shot prompting, RAG, Fine-tunig, Agentization).
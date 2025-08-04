# LLM Eng
Notes and implementations based on learnings in [Udemy LLM Course](https://www.udemy.com/course/llm-engineering-master-ai-and-large-language-models).

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

## Ollama
- Visit [ollama.com](ollama.com) and install! Follow http://localhost:11434/ to find a message `"Ollama is running"`.
- Otherwise, run `ollama serve`.
- In another Terminal, run `ollama pull llama3.2`.
- Then revisit http://localhost:11434/.
- If Ollama is slow on your machine, try `MODEL = "llama3.2:1b"`.
- To make sure the model is loaded, run `!ollama pull llama3.2`.


## Notes:
- Summarization is one of the most common AI use-cases.
- OpenAI is a frontier model
- Perplexity is a search engine powered by AI.
- The thre dimensions of LLM Engineering are Models, Tools and Techniques.
    - Models
        - open source e.g. Llama, Mixtral, Qwen, Gemma, Phi
        - closed e.g. GPT, Claude, Gemini, Command R+, Perplexity paid modals
        - multi-modal
        - Architecture
        - Selecting
    - Tools (HuggingFace, LangChain, Gradio, Weights, biases, Modal).
    - Techniques (APIs, Multip-shot prompting, RAG, Fine-tunig, Agentization).
- Three ways to use models:
    - Chat interface e.g. Chat GPT
    - Cloud APIs e.g.
        - LLM API like OpenAI
        - Framework like LangChain (which ustilize multiple AIs under the hood)
        - Managed AI cloud services e.g. Amazn Bedrock, Google Vertex, Azure ML
    - Direct Interface e.g.
        - HuggingFace Transformers Library
        - Ollama running locally e.g. for confidential data that must not go to some cloud server
- Limitations of Frontier Models:
    - They are not great at specialized domains e.g. most are not at PhD level but are getting very close.
    - They're not great at recent events because of the date limit on the data they're trained on.
    - They confidently make mistakes i.e. giving wrong answers with blindspots.
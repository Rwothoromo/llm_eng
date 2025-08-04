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


## Notes - Understanding LLMs and Their Applications:
A breakdown of key concepts and developments in the world of Large Language Models (LLMs):

### Core AI Use Cases

* **Summarization** is one of the most common and practical applications of AI.
* **Perplexity** is an AI-powered search engine, demonstrating how AI is enhancing information retrieval.
* **OpenAI** stands out as a developer of frontier models, pushing the boundaries of AI capabilities.

---

### The Three Dimensions of LLM Engineering

Effective LLM development and deployment revolve around three main pillars:

1.  **Models:**
    * **Open Source:** Accessible and customizable models like **Llama, Mixtral, Qwen, Gemma,** and **Phi**.
    * **Closed Source:** Proprietary models such as **GPT, Claude, Gemini, Command R+,** and paid versions of **Perplexity**.
    * **Multi-modal:** Models capable of processing and generating various types of data (e.g., text, images).
    * **Architecture:** The underlying design of the model (e.g., Transformer).
    * **Selection:** Choosing the right model for a specific task based on its strengths and limitations.

2.  **Tools:**
    * **Hugging Face:** A hub for pre-trained models and datasets.
    * **LangChain:** A framework for developing applications powered by language models.
    * **Gradio:** For building interactive web demos of machine learning models.
    * **Weights & Biases:** For tracking and visualizing machine learning experiments.
    * **Modal:** For deploying AI models at scale.

3.  **Techniques:**
    * **APIs (Application Programming Interfaces):** For programmatic access to models.
    * **Multi-shot Prompting:** Providing multiple examples in a prompt to guide the model's response.
    * **RAG (Retrieval-Augmented Generation):** Combining information retrieval with text generation to improve accuracy and relevance.
    * **Fine-tuning:** Adapting a pre-trained model to a specific task or dataset.
    * **Agentization:** Giving LLMs the ability to plan and execute multi-step tasks.

---

### Ways to Interact with LLMs

There are three primary methods for utilizing LLMs:

1.  **Chat Interfaces:**
    * Direct conversational interaction, exemplified by **ChatGPT**.

2.  **Cloud APIs:**
    * **LLM APIs:** Direct access to models, such as **OpenAI's API**.
    * **Frameworks:** Libraries like **LangChain** that orchestrate multiple AI services.
    * **Managed AI Cloud Services:** Platforms like **Amazon Bedrock, Google Vertex AI,** and **Azure ML** offering integrated LLM solutions.

3.  **Direct Interfaces:**
    * **Hugging Face Transformers Library:** For local development and fine-tuning.
    * **Ollama:** Running models locally, ideal for sensitive or confidential data that shouldn't leave your environment.

---

### Limitations of Frontier Models

Despite their advancements, current frontier models have some limitations:

* **Specialized Domains:** They may not perform at a PhD level in highly niche fields, though they are rapidly improving.
* **Knowledge Cut-off:** Their training data has a specific date limit, meaning they lack knowledge of very recent events.
* **Confident Mistakes (Hallucinations):** They can sometimes generate incorrect information with high confidence due to inherent blind spots.

---

### Key Milestones and Trends

The LLM landscape is evolving rapidly:

* **2017:** Google scientists introduced the groundbreaking **Transformer Architecture**, a foundational element for many modern LLMs.
* **Evolution of Roles:** The "Prompt Engineer" role has seen a significant shift; prompting is now more intuitive and accessible. The pay and hiring hype dipped.
* **Customization and Ecosystems:** The emergence of **Custom GPTs** and the **GPT Store** allows for personalized AI applications.
* **Integrated AI Assistants:** The rise of **CoPilots** like **Microsoft Copilot** and **GitHub Copilot**, which integrate AI directly into workflows.
* **Advanced Agentization:** Further development in AI agents, such as **GitHub Copilot Workspace**, where LLMs can be assigned complex tasks and even take on planning roles.

---
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


# A Guide to Large Language Models (LLMs)

This guide provides a foundational understanding of Large Language Models (LLMs), covering how they work, the tools and techniques used to interact with them, and a look at key industry trends.

---

### Core Concepts

#### What is an LLM?

An LLM is a type of AI that processes and generates human-like text. The size of an LLM is typically measured by its number of **parameters**, which are internal values the model uses for predictions. More parameters generally indicate a more capable and complex model.

- **GPT-1** (2018) had 117 million parameters.
- The largest version of **Llama 3.1** has 405 billion parameters.
- **GPT-4** is estimated to have over 1 trillion parameters.

#### How Do Tokens and Context Windows Work?

LLMs process text by first breaking it down into smaller units called **tokens**. Tokens are the fundamental building blocks of language for an AI.

- A single token is roughly equal to **4 characters** or about **three-quarters of a word**.
- **1,000 tokens** typically equate to around **750 words**.
- The **context window** is the maximum number of tokens an LLM can process in a single prompt, which determines how much information the model can "remember".

### The Three Pillars of LLM Engineering

Effective LLM development and deployment are built on three main pillars: Models, Tools, and Techniques.

1.  **Models:** The foundation of any AI application.
    - **Open Source:** Models like **Llama**, **Mixtral**, and **Gemma** are accessible and customizable.
    - **Closed Source:** Proprietary models such as **GPT**, **[Claude](https://claude.ai)**, and **[Gemini](https://gemini.google.com)**.
    - **Multi-modal:** Models that can process and generate various data types, including text, images, and audio.
2.  **Tools:** Software and frameworks for building applications.
    - **Hugging Face:** A central hub for pre-trained models and datasets.
    - **LangChain** & **LlamaIndex:** Frameworks for building complex applications.
    - **Weights & Biases:** A platform for tracking and visualizing machine learning experiments.
3.  **Techniques:** Methods for getting the most out of an LLM.
    - **Prompting:** The art of crafting effective instructions, including **Zero-shot** (no examples), **One-shot** (one example), and **Multi-shot** (multiple examples).
    - **Fine-tuning:** Adapting a pre-trained model to a specific task or dataset.
    - **RAG (Retrieval-Augmented Generation):** A technique that combines information retrieval with text generation to improve accuracy.
    - **Agentization:** Giving LLMs the ability to plan and execute multi-step tasks.

### Key Milestones and Trends

- **The Transformer Architecture (2017):** This foundational architecture from Google is the basis for most modern LLMs.
    - The Transformer architecture is a type of neural network that processes entire sequences of data, such as a sentence, in parallel, which makes training significantly faster. This is a key difference from older models like recurrent neural networks (RNNs), which processed data sequentially. The core innovation of the transformer model is its **self-attention** mechanism, which allows it to weigh the importance of different words in a sentence relative to each other, regardless of their position. This mechanism helps the model understand the context and relationships between words in a sentence.
    - **Encoder-Decoder Architecture**: The original transformer model has an **encoder**, which processes the input text to create a numerical representation of its meaning and context, and a **decoder**, which uses this representation to generate the output text.
    - **Self-Attention Mechanism**: This is the core component that allows transformers to consider the importance of every other word in a sentence when processing a single word. It calculates attention scores to determine how much it should "pay attention" to other words in the sequence.
    - **Multi-Head Attention**: This is an extension of the self-attention mechanism that runs several attention functions in parallel. Each "head" focuses on different aspects of the relationships between words, allowing for a richer understanding of the input.
    - **Positional Encoding**: Since transformers process words simultaneously, they need a way to understand the order of words. Positional encoding adds a numerical value to each word's representation to signify its position in the sequence, which helps the model understand the structure of the language.
    - **Tokenization and Embeddings**: Before processing text, the model converts words or sub-words (**tokens**) into numerical representations called **embeddings**. These high-dimensional vectors capture the semantic meaning of the words.
    - **Integrated AI Assistants:** The rise of **CoPilots** like **Microsoft Copilot** and **GitHub Copilot** has embedded AI directly into workflows.

### Limitations of Frontier Models

Even the most advanced LLMs have limitations:

- **Knowledge Cut-off:** Their training data has a specific date limit, so they lack knowledge of recent events.
- **Confident Mistakes (Hallucinations):** They can sometimes generate incorrect information with high confidence.
- **Specialized Domains:** They may not have a deep understanding of highly niche or specialized fields.

### Key Terms

- **RAG (Retrieval-Augmented Generation):** A technique that improves a chatbot's response by retrieving relevant external information and adding it to the prompt.
- **Tools:** External functions or APIs that an LLM can use to perform specific tasks, such as fetching data, performing actions, or handling calculations.
- **Agents:** Autonomous software entities that can reason, plan, and execute tasks to achieve a goal. They use various tools to break down and solve complex problems with limited human oversight.

### HuggingFace Platform
- The Ubiquitous platform for LLM Engineers (https://huggingface.co/).
- **Models:** Has over 1.9M Open Source models
- **Datasets:** Has over 200k datasets
- **Spaces:** Many apps on HuggingFace cloud/platform (most built with Gradio). Also StreamLit, Leaderboards.
- **HuggingFace Libraries:**
    - hub
    - datasets
    - transformers
    - peft (parameter, efficient fine tuning)
    - trl (transformer reinforcement learning)
    - accelerate (allows tranformers to run on distributed set-ups)
- There are two API levels of HuggingFace
    - Pipelines (Higher level APIs to carry out standard tasks incredibly fast).
        - **Sentiment analysis.** Checking for the emotion conveyed in agiven sentence.
        - **Classifier/Classification.** Putting things into buckets.
        - **Named Entity Recognition.** Take words in a sentence and tag them as things like people, locations, etc.
        - **Question Answering.** You have some context and need to ask questions based on that.
        - **Summarizing/Summarization.** You provide a block of text to be turned into a summary.
        - **Translation.** Translate between one language and another.
        - Use pipelines to generate Text, Images, and Audio.
    - Tokenizers and Models (Lower level APIs to provide the most power and control).

### Google Colaab
- Run Jupyter Notebook on [Google Colab](https://colab.research.google.com/), to share and collaborate with colleagues.
- Different runtimes availabe; CPU-based, Lower spec GPU, Higher spec GPU for resource-intensive stuff.


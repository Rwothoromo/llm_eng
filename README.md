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

### The Tokenizer
- An object that translates/maps between text and tokens with encode() (strings to tokens) and decode() (tokens to strings) methods, for a particular model.
- Contains a Vocab (all the different fragments of characters that make up a token, also a special token that tells the model something e.g. start of a sentence).
- Has a Chat Template that can take a set of mesagges and makes them a set of tokens.

#### Tokenizers for Key Models
- Llama 3.1 (Meta).
- Phi 3 (Microsoft).
- Qwen2 (Alibaba Cloud).
- Starcoder2 (Coding model). Built in collaboration by HuggingFace, ServiceNow and Nvidia.

### Quantization
- This allows us to load a model into memory and use less memory when running things.

### Compare LLMs

#### Basics
- Open-source or closed.
- Release date and knowledge cut-off (last date of training data / limit of knowledge on current events).
- N. of Parameters (strength of model, costs and how much training data is needed to fine tune the model).
- Training tokens (tokens used during training / size of training dataset).
- Context length (size of context window / how many tokens are kept in memory when predictting the next token e.g. chat history)

#### Other Basics
- **Inference Costs:** The expense each time the model generates output in production.
    - For frontier models, API costs depend on input and output token counts.
    - For open source models run locally or on cloud platforms like Colab or Modal, runtime compute costs apply.
- **Training Costs:** If you fine-tune an open source model, training costs must be factored in. Out-of-the-box frontier models typically have no training costs unless fine-tuned.
- **Build Costs** and **Time to Market**: The **effort** and **duration** required to develop a solution. Frontier models often have low build costs and fast time to market, while fine-tuning open source models usually takes longer and is more complex.
- **Rate Limits and Reliability:** Frontier models accessed via APIs may have rate limits and occasional stability issues, such as overload errors.
- **Speed** and **Latency**: Throughput speed (how fast tokens are generated) and latency (time to first token response) are important, especially in interactive applications like chatbots or multimodal assistants.
- **Licensing**: Be aware of license restrictions for both open and closed source models, including commercial use limitations and terms of service agreements.

### The Chinchilla Scaling Law
Thhe number of parameters is approimately/roughly proportional to the number/size of training tokens.

### Benchmarks
Tests applied and used to rank LLMs. See the name and what gets evaluated.
- **ARC**: Scientific Reasoning (multi-choice questions)
- **DROP**: Language Comparison
- **HellaSwag**: Common Sense
- **MMLU**: Understanding - Massive Multitask Language Understanding.
- **TruthfulQA**: Accuracy
- **Winogrande**: Context (does the LLM understand the context and resolve ambiguity?).
- **GSM8K**: Math (Grade Scoring Math for elementary and middle school).
- **ELO**: Chat (eval chat between models e.g. face-offs, zero-sum games, 1 loser and 1 winner).
- **HumanEval**: Python Coding test (generate code based on DocStrings).
- **MultiPL-E**: Broader Coding (Translate **HumanEval** to 18 languages).

### Limitations of Benchmarks
Benchmarks are useful for comparing where different models excel and where they do not intend to be used.
- Benchmarks are **not consistently applied**. Depending on the source, especially if it is a company press release, the methodology, hardware used, and other factors can vary. There is no gold standard that standardizes these measurements, so all results should be taken with a pinch of salt.
- Benchmarks can be **too narrow in scope**, particularly when involving multiple choice style questions. It is difficult to measure nuanced reasoning with such formats.
- **Training data leakage** is a significant problem. It is challenging to ensure that answers are not present in the training data, especially as models are trained on increasingly recent data that may include information about these benchmarks.
- **Overfitting** is a common issue from traditional data science. Models may perform exceptionally well on benchmarks because of extensive hyperparameter tuning and repeated testing, effectively solving the benchmark rather than the underlying task. This can lead to poor performance on out-of-sample questions that test the same skills but are phrased differently.
- **New Speculative Problem:** Frontier LLMs may be aware that they're being evaluated.

### Hard, Next-Level Benchmarks
- **GPQA**: Google-proof Q&A (resistant to Google search findings) Graduate Tests (448 expert questions in Physics, Chemistry and Biology for PhD level. Regular non-PhDs score 34% even with web access). PhD people score 65% on average. Claude 3.5 sonnet is the best currently (59.4% score).
- **BBHard**: Big-Bench Hard. 204 tasks that **used to be** beyond the LLMs' capabilities.
- **Math Lv 5**: Hardest tear of Math questions for high school competitions
- **IFEval**: Difficult instructions.
- **MuSR**: Multistep Soft Reasoning. For logical deductions.
- **MMLU PRO:** Harder **MMLU**. Massive Multitask Language Understanding - Professional has 10 multi-choice questions.

## LLM Leaderboards
- HuggingFace Open LLM leaderboad (https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard).
- HuggingFace BigCode
- HuggingFace LLM Perf
- HuggingFace Others e.g. Medical, Human Language-specific
- Vellum (has API costs and context windows comparisons)
- SEAL (expert level)

Check for different leaderboards to help you select the best open-source model for your usecase
- HuggingFace has https://huggingface.co/spaces?search=leaderboard
- Vellum has https://vellum.ai/llm-leaderboard
- Scale has SEAL leaderboads https://scale.com/leaderboard
- LMSYS Chatbot Arena https://lmarena.ai/leaderboard (Assess models by Humans)

### Some AI usecases
- harvey.ai for Law stuff
- nebula.io for talent hiring
- bloop.ai for porting legacy codebases into Java e.g. Old COBOL code.
- Einstein Copilot for Health by Salesforce (https://www.salesforce.com/agentforce/einstein-copilot/)

### Evaluate Performance of Gen AI solutions
- Model-centric or Technical Metrics (Easiest to optimize with)
    - Loss (how poorly an LLM has performed in its task) e.g. cross-entropy loss
    - Perplexity (e to-the-power-of cross-entropy loss).
    - Accuracy
    - Precision, Recall, F1
    - AUC-ROC
- Business-centric or Outcome Metrics (Most tangible impact)
    - KPIs tied to business objectives
    - ROI
    - Improvements in time, cost or resources
    - Customer satisfaction
    - Benchmark comparisons

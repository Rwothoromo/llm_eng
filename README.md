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

Of course\! Here is an improved version of your Markdown content. It has been restructured for better flow, clarity, and readability, with enhanced formatting and more cohesive explanations.

-----

# A Comprehensive Guide to Large Language Models (LLMs)

This guide offers a foundational understanding of Large Language Models (LLMs), from their core technical concepts to the tools, techniques, and trends shaping the industry.

-----

## Part 1: Core Concepts

### What is a Large Language Model?

A Large Language Model (LLM) is a sophisticated type of artificial intelligence (AI) designed to understand, process, and generate human-like text. The "large" in its name refers to the immense scale of both its architecture and the data it's trained on. The capability of an LLM is often measured by its number of **parameters**—internal variables the model learns during training and uses to make predictions. Generally, a higher parameter count indicates a more powerful and complex model.

The growth in model size has been exponential:

  - **GPT-1 (2018):** 117 million parameters
  - **Llama 3.1 (405B version):** 405 billion parameters
  - **GPT-4 (estimate):** Over 1 trillion parameters

### How LLMs "Read": Tokens and Context Windows

LLMs don't process text as whole words or sentences. Instead, they break it down into smaller units called **tokens**.

  - **Tokens:** The fundamental building blocks of language for an AI.
  - **Rule of Thumb:** 1 token is roughly 4 characters or ¾ of a word. This means 1,000 tokens equate to approximately 750 words.
  - **The Tokenizer:** An essential component that maps text to a sequence of tokens (`encode`) and tokens back to text (`decode`). Each model has its own tokenizer and a specific **vocabulary** (the set of all possible tokens it understands).

The **context window** is the maximum number of tokens an LLM can process at once. This includes both the input prompt and the generated output. The size of the context window dictates how much information the model can "remember" in a single conversation or task.

### The Engine: The Transformer Architecture

The foundation for nearly all modern LLMs is the **Transformer architecture**, introduced by Google in 2017. Its key innovation was processing entire sequences of data in parallel, which dramatically accelerated training compared to older, sequential models like Recurrent Neural Networks (RNNs).

Key components of the Transformer include:

  - **Self-Attention Mechanism:** This is the core concept. It allows the model to weigh the importance of different tokens within the input text relative to each other, regardless of their position. This is how LLMs capture context and understand complex relationships in language.
  - **Encoder-Decoder Structure:** The original design includes an **encoder** to create a rich numerical representation of the input and a **decoder** to generate the text output from that representation.
  - **Multi-Head Attention:** An enhancement where multiple self-attention mechanisms run in parallel, each focusing on different aspects of the token relationships, creating a more nuanced understanding.
  - **Positional Encoding:** Since tokens are processed in parallel, the model needs a way to understand word order. Positional encodings are added to the token representations (embeddings) to provide this crucial positional information.

-----

## Part 2: The LLM Engineering Ecosystem

Developing effective LLM applications relies on three pillars: the models themselves, the tools to build with them, and the techniques to optimize their performance.

### Pillar 1: Models

The model is the foundational engine of any AI application.

  - **Open-Source Models:** Freely available for modification and self-hosting. Examples include Meta's **Llama** family, Mistral's **Mixtral**, and Google's **Gemma**.
  - **Closed-Source Models:** Proprietary models accessed via APIs. Examples include OpenAI's **GPT** series, Anthropic's **[Claude](https://claude.ai)**, and **[Google's Gemini](https://gemini.google.com)**.
  - **Multimodal Models:** Capable of processing and generating information across different data types, such as text, images, audio, and video.

### Pillar 2: Tools & Platforms

These are the frameworks and infrastructure used to build, train, and deploy LLMs.

  - **[Hugging Face](https://huggingface.co/):** The central hub for the open-source AI community, offering:
      - **Model Hub:** Over 1.9 million pre-trained models.
      - **Datasets:** Over 200,000 datasets for training and evaluation.
      - **Spaces:** A platform for hosting and sharing AI demos.
      - **Libraries:** Essential tools like `transformers`, `datasets`, `peft` (Parameter-Efficient Fine-Tuning), and `accelerate`.
  - **Development Frameworks:**
      - **LangChain & LlamaIndex:** Help build complex, data-aware applications by chaining LLM calls with other data sources and APIs.
  - **Experiment Tracking:**
      - **Weights & Biases:** A platform for visualizing and tracking machine learning experiments.
  - **Development Environments:**
      - **[Google Colab](https://colab.research.google.com/):** A cloud-based Jupyter Notebook environment that provides free access to GPUs, making it ideal for experimenting and training models.

### Pillar 3: Techniques

These are methods for eliciting the best possible performance from an LLM.

  - **Prompting:** The art of crafting effective instructions for the model.
      - **Zero-shot:** The model responds to a task without any examples.
      - **One-shot / Few-shot:** The prompt includes one or more examples to guide the model's response.
  - **Retrieval-Augmented Generation (RAG):** Enhances model accuracy by retrieving relevant information from an external knowledge base (like a company's internal documents) and adding it to the prompt as context. This helps reduce hallucinations and keeps information current.
  - **Fine-Tuning:** Adapting a pre-trained model to a specific task or domain by training it further on a smaller, specialized dataset.
  - **Quantization:** A compression technique that reduces the memory footprint of a model, allowing it to run on less powerful hardware.
  - **Agentization:** Building autonomous agents that can use LLMs to reason, plan, and execute multi-step tasks by interacting with external tools (e.g., APIs, databases).

-----

## Part 3: Evaluating and Selecting an LLM

Choosing the right model requires a careful analysis of its capabilities, costs, and performance on relevant benchmarks.

### Key Comparison Criteria

**Model Basics:**

  - **Source:** Open-source or closed-source?
  - **Parameters:** How large is the model? This impacts capability and fine-tuning costs.
  - **Training Data:** What was the size (in tokens) and knowledge cut-off date of its training corpus?
  - **Context Length:** What is the maximum size of the context window?

**Operational Factors:**

  - **Costs (Inference & Training):** Consider API costs for proprietary models versus compute costs for self-hosted open-source models. Fine-tuning introduces additional training costs.
  - **Speed & Latency:** How many tokens per second can it generate (throughput), and how long does it take to get the first token (latency)?
  - **Rate Limits & Reliability:** API-based models often have usage limits and can experience downtime.
  - **Licensing:** Always check the license for commercial use restrictions, especially for open-source models.

### The Chinchilla Scaling Law

A key finding from DeepMind suggests that for optimal performance, the model size (number of parameters) and the training data size (number of tokens) should be scaled proportionally. A model can be undertrained if it's too large for its dataset.

### Benchmarking LLM Performance

Benchmarks are standardized tests used to rank and compare LLMs across various capabilities.

  - **MMLU:** Massive Multitask Language Understanding.
  - **GSM8K:** Grade-school math word problems.
  - **HumanEval:** Python code generation.
  - **ARC:** Scientific reasoning challenges.
  - **HellaSwag:** Common-sense reasoning.
  - **TruthfulQA:** Measures a model's tendency to generate factual, non-hallucinated answers.
  - **ELO Rating:** A system (often used in leaderboards like the Chatbot Arena) where human raters compare model outputs side-by-side.

#### Limitations of Benchmarks

  - **Inconsistent Application:** Different providers may use slightly different methodologies.
  - **Data Contamination:** Benchmark questions may have leaked into the models' training data, inflating scores.
  - **Overfitting:** Models can be "tuned" to perform well on specific benchmarks without generalizing the underlying skill.
  - **Narrow Scope:** Multiple-choice questions often fail to capture nuanced reasoning abilities.

#### Advanced Benchmarks

New, harder tests are emerging to push the limits of frontier models:

  - **GPQA:** Graduate-level questions designed to be "Google-proof."
  - **MMLU-PRO:** A more challenging version of the original MMLU.
  - **IFEval:** Tests a model's ability to follow complex instructions.

### LLM Leaderboards

These platforms aggregate benchmark results to help with model selection:

  - **Hugging Face:** [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) and others for specific tasks.
  - **LMSYS:** [Chatbot Arena Leaderboard](https://lmarena.ai/leaderboard) based on human preferences.
  - **Vellum:** [LLM Leaderboard](https://vellum.ai/llm-leaderboard) with comparisons of cost and context windows.
  - **Scale:** [SEAL Leaderboards](https://scale.com/leaderboard) for expert-level evaluation.

-----

## Part 4: Applications and Inherent Limitations

### Real-World Use Cases

LLMs are being integrated into specialized professional tools across industries:

  - **Law:** [Harvey.ai](https://www.harvey.ai/) provides AI assistance for legal work.
  - **Hiring:** [Nebula.io](http://nebula.io/) assists in talent acquisition.
  - **Software Engineering:** [Bloop.ai](https://bloop.ai/) helps understand and migrate legacy codebases.
  - **CRM:** Salesforce's [Einstein Copilot](https://www.google.com/search?q=https://www.salesforce.com/products/einstein/einstein-copilot/) integrates generative AI into customer relationship management.

### Measuring Success

The performance of a GenAI solution can be evaluated from two perspectives:

1.  **Model-Centric (Technical) Metrics:** Easier to optimize directly.
      - **Loss / Perplexity:** Measures how "surprised" a model is by the correct answer.
      - **Accuracy, Precision, Recall, F1:** Standard classification metrics.
2.  **Business-Centric (Outcome) Metrics:** Measure the tangible impact.
      - **ROI:** Return on investment.
      - **KPIs:** Improvements in time, cost, customer satisfaction, or other key business indicators.

### Limitations of Frontier Models

Even the most advanced LLMs have fundamental weaknesses:

  - **Knowledge Cut-off:** Their knowledge is frozen at the end of their training date, so they are unaware of recent events.
  - **Hallucinations:** They can generate plausible but incorrect or nonsensical information with high confidence.
  - **Niche Domains:** They may lack the deep, specialized expertise required for highly technical or niche fields.


### LangChain
- LangChain is a framework created in late 2022 to simplify building LLM applications by chaining functionalities.
- It standardizes and simplifies retrieval-augmented generation (RAG) workflows, enabling quick time to market.
- LangChain provides wrappers around common LLM APIs, allowing easy switching between models like OpenAI and Claude.
- Despite decreasing need due to maturing APIs, LangChain remains a valuable tool for loading, chunking, and vectorizing knowledge bases efficiently.
- LangChain significantly simplifies creating applications for common tasks like assistance and retrieval-augmented generation (RAG).

### Document structuring strategies for creating a knowledge base for RAG
- Document/Text splitting with LangChain (https://python.langchain.com/docs/concepts/text_splitters/).
- How partitioning a PDF works (https://docs.unstructured.io/api-reference/partition/partitioning).
- How Chunking works (https://docs.unstructured.io/api-reference/partition/chunking).
- Unstructured is an open-source tool for converting documents to structured data effortlessly (https://github.com/Unstructured-IO/unstructured).

### Chroma
A lovely datastore for keeping vectors. We can visualize the data thereafter.
For example:
```python
# Use OpenAI Embeddings model for text representation.
embeddings = OpenAIEmbeddings()

# Create Chroma Vector Store
vectorstore = Chroma.from_documents(
  documents=chunks, # populate the vectorstore with chunks (likely text segments)
  embedding-embeddings, # the embeddings model
  persist_directory=db_name # Persist the Vector Store to a local directory for later use
)
```

### Key Abstractions in LangChain
LangChain defines several abstractions that simplify building applications:
- **LLM:** Represents a language model, such as OpenAI's GPT. This abstraction encapsulates the model interface.
- **Retriever:** An interface to a vector store, such as Chroma, used for retrieval in RAG workflows. It enriches prompts by retrieving relevant documents.
- **Memory:** Represents the history of a conversation with a chatbot. It abstracts the underlying data structure, typically a list of messages, managing context across interactions.

For example:
```python
# create a new Chat with OpenAI with a temperature setting of 0.7, influencing the creativity of the responses
llm = ChatOpenAI (temperature=0.7)

# set up the conversation memory for the chat - configured to store chat history, allowing the AI to maintain context throughout the conversation
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# create a retriever from the Chroma datastore, enabling the retrieval of relevant information for generating responses
retriever = vectorstore.as_retriever()

# putting it together - The LLM, retriever, and memory are combined to form a ConversationalRetrievalChain, which facilitates interactive conversations with the AI, leveraging both its language generation capabilities and retrieved information.
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
```

For migrating DB, see https://python.langchain.com/docs/versions/migrating_memory/

### LangChain Expresiol Langiuage (LCEL)
- LCEL is a declarative language (using YML) that can be used as an alternative to the code approach.
- LCEL allows declarative setup of chains using YAML files.
- LCEL closely maps to Python code for defining models, memory, embeddings, vector stores, retrievers, and chains.

### WEEK 5 CHALLENGES (Consider doing when you get time)
- Major challenge for you - your own private Knowledge Worker
- Create a Knowledge Worker on your information to boost productivity
- Assemble all your files in 1 place; your personal Knowledge Base
- Vectorize everything in Chroma - your vector datastore
- Build a Conversational Al and ask questions!
- Advanced ideas to take it to the next level
- If you use Google Workspace, use Google's API to read your own docs
- If you use MS Office, use libraries to read Office docs
- Harder - use libraries to connect to your email inbox, and Slack, and more!

### Training (Transfer Learning)
- Take a pre-trained multi-billion prameter model, and use aditional training data to fine-tune it to do our task.
- Training such a model from scratch would otherwise be expensive (10s to 100s of millions of dollars $).

### Sources of Datasets:
- Personal or Company Proprietary Data
- Kaggle
- HuggingFace
- Synthetic data (faked or LLM-generated data)
- Specialist Companies that curate or craft data for you e.g. scale.com

### Digging into data
1. Investigate - understand the data, its quality et al
2. Parse - into a structure easier to handle
3. Visualize - see how spread out the values are
4. Assess data quality - understand the limitations
5. Curate - decide how to craft the dataset e.g. exclude the poor quality dataset
6. Save - upload to the HuggingFace Hub

### Five-Step Strategy for Applying Models to Commercial Problems
A five-step strategy guides the deployment of LLM solutions from business problem to production.
1. **Step 1 - Understanding:** Gather business requirements and evaluate how success will be measured.
- Data science metrics and how users and business sponsors will judge the project's achievement.
- Understand the ultimate business metrics, even if you do not have immediate influence over them.
- Analyze the data: quantity, quality, and format (structured, unstructured, or mixed). This understanding will influence your model choice and approach.
- Determine non-functional requirements such as budget, scalability, latency (response time requirements), and time to market. For example, if a rapid build is required, using a frontier model via API and a user interface like Gradio can enable deployment within minutes.
2. **Step 2 - Preparation:**
- Research existing solutions that address the problem today, including non-data science approaches such as rule-based systems.
- Explore traditional data science models, for example, linear regression for price prediction.
- Establish baselines even if you are confident that an LLM will outperform existing solutions. This baseline provides a starting point and quantifies improvements from new models.
- Compare relevant LLMs considering price, context length, licensing constraints, and benchmark performance using leaderboards and specialist scores.
- Curate your data by scrubbing and pre-processing it. Split your data into training, validation, and test sets. Use validation data to tune hyperparameters and test data to evaluate final model success.
3. **Step 3 - Selecting:**
- Choose LLMs based on the criteria established during preparation. Experiment with different models, train and validate them using your curated data. This phase is forthcoming and will be exciting to undertake.
4. **Step 4 - Optimizing Models to Solve Commercial Problems:**
- Going beyond the pre-trained or existing frontier models to extract more value from them.
- There are three main approaches to optimization: `prompting`, `retrieval-augmented generation (RAG)`, and `fine-tuning` (training the model further to improve its performance).
  - Prompting and RAG are inference-time techniques, meaning they work with a trained model during inference to enhance results.
  - Fine-tuning, however, is a training-time technique where the model's weights are adjusted with additional data to improve its problem-solving capabilities.
5. **Step 4 - Productionizing AI Models:**
- Involves defining APIs, hosting, deployment, monitoring, security, scaling, and continuous improvement.
- Define an API that can be used to call our model or code. This API might either use an open-source model or make a call to a frontier model with various scaffolding around it, such as retrieval-augmented generation (RAG) or prompt engineering.
- Decide how to host and deploy our model and API.
  - We address considerations such as monitoring, information security concerns, and how the system will scale.
  - We then measure the business metrics identified in step one and continuously retrain, measure performance, and improve our model on an ongoing basis.
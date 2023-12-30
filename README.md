# Curated-List-of-Generative-AI-Tools
This repo contains the curated list of tools for generative AI

# Rise of AI

![1686648543214](https://github.com/ParthaPRay/Curated-List-of-Generative-AI-Tools/assets/1689639/694a63ef-fb30-4515-b206-67007b2560b1)


![1686648771444](https://github.com/ParthaPRay/Curated-List-of-Generative-AI-Tools/assets/1689639/78b8060b-849c-48bf-80a4-1b90b8e70637)



# Innovations to be Fueled by Generative AI

Generative AI is revolutionizing various sectors, offering a wide array of innovations and capabilities. Let's delve into each of the critical technologies you mentioned:

* **Artificial General Intelligence (AGI):** This refers to a machine's ability to understand, learn, and apply intellectual skills at a level equal to or surpassing human intelligence. AGI remains a theoretical concept but represents the ultimate goal of many AI research endeavors.

* **AI Engineering:** This is about creating a systematic approach to developing, maintaining, and supporting AI systems in enterprise environments. It ensures that AI applications are scalable, sustainable, and effectively integrated into existing business processes.

* **Autonomic Systems:** These are systems capable of self-management, adapting to changes in their environment while maintaining their objectives. They are autonomous, learn from interactions, and make decisions based on their programming and experiences.

* **Cloud AI Services:** These services provide tools for building AI models, APIs for existing services, and middleware support. They enable the development, deployment, and operation of machine learning models as cloud-based services, making AI more accessible and scalable.

* **Composite AI:** This involves integrating various AI techniques to enhance learning efficiency and broaden the scope of knowledge representations. It addresses a wider range of business problems more effectively by combining different AI approaches.

* **Computer Vision:** This technology focuses on interpreting and understanding visual information from the physical world. It involves capturing, processing, and analyzing images and videos to extract meaningful insights.

* **Data-centric AI:** This approach emphasizes improving training data quality to enhance AI outcomes. It deals with data quality, privacy, and scalability, focusing on the data used in AI systems rather than just the algorithms.

* **Edge AI:** This refers to AI systems implemented at the 'edge' of networks, such as in IoT devices, rather than centralized in cloud-based systems. It's crucial for real-time processing in applications like autonomous vehicles and medical diagnostics.

* **Intelligent Applications:** These applications adapt and respond autonomously to interactions with people and other machines, learning from these interactions to improve their responses and actions.

* **Model Operationalization (ModelOps):** This focuses on managing the entire lifecycle of AI models, including development, deployment, monitoring, and governance. It's essential for maintaining the effectiveness and integrity of AI systems.

* **Operational AI Systems (OAISys):** These systems facilitate the orchestration, automation, and scaling of AI applications in enterprise settings, encompassing machine learning, deep neural networks, and generative AI.

* **Prompt Engineering:** This involves crafting inputs for AI models to guide the responses they generate. It's particularly relevant for generative AI models where the input significantly influences the output.

* **Smart Robots:** These are autonomous, often mobile robots equipped with AI, capable of performing physical tasks independently.

* **Synthetic Data:** This is data generated through algorithms or simulations, used as an alternative to real-world data for training AI models. It's particularly useful in situations where real data is scarce, expensive, or sensitive.

Each of these technologies contributes to the rapidly evolving landscape of generative AI, pushing the boundaries of what's possible and opening up new opportunities across various industries.



# Foundation Model

A foundation model is an AI model that is trained on broad and extensive datasets, allowing it to be applied across a wide range of use cases. These models have become instrumental in the field of artificial intelligence and have powered various applications, including chatbots and generative AI. The term "foundation model" was popularized by the Center for Research on Foundation Models (CRFM) at the Stanford Institute for Human-Centered Artificial Intelligence (HAI). 

The term "foundation model," as coined by the Stanford Institute for Human-Centered Artificial Intelligence's (HAI) Center for Research on Foundation Models (CRFM) in August 2021, refers to a class of AI models that have been meticulously designed to be adaptable powerhouses in the realm of artificial intelligence. These models are characterized by their extensive training on diverse data using self-supervision at scale, making them versatile and capable of tackling a wide range of tasks. The term was chosen with great care to emphasize their intended function, which is to serve as the foundational building blocks for diverse AI applications. Unlike narrower terms like "large language model" or "self-supervised model," "foundation model" underscores their adaptability and applicability across various domains, thereby avoiding misconceptions about their capabilities and training methods. In essence, foundation models represent a groundbreaking approach to AI development, offering boundless potential for innovation and problem-solving across different fields and modalities.

Key points about foundation models:

* General-Purpose Technology: Foundation models are designed to be general-purpose technologies that can support a diverse range of applications. They are versatile and can be adapted to various tasks.

* Resource-Intensive Development: Building foundation models can be highly resource-intensive, with significant costs involved. Some of the most advanced models require substantial investments in data collection and computational power, often costing hundreds of millions of dollars.

* Examples Across Modalities: Foundation models are not limited to text-based applications. They have been developed for various modalities, including images (e.g., DALL-E and Flamingo), music (e.g., MusicGen), robotic control (e.g., RT-2), and more. This broadens their applicability.

* Diverse Fields of Application: Foundation models are being developed and applied in a wide range of fields, including astronomy, radiology, robotics, genomics, music composition, coding, mathematics, and others. They are seen as transformative in AI development across multiple domains.

* Definitions and Regulation: The term "foundation model" was coined by the CRFM, and various definitions have emerged as governments and regulatory bodies aim to provide legal frameworks for these models. In the U.S., a foundation model is defined as having broad data, self-supervision, and tens of billions of parameters. The European Union and the United Kingdom have their own definitions with some subtle distinctions.

* Personalization: Foundation models are not inherently capable of handling specific personal concepts. Methods have been developed to augment these models with personalized information or concepts without requiring a full retraining of the model. This personalization can be achieved for various tasks, such as image retrieval or text-to-image generation.

* Opportunities and Risks: Foundation models offer tremendous opportunities in various fields, including language processing, vision, robotics, and more. However, they also come with risks, including concerns about inequity, misuse, economic and environmental impacts, and ethical considerations. The widespread use of foundation models has raised questions about the concentration of economic and political power.

# Large-scale Language Models

Large-scale language models (LLMs) are distinguished by their comprehensive language comprehension and generation abilities. These models are trained on vast data sets, learning billions of parameters, and require significant computational power for both training and operation. Typically structured as artificial neural networks, predominantly transformers, LLMs are trained through self-supervised and semi-supervised learning methods.

Functioning as autoregressive language models, LLMs process input text and iteratively predict subsequent words or tokens. Until 2020, fine-tuning was the sole approach for tailoring these models to specific tasks. However, larger models like GPT-3 have demonstrated that prompt engineering can achieve comparable results. LLMs are believed to assimilate knowledge of syntax, semantics, and "ontology" from human language data, but they also inherit any inaccuracies and biases present in these data sources.

Prominent examples of LLMs include OpenAI's GPT series (such as GPT-3.5 and GPT-4 used in ChatGPT), Google's PaLM (utilized in Bard), Meta's LLaMA, along with BLOOM, Ernie 3.0 Titan, and Anthropic's Claude 2.

We present the comparative list of LLMs below. Traning cost is presented as (petaFLOP/day). For the training cost column, 1 petaFLOP-day = 1 petaFLOP/sec × 1 day = 8.64E19 FLOP.


| Model Name                    | Release Year   | Developer                     | #Parameters  | Corpus size                                                      | Training cost  | License                 | Comments                                                                                                                                                                                |
|-------------------------|-------------------|-------------------------------|--------------------------|------------------------------------------------------------------|------------------------------|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GPT-1                   | Jun-18            | OpenAI                        | 117 million              |                                                                  |                              |                             | First GPT model, decoder-only transformer                                                                                                                                             |
| BERT                    | Oct-18            | Google                        | 340 million        | 3.3 billion words                                          | 9                      | Apache 2.0             | An early and influential language model, but encoder-only and thus not built to be prompted or generative                                                                   |
| XLNet                   | Jun-19            | Google                        | ~340 million        | 33 billion words                                                |                              |                             | An alternative to BERT; designed as encoder-only                                                                                                                             |
| GPT-2                   | Feb-19            | OpenAI                        | 1.5 billion       | 40GB (~10 billion tokens)                              |                              | MIT                   | general-purpose model based on transformer architecture                                                                                                                                |
| GPT-3                   | May-20            | OpenAI                        | 175 billion         | 300 billion tokens                                        | 3640                    | proprietary                 | A fine-tuned variant of GPT-3, termed GPT-3.5, was made available to the public through a web interface called ChatGPT in 2022                                                   |
| GPT-Neo                 | Mar-21            | EleutherAI                    | 2.7 billion         | 825 GiB                                                   |                              | MIT                    | The first of a series of free GPT-3 alternatives released by EleutherAI. GPT-Neo outperformed an equivalent-size GPT-3 model on some benchmarks, but was significantly worse than the largest GPT-3 |
| GPT-J                   | Jun-21            | EleutherAI                    | 6 billion           | 825 GiB                                                  | 200                    | Apache 2.0                  | GPT-3-style language model                                                                                                                                                             |
| Megatron-Turing NLG     | October 2021 | Microsoft and Nvidia          | 530 billion        | 338.6 billion tokens                                      |                              | Restricted web access       | Standard architecture but trained on a supercomputing cluster                                                                                                                        |
| Ernie 3.0 Titan         | Dec-21            | Baidu                         | 260 billion         | 4 Tb                                                            |                              | Proprietary                 | Chinese-language LLM. Ernie Bot is based on this model                                                                                                                                |
| Claude             | Dec-21            | Anthropic                     | 52 billion        | 400 billion tokens                                        |                              | beta                        | Fine-tuned for desirable behavior in conversations                                                                                                                               |
| GLaM (Generalist Language Model)                   | Dec-21            | Google                        | 1.2 trillion       | 1.6 trillion tokens                                         | 5600                     | Proprietary                 | Sparse mixture of experts model, making it more expensive to train but cheaper to run inference compared to GPT-3                                                                     |
| Gopher                  | Dec-21            | DeepMind                      | 280 billion        | 300 billion tokens                                         | 5833                    | Proprietary                 | Further developed into the Chinchilla model                                                                                                                                           |
| LaMDA (Language Models for Dialog Applications)                  | Jan-22            | Google                        | 137 billion        | 1.56T words, 168 billion tokens                        | 4110                   | Proprietary                 | Specialized for response generation in conversations                                                                                                                                 |
| GPT-NeoX                | Feb-22            | EleutherAI                    | 20 billion         | 825 GiB                                                    | 740                     | Apache 2.0                  | based on the Megatron architecture                                                                                                                                                      |
| Chinchilla              | Mar-22            | DeepMind                      | 70 billion        | 1.4 trillion tokens                                   | 6805                  | Proprietary                 | Reduced-parameter model trained on more data. Used in the Sparrow bot. Often cited for its neural scaling law                                                                         |
| PaLM (Pathways Language Model)                   | Apr-22            | Google                        | 540 billion         | 768 billion tokens                                         | 29250                  | Proprietary                 | aimed to reach the practical limits of model scale                                                                                                                                     |
| OPT (Open Pretrained Transformer)                    | May-22            | Meta                          | 175 billion        | 180 billion tokens                                        | 310                    | Non-commercial research  | GPT-3 architecture with some adaptations from Megatron                                                                                                                                 |
| YaLM 100B               | Jun-22            | Yandex                        | 100 billion         | 1.7TB                                                      |                              | Apache 2.0                  | English-Russian model based on Microsoft's Megatron-LM                                                                                                                                |
| Minerva                 | Jun-22            | Google                        | 540 billion        | 38.5B tokens from webpages filtered for mathematical content and from papers submitted to the arXiv preprint server |                              | Proprietary                 | LLM trained for solving "mathematical and scientific questions using step-by-step reasoning". Minerva is based on PaLM model, further trained on mathematical and scientific data |
| BLOOM                   | Jul-22            | Large collaboration led by Hugging Face | 175 billion         | 350 billion tokens (1.6TB)                                  |                              | Responsible AI              | Essentially GPT-3 but trained on a multi-lingual corpus (30% English excluding programming languages)                                                                                  |
| Galactica               | Nov-22            | Meta                          | 120 billion              | 106 billion tokens                                        | unknown                       | CC-BY-NC-4.0                | Trained on scientific text and modalities                                                                                                                                             |
| AlexaTM (Teacher Models)                | Nov-22            | Amazon                        | 20 billion          | 1.3 trillion                                               |                              | proprietary           | bidirectional sequence-to-sequence architecture                                                                                                                                       |
| LLaMA (Large Language Model Meta AI)                  | Feb-23            | Meta                          | 65 billion          | 1.4 trillion                                               | 6300                    | Non-commercial research  | Trained on a large 20-language corpus to aim for better performance with fewer parameters. Researchers from Stanford University trained a fine-tuned model based on LLaMA weights, called Alpaca |
| GPT-4                   | Mar-23            | OpenAI                        | Exact number unknown   | Unknown                                                         | Unknown                       | proprietary                 | Available for ChatGPT Plus users and used in several products                                                                                                                         |
| Cerebras-GPT            | Mar-23            | Cerebras                      | 13 billion         |                                                                  | 270                      | Apache 2.0                  | Trained with Chinchilla formula                                                                                                                                                      |
| Falcon                  | Mar-23            | Technology Innovation Institute | 40 billion          | 1 trillion tokens, from RefinedWeb (filtered web text corpus) plus some "curated corpora" | 2800                   | Apache 2.0             |                                                                                                                                                                                        |
| BloombergGPT            | Mar-23            | Bloomberg L.P.                | 50 billion               | 363 billion token dataset based on Bloomberg's data sources, plus 345 billion tokens from general purpose datasets  |                              | Proprietary                 | LLM trained on financial data from proprietary sources, that "outperforms existing models on financial tasks by significant margins without sacrificing performance on general LLM benchmarks" |
| PanGu-Σ                 | Mar-23            | Huawei                        | 1.085 trillion            | 329 billion tokens                                        |                              | Proprietary                 |                                                                                                                                                                                        |
| OpenAssistant      | Mar-23            | LAION                         | 17 billion               | 1.5 trillion tokens                                              |                              | Apache 2.0                  | Trained on crowdsourced open data                                                                                                                                                       |
| Jurassic-2         | Mar-23            | AI21 Labs                     | Exact size unknown        | Unknown                                                         |                              | Proprietary                 | Multilingual                                                                                                                                                                     |
| PaLM 2                  | May-23            | Google                        | 340 billion         | 3.6 trillion tokens                                        | 85000                   | Proprietary                 | Used in Bard chatbot                                                                                                                                                           |
| Llama 2               | Jul-23       | Meta                                | 70 billion       | 2 trillion tokens |                              | Llama 2 license     | Successor of LLaMA                                                                                                                                                                                                                 |
| Claude 2              | Jul-23       | Anthropic                           | Unknown                 | Unknown                | Unknown                       | Proprietary         | Used in Claude chatbot                                                                                                                                                                                                       |
| Falcon 180B           | Sep-23       | Technology Innovation Institute     | 180 billion       | 3.5 trillion tokens|                              | Falcon 180B TII license |                                                                                                                                                                                                                                    |
| Mistral 7B            | Sep-23       | Mistral AI                          | 7.3 billion       | Unknown                |                              | Apache 2.0          |                                                                                                                                                                                                                                    |
| OpenHermes-15B  | Sep-23       | Nous Research                       | 13 billion        | Unknown                | Unknown                       | MIT                 |                                                                                                                                                                                                                                    |
| Claude 2.1            | Nov-23       | Anthropic                           | Unknown                 | Unknown                | Unknown                       | Proprietary         | Used in Claude chatbot. Has a context window of 200,000 tokens, or ~500 pages                                                                                                                                                |
| Grok-1                | Nov-23       | x.AI                                | Unknown                 | Unknown                | Unknown                       | Proprietary         | Used in Grok chatbot. Grok-1 has a context length of 8,192 tokens and has access to X (Twitter)                                                                                                                               |
| Gemini                | Dec-23       | Google DeepMind                     | Unknown                 | Unknown                | Unknown                       | Proprietary         | Multimodal model, comes in three sizes. Used in Bard chatbot                                                                                                                                                               |
| Mixtral 8x7B          | Dec-23       | Mistral AI                          | 46.7B total, 12.9B parameters per token | Unknown | Unknown                       | Apache 2.0          | Mixture of experts model, outperforms GPT-3.5 and Llama 2 70B on many benchmarks. All weights were released via torrent                                                                                                      |
| Phi-2                 | Dec-23       | Microsoft                           | 2.7B                    | 1.4T tokens             | Unknown                       | Proprietary         | So-called small language model, that "matches or outperforms models up to 25x larger", trained on "textbook-quality" data based on the paper "Textbooks Are All You Need". Model training took "14 days on 96 A100 GPUs"     |



# Emerging LLM App Stack

The emerging tech stack for LLMs represents a rapidly evolving ecosystem of tools and platforms that empower developers to build and deploy LLM-based applications. With the continuous growth and innovation in the LLM field, it's crucial to highlight the tooling available to complement these models.

One essential component in the LLM app stack is "Playgrounds." Playgrounds serve as user-friendly interfaces that allow developers to experiment with LLM-based applications. They provide an entry point for individuals to interact with LLMs, such as generating text based on prompts or transcribing audio files. These browser-based interfaces often come equipped with the necessary resources, such as GPU access, making them accessible for experimentation.

In terms of app hosting, developers have several options. Local hosting, while cost-effective during the development phase, is limited to individual use and may not scale well for production applications. Self-hosting offers more control over data privacy and application management but comes with significant GPU costs and quality considerations.

Emerging app hosting products like Vercel, Steamship, Streamlit, and Modal are simplifying the deployment of LLM applications. Vercel, for instance, streamlines front-end deployment, allowing developers to quickly deploy AI apps using pre-built templates. Steamship focuses on building AI agents powered by LLMs for problem-solving and automation. Streamlit, an open-source Python library, enables developers to create web front-ends for LLM projects without prior front-end experience. Modal abstracts complexities related to cloud deployment, improving the feedback loop between local development and cloud execution.

The common theme among these emerging tools is their ability to abstract complex technologies, allowing developers to focus on their code and applications. As the AI landscape evolves rapidly, these tools play a crucial role in reducing the time and effort required for building and deploying LLM applications, making them invaluable resources for developers in this dynamic field.
![Screenshot 2023-12-30 121743](https://github.com/ParthaPRay/Curated-List-of-Generative-AI-Tools/assets/1689639/933871b9-30a3-45a7-8b7a-c6f271d0a1f7)


# Evaluating Models

Evaluating a generative AI model involves a multifaceted assessment that encompasses several critical aspects. Firstly, assessing the quality of the model involves scrutinizing the accuracy and relevance of its generated output. However, with the increasing complexity of these models, their behavior can sometimes become unpredictable, potentially leading to outputs that may not always be reliable. Secondly, evaluating the model's robustness is essential, focusing on its ability to handle a wide range of inputs effectively. A pressing concern in the evaluation process is the presence of biases in AI models, which can inadvertently surface due to the inherent biases in the human-generated data used for training. Addressing these biases and navigating the ethical considerations surrounding AI technology are formidable challenges that the AI community must actively address and mitigate.

![model-evaluation](https://github.com/ParthaPRay/Curated-List-of-Generative-AI-Tools/assets/1689639/276edc60-3c19-44ea-abb7-3dab27757119)


# Developer Tools

The Forbes present a technology stack leveraging avrious tools, models and frameworks for developing Generative AI.

![Screenshot 2023-12-30 115145](https://github.com/ParthaPRay/Curated-List-of-Generative-AI-Tools/assets/1689639/286697e9-6458-470f-a6ab-d679751ef186)

As of December, 2023, we show the most used tool sets in generative AI development below.

![52307a3c-6727-4ca5-a4da-208969e7b833_1944x1090](https://github.com/ParthaPRay/Curated-List-of-Generative-AI-Tools/assets/1689639/fc887bda-5341-4099-aebc-cd4170d41bdd)


 
# Chatbots

1. ChatGPT - ChatGPT by OpenAI is a large language model that interacts in a conversational way.
2. Bing Chat - A conversational AI language model powered by Microsoft Bing.
3. Bard - An experimental AI chatbot by Google, powered by the LaMDA model.
4. Character.AI - Character.AI lets you create characters and chat to them.
5. ChatPDF - Chat with any PDF.
5. ChatSonic - An AI-powered assistant that enables text and image creation.


# References

1. https://en.wikipedia.org/wiki/Generative_artificial_intelligence
2. https://en.wikipedia.org/wiki/Large_language_model
3. https://github.com/steven2358/awesome-generative-ai
4. https://www.turing.com/resources/generative-ai-tools
5. https://aimagazine.com/top10/top-10-generative-ai-tools
6. https://www.linkedin.com/pulse/generative-ai-landscape-2023-florian-belschner/
7. https://www.forbes.com/sites/konstantinebuhler/2023/04/11/ai-50-2023-generative-ai-trends/?sh=3e21848d7c0e
8. https://www.gartner.com/en/articles/what-s-new-in-artificial-intelligence-from-the-2023-gartner-hype-cycle
9. https://www.aitidbits.ai/p/most-used-tools
10. https://clickup.com/blog/ai-tools/
11. https://www.linkedin.com/pulse/aiaa-alternative-intelligence-alien-augmented-data-azamat-abdoullaev/
12. https://www.analyticsvidhya.com/blog/2023/09/evaluation-of-generative-ai-models-and-search-use-case/
13. https://blog.gopenai.com/a-deep-dive-into-a16z-emerging-llm-app-stack-playgrounds-and-app-hosting-bf2c9fe7cf18
14. https://www.linkedin.com/pulse/emerging-architectures-large-language-models-data-science-dojo/

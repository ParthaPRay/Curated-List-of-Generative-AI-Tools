# Curated-List-of-Generative-AI-Tools
This repo contains the curated list of tools for generative AI




# Models

| Name                    | Release date[a]   | Developer                     | Number of parameters[b]  | Corpus size                                                      | Training cost (petaFLOP-day) | License[c]                  | Notes                                                                                                                                                                                  |
|-------------------------|-------------------|-------------------------------|--------------------------|------------------------------------------------------------------|------------------------------|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GPT-1                   | Jun-18            | OpenAI                        | 117 million              |                                                                  |                              |                             | First GPT model, decoder-only transformer.                                                                                                                                             |
| BERT                    | Oct-18            | Google                        | 340 million[103]         | 3.3 billion words[103]                                          | 9[104]                       | Apache 2.0[105]             | An early and influential language model,[5] but encoder-only and thus not built to be prompted or generative[106]                                                                      |
| XLNet                   | Jun-19            | Google                        | ~340 million[107]        | 33 billion words                                                |                              |                             | An alternative to BERT; designed as encoder-only[108][109]                                                                                                                             |
| GPT-2                   | Feb-19            | OpenAI                        | 1.5 billion[110]         | 40GB[111] (~10 billion tokens)[112]                              |                              | MIT[113]                    | general-purpose model based on transformer architecture                                                                                                                                |
| GPT-3                   | May-20            | OpenAI                        | 175 billion[24]          | 300 billion tokens[112]                                         | 3640[114]                    | proprietary                 | A fine-tuned variant of GPT-3, termed GPT-3.5, was made available to the public through a web interface called ChatGPT in 2022.[115]                                                    |
| GPT-Neo                 | Mar-21            | EleutherAI                    | 2.7 billion[116]         | 825 GiB[117]                                                    |                              | MIT[118]                    | The first of a series of free GPT-3 alternatives released by EleutherAI. GPT-Neo outperformed an equivalent-size GPT-3 model on some benchmarks, but was significantly worse than the largest GPT-3.[118] |
| GPT-J                   | Jun-21            | EleutherAI                    | 6 billion[119]           | 825 GiB[117]                                                    | 200[120]                     | Apache 2.0                  | GPT-3-style language model                                                                                                                                                             |
| Megatron-Turing NLG     | October 2021[121] | Microsoft and Nvidia          | 530 billion[122]         | 338.6 billion tokens[122]                                       |                              | Restricted web access       | Standard architecture but trained on a supercomputing cluster.                                                                                                                         |
| Ernie 3.0 Titan         | Dec-21            | Baidu                         | 260 billion[123]         | 4 Tb                                                            |                              | Proprietary                 | Chinese-language LLM. Ernie Bot is based on this model.                                                                                                                                |
| Claude[124]             | Dec-21            | Anthropic                     | 52 billion[125]          | 400 billion tokens[125]                                         |                              | beta                        | Fine-tuned for desirable behavior in conversations.[126]                                                                                                                               |
| GLaM                    | Dec-21            | Google                        | 1.2 trillion[18]         | 1.6 trillion tokens[18]                                         | 5600[18]                     | Proprietary                 | Sparse mixture of experts model, making it more expensive to train but cheaper to run inference compared to GPT-3.                                                                     |
| Gopher                  | Dec-21            | DeepMind                      | 280 billion[127]         | 300 billion tokens[128]                                         | 5833[129]                    | Proprietary                 | Further developed into the Chinchilla model.                                                                                                                                           |
| LaMDA                   | Jan-22            | Google                        | 137 billion[130]         | 1.56T words,[130] 168 billion tokens[128]                        | 4110[131]                    | Proprietary                 | Specialized for response generation in conversations.                                                                                                                                 |
| GPT-NeoX                | Feb-22            | EleutherAI                    | 20 billion[132]          | 825 GiB[117]                                                    | 740[120]                     | Apache 2.0                  | based on the Megatron architecture                                                                                                                                                      |
| Chinchilla              | Mar-22            | DeepMind                      | 70 billion[133]          | 1.4 trillion tokens[133][128]                                   | 6805[129]                    | Proprietary                 | Reduced-parameter model trained on more data. Used in the Sparrow bot. Often cited for its neural scaling law.                                                                         |
| PaLM                    | Apr-22            | Google                        | 540 billion[134]         | 768 billion tokens[133]                                         | 29250[129]                   | Proprietary                 | aimed to reach the practical limits of model scale                                                                                                                                     |
| OPT                     | May-22            | Meta                          | 175 billion[135]         | 180 billion tokens[136]                                         | 310[120]                     | Non-commercial research[d]  | GPT-3 architecture with some adaptations from Megatron                                                                                                                                 |
| YaLM 100B               | Jun-22            | Yandex                        | 100 billion[137]         | 1.7TB[137]                                                      |                              | Apache 2.0                  | English-Russian model based on Microsoft's Megatron-LM.                                                                                                                                |
| Minerva                 | Jun-22            | Google                        | 540 billion[138]         | 38.5B tokens from webpages filtered for mathematical content and from papers submitted to the arXiv preprint server[138] |                              | Proprietary                 | LLM trained for solving "mathematical and scientific questions using step-by-step reasoning".[139] Minerva is based on PaLM model, further trained on mathematical and scientific data. |
| BLOOM                   | Jul-22            | Large collaboration led by Hugging Face | 175 billion[140]         | 350 billion tokens (1.6TB)[141]                                  |                              | Responsible AI              | Essentially GPT-3 but trained on a multi-lingual corpus (30% English excluding programming languages)                                                                                  |
| Galactica               | Nov-22            | Meta                          | 120 billion              | 106 billion tokens[142]                                         | unknown                       | CC-BY-NC-4.0                | Trained on scientific text and modalities.                                                                                                                                             |
| AlexaTM                 | Nov-22            | Amazon                        | 20 billion[143]          | 1.3 trillion[144]                                               |                              | proprietary[145]            | bidirectional sequence-to-sequence architecture                                                                                                                                       |
| LLaMA                   | Feb-23            | Meta                          | 65 billion[146]          | 1.4 trillion[146]                                               | 6300[147]                     | Non-commercial research[e]  | Trained on a large 20-language corpus to aim for better performance with fewer parameters.[146] Researchers from Stanford University trained a fine-tuned model based on LLaMA weights, called Alpaca.[148] |
| GPT-4                   | Mar-23            | OpenAI                        | Exact number unknown[f]   | Unknown                                                         | Unknown                       | proprietary                 | Available for ChatGPT Plus users and used in several products.                                                                                                                         |
| Cerebras-GPT            | Mar-23            | Cerebras                      | 13 billion[150]          |                                                                  | 270[120]                      | Apache 2.0                  | Trained with Chinchilla formula.                                                                                                                                                       |
| Falcon                  | Mar-23            | Technology Innovation Institute | 40 billion[151]          | 1 trillion tokens, from RefinedWeb (filtered web text corpus)[152] plus some "curated corpora".[153] | 2800[147]                    | Apache 2.0[154]             |                                                                                                                                                                                        |
| BloombergGPT            | Mar-23            | Bloomberg L.P.                | 50 billion               | 363 billion token dataset based on Bloomberg's data sources, plus 345 billion tokens from general purpose datasets[155]  |                              | Proprietary                 | LLM trained on financial data from proprietary sources, that "outperforms existing models on financial tasks by significant margins without sacrificing performance on general LLM benchmarks" |
| PanGu-Σ                 | Mar-23            | Huawei                        | 1.085 trillion            | 329 billion tokens[156]                                         |                              | Proprietary                 |                                                                                                                                                                                        |
| OpenAssistant[157]      | Mar-23            | LAION                         | 17 billion               | 1.5 trillion tokens                                              |                              | Apache 2.0                  | Trained on crowdsourced open data                                                                                                                                                       |
| Jurassic-2[158]         | Mar-23            | AI21 Labs                     | Exact size unknown        | Unknown                                                         |                              | Proprietary                 | Multilingual[159]                                                                                                                                                                      |
| PaLM 2                  | May-23            | Google                        | 340 billion[160]         | 3.6 trillion tokens[160]                                        | 85000[147]                   | Proprietary                 | Used in Bard chatbot.[161]                                                                                                                                                            |


# Developer Tools

1. 


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

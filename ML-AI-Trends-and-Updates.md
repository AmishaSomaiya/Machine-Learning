
Idefics2 by Hugging Face on April 15, 2024
https://www.linkedin.com/posts/huggingface_last-week-hugging-face-released-idefics2-activity-7193950827680518144-tBtq?utm_source=share&utm_medium=member_desktop
- 8B open-source visual language model
- based on Google's Flamingo (not open-sourced)
- convert image to json
- smaller parameter size
- an open license
- improved Optical Character Recognition (OCR) capabilities
- Image-aware Decorder Enhanced à la Flamingo with Interleaved Cross-attentionS, Idefics is a general multimodal model that can respond to text and image prompts. While its predecessor has a parameter size of 80 billion, Idefics2 is a tenth of the size at 8 billion, comparable to DeepSeek-VL and LLaVA-NeXT-Mistral-7B.
- better image manipulation in the native resolution of up to 980 x 980 pixels and native aspect ratios. Images will no longer need to be resized to accommodate a fixed-size square ratio, which is traditionally done in computer vision.
- OCR abilities have been enhanced through data integration generated from transcribing text in an image or document. Hugging Face’s team has also improved Idefics’ ability to answer questions on charts, figures and documents.



CatLIP by Apple April 25, 2024

Llama3 + Groq April 21, 2024 
https://pub.towardsai.net/llama-3-groq-is-the-ai-heaven-337b6afeced3 


Corenet SLM by Apple   (around April 24 2024)
Phi3 mini SLM by Microsoft (around April 23 2024)

gemini, claude (need to check timeline position)

Loreft 

Llama3 8B by Meta and 140B in training, April 18 2024

Infinite Context by Google, April 10 2024

1 bit LLM by Microsoft, Winter 2024

Llama model by Stanford team used GPT-3 generated instructions, costing less but with reduced performance compared to GPT-3, only in 600$


LLaMA = open source chinchilla model from meta in several sizes and each saw at least 1 trillion tokens 
Llama is an open-source chinchilla optimal LLM from Meta Research
Several sizes available, ranging from 7 billion to 65 billion, with at least 1 trillion tokens
Competitively benchmarks against GPT-3 and other state-of-the-art LLMs
Open source but non-commercial license for pre-trained weights
Trained on custom common crawl filtering, C4, GitHub, Wikipedia, books, and scientific papers
Data set replicated by Red Pajama, which is also training models to replicate Llama
Interesting inclusion of GitHub as a training resource for llama, even though  T5 removed code from training data , inclusion of code improves perf on non code tasks : found by openai on the codex model which was perf better on reasoning tasks ac compsred to gpt 

OpenAI found this with their Codex model, which was fine-tuned on code and outperformed GPT-3 on reasoning tasks
Since then, people have been adding code to training data
Open source dataset called 'the stack' collects code from GitHub while respecting licenses

"Open Assistant" A specific data set for instruction tuning in chat-based paradigms


chinchilla (70B para, 1.4 trillion tokens) beats gofer (280B, 300 billion tokens=gpt3, and not seeing all the data that we have) by 4x lesser para but needs 4x more data 


gpt 4 (2023) still larger, unknown
1.76 trillion paras, 13 trillion tokens = ~ 10 trillion words , 2 epochs for text based and 4 epochs for code based data , can process 25000 words at once = 8x more than gpt 3
can receive upto 128k input tokens but output only upto 4096 tokens
The biggest GPT-4 model can only process ~50 pages of input text, and performance (measured by inference time and accuracy) degrades badly as you approach this limit, called a context window.


-gpt3 (2022) 12-96 layers 175Bi parameters
100x larger than gpt2 = 175B paras, excellent zero shot and few shot learning, data = webtext + raw common crael + selection of books + all of wiki = 500 bi tokens, but only trained on 300 bi tokens 

shift from text completion mindset to instruction following mindset
OpenAI hired thousands of contractors to gather zero-shot data and used reinforcement learning for training
GPT model lineage includes DaVinci, Codex, and various iterations, fine-tuning for specific applications
Chatgpt further trained on not just zero or few shot but full conversations 


-bert (2019) encoder only 100 mi paras
-t5 (2020) encoder-decoder 11bi paras, 160 bi tokens


gpt2 (2019) decoder only 1.5 B paras, data =webtext =  45M Redit -> 8M docs after filtering -> 40GB text

-	Transformer and attention map: 2017 by Google

Separate NN for each task 
RNN + LSTM for seq tasks
CNN for vision
etc 

-	Diffusion Models : 2015 and Stable diff
-	GANs and StyleGANs : 2014
-	Markov chains: 1906 : next word prediction but simple limited the ability to generate plausible text

References : 
https://fullstackdeeplearning.com/llm-bootcamp/spring-2023/llm-foundations/
LinkedIn Posts

https://a16z.com/emerging-architectures-for-llm-applications/


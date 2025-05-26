Deploy optimized LLMs across a multidevice architecture with the OpenVINO toolkit
With the AI PC from Intel, you can harness the power of the latest LLMs from Hugging Face* on your own device with a single line of code. The complete code base for this solution is available open source on GitHub*. The GitHub repository contains the development tools needed to optimize LLMs sourced from Hugging Face using OpenVINO™ toolkit model optimizations and run comprehensive AI inference on your AI PC using Intel® Core Ultra™ processors.

The OpenVINO toolkit is open source and used to optimize and deploy AI inference.

Boost deep learning performance in computer vision, automatic speech recognition, natural language processing, and other common tasks.
Use models trained with popular frameworks like TensorFlow*, PyTorch*, and more.
Reduce resource demands and efficiently deploy on a range of Intel platforms from edge to cloud.
This open source version includes several components:

OpenVINO™ model server
OpenVINO Runtime
CPU, GPU, multidevice, and heterogeneous plug-ins to accelerate deep learning inference on Intel CPUs and Intel processor graphics
The toolkit supports pretrained models from Open Model Zoo, along with 100+ open source and public models in popular formats such as TensorFlow, ONNX* (Open Neural Network Exchange), PaddlePaddle*, Apache MXNet*, Caffe*, and Kaldi.

For more information on the OpenVINO toolkit, see the GitHub repository.

System Requirements
Before running this application, please ensure your AI PC meets OpenVINO toolkit system requirements.

Install Dependencies
To use this application, first clone the repository and install the dependencies:

git clone https://github.com/intel/ai-innovation-bridgecd ai-innovation-bridge/utilities/model-card-tools/openvinopip install -r requirements.txt
Once you have successfully installed the dependencies, you are ready to optimize your Hugging Face LLMs using the OpenVINO toolkit.

Supported LLM Tasks
The openvino_llm_optimizations.py script currently supports the following LLM tasks:

text-generation
translation_en_to_fr
Model Optimizations
You can customize the openvino_llm_optimizations.py script by modifying the following parameters:

model_path: The model path to LLM on Hugging Face (for example, helenai/gpt2-ov).
task: The LLM task (supported options include text-generation and translation_en_to_fr)
device: The device on the AI PC to optimize the LLM (supported options include GPU, NPU, and CPU)
prompt: The input prompt for the LLM inference task
Example Use
Text Generation
To run optimized inference with a text generation LLM with the OpenVINO toolkit on your AI PC with an Intel® Arc™ GPU, use the following command:

python openvino_llm_optimizations.py --model_path=helenai/gpt2-ov --task=text-generation --device=gpu --prompt="In the spring, flowers bloom"
Your output should be similar to:

GPU device selected is available. Compiling model to GPU.Optimizing helenai/gpt2-ov LLM with OpenVINO.helenai/gpt2-ov LLM optimized with OpenVINO on GPU and inference completed in 4.94 seconds!

Prompt entered: In the spring, flowers bloomResponse: In the spring, flowers bloom all over the land. The flowers bloom in summer, when the rains soak away.
Text Translation
To run optimized inference with a text translation LLM with the OpenVINO toolkit on your AI PC with an Intel Arc GPU, use the following command:

python openvino_llm_optimizations.py --model_path=t5-small --task=translation_en_to_fr --device=gpu --prompt="In the spring, flowers bloom"
Your output should be similar to:

GPU device selected is available. Compiling model to GPU.Optimizing t5-small LLM with OpenVINO.t5-small LLM optimized with OpenVINO on GPU and inference completed in 21.335 seconds!

Text to translate: In the spring, flowers bloomTranslation: Au printemps, les fleurs fleurissent

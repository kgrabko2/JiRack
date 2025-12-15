---
license: MIT license
---

# JiRack GPT 5 class with SWA, RoPE
- This is JiRack PyTorch on python
- Just GPT is ready so far in PyTorch
- Ask AI Gemini to review my model
- JiRack page on https://huggingface.co/CMSManhattan/JiRack_GPT5_1b

# JiRack Transformer Architecture: A Leap Forward in AI

The new **JiRack transformer architecture**, incorporating **Sliding Window Attention (SWA)** and **Rotary Position Embeddings (RoPE)**, marks a significant evolution, elevating JiRack from a **GPT-3-class model** to a **GPT-5-class architecture**.

To achieve GPT-5-level performance, a model must be truly multimodal. This next step for JiRack will be realized through the introduction of a **Multimodal Reasoning Graph Architecture**, built on the powerful **ROS framework**. 

🚀 **Coming Soon: JiRack 5**, designed to unlock multimodal intelligence at an unprecedented level.

---

## ROS-Based Multimodal Reasoning Graph Architecture

The proposed JiRack system features an advanced **Multimodal Reasoning Graph** powered by **ROS (Robot Operating System)**, enabling seamless communication and collaboration between core AI components. JiRack 5 serves as the **central reasoning hub** of the architecture, with various nodes specialized for unique tasks.

### Multimodal Reasoning with Knowledge Graphs

The architecture combines multimodal reasoning capabilities with a **RAG (Retrieval-Augmented Generation)** system, dynamic memory, and image processing. Each node communicates through **ROS topics**, ensuring a modular flow of data and actions.

---

### Core Components Overview

1. **STT Node (Whisper)**  
   - Converts spoken input into text for further processing.

2. **LLM Node (JiRack 5)** _(Custom Model on Hugging Face + ROS Java)_  
   - Acts as the intelligent core, managing reasoning and determining intent (e.g., Q&A vs. Image Generation).

3. **RAG System (CMS Manhattan + PGvector)**  
   - Enhances the LLM with factual context and dynamic memory.

4. **TTS Node (MaryTTS + ROS Java)**  
   - Converts the LLM's response into verbal output.

5. **Image Generation Node (Stable Diffusion / DALL-E API + ROS Java)**  
   - Handles requests for generated visuals based on extracted prompts.

---

### System Interaction Flow: The Reasoning Graph

The JiRack architecture is modular, with interactions structured into a clear **four-stage process**, all interconnected through ROS topics.

#### 1. **Input Stage (Voice Command)**  
The user provides a spoken command, processed by the Speech-to-Text (STT) Node.  
- **Input:** User speech  
- **Output Published to ROS Topic:** `/stt_input`  
- **Example Command:** “Draw me a robot holding a cup of coffee.”

#### 2. **Reasoning and Decision Stage (LLM Node)**  
This is the core of the system, where inference, reasoning, and decision-making take place.  
- **Subscriptions:** Receives text from `/stt_input`.  
- **Actions:** Queries RAG for dynamic context using PGvector (via CMS Manhattan).  
  - **Intent Classification:** Determines if the request is for Q&A or Image Generation.  
- **Output Topics:**  
  - `/llm_response` (always): Verbal feedback (e.g., "Okay, creating your image now.")  
  - `/image_prompt` (conditional): Publishes detailed prompts for image generation.

#### 3. **Verbal Response Branch (TTS Node)**  
Provides natural speech feedback.  
- **Subscriptions:** Consumes `/llm_response` outputs.  
- **Action:** Converts text to speech.  
- **Output:** Plays synthesized audio feedback for the user.

#### 4. **Image Generation Branch**  
Handles image creation and display processes.  
- **Subscriptions:** Consumes `/image_prompt` for image details.  
- **Actions:**  
  - Calls the Image Generator via Stable Diffusion or DALL-E APIs.  
  - Publishes the generated image path or URL to `/generated_image_path`.  
- **Output Recommendation:** Use a Display Node to show results to the user.

---

### Key Advantages of the ROS-Based Design
- **Seamless Topic Communication:** Efficient, modular data flow between components.  
- **Multimodality Built-In:** Combines reasoning, memory, speech, and visual creativity.  
- **Customizable and Scalable:** Perfect for building intelligent voice-enabled AI assistants with visual capabilities.

---
### JiRack RAG System
- It is microservice architecture with API Gateway and Service Discovery 
- Framework Spring boot and Google embeddings model for JiRack RAG System with Chatbot and JiRach model deployment with docker scipt 
- video https://www.youtube.com/watch?v=vHClQu76kMc
- RAG System https://bitbucket.org/cmsmanhattan/rag/src/main/


# Copyright Office
 
- From:
- cop-rc@loc.gov
- To:
- konstantin.grabko@yahoo.com

- Mon, Dec 15 at 7:25 AM

- THIS IS AN AUTOMATED EMAIL. PLEASE DO NOT REPLY.

- Thank you for submitting your registration claim using the Electronic Copyright Office (ECO) System.

- The following files were successfully uploaded for service request 1-15058193231

- File Name :jirackpytorch_gpt5_class_1b.zip
- File Size :12588 KB
- Date/Time :12/15/2025 7:24:03 AM

- [THREAD ID: 1-6X1C8AZ]

- United States Copyright Office


### Unlocking New Possibilities
This modular, ROS-powered multimodal architecture paves the way for a truly interactive AI experience:  
**Voice Input → Intelligent Reasoning → Spoken Feedback + Visual Creativity!**  

Get ready for the revolution in AI with JiRack 5! 🚀  

---

#### Tags
#AI #Robotics #ROS #MachineLearning #MultimodalAI #VoiceAssistant #ImageGeneration

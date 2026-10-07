# 🧠 Micro-VLM: The Story of What We Built, Why We Built It, and How It Works

> **A human-friendly guide to building a complete multimodal AI from scratch.**  
> *No Ph.D. required. No hand-wavy buzzwords. Just the real engineering story.*  
> *(For raw code, equations, and tensor shapes, see the [Developer README](README.md).)*

---

## 🧭 The Core Motivation: Why Build This?

Whenever you hear about AI today, it’s always about scale:
* *"Model X has 400 billion parameters!"*
* *"Trained on 25,000 graphics cards for $50 million!"*

When tutorials claim to show you how to *"build a modern AI from scratch,"* they almost always pull a sleight-of-hand: they tell you to install a giant library like Hugging Face, download gigabytes of pre-made weights that Big Tech trained, write five lines of glue code, and call it a day.

**We wanted to know what actually happens under the hood.**

If you take away the billion-dollar server farms and third-party libraries, how does an artificial neural network actually:
1. Turn a stream of human words into numbers?
2. Look at a grid of colored pixels and extract concepts like *"red"* or *"circle"*?
3. Combine text and vision into a single stream of thought?
4. Make logical decisions without melting a normal laptop CPU?
5. Teach itself to get better at reasoning through trial and error?

To answer those questions, we built **Micro-VLM**: a 17.5-million-parameter Vision-Language Model written from the ground up in standard Python and PyTorch. Every single layer, every matrix operation, every training loop, and every reinforcement learning step was crafted by hand.

---

## 🏗️ What We Actually Built (The 30,000-Foot View)

Micro-VLM is a compact artificial brain that can **read text**, **look at pictures**, and **solve reasoning problems**. 

Here is the entire system laid out simply:

```
┌─────────────────┐       ┌────────────────────────┐
│  Image (64x64)  │ ───►  │  Vision Eye (ViT)      │ ───►  16 Visual Thoughts
└─────────────────┘       └────────────────────────┘             │
                                                                 ▼
┌─────────────────┐       ┌────────────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Text Prompt    │ ───►  │  Universal Byte Reader │ ───► Splicer merges  │ ──►│ MoE Brain Trunk │ ──► Decision
└─────────────────┘       └────────────────────────┘    │ Image into Text │    │ (4 Experts)     │
                                                        └─────────────────┘    └─────────────────┘
```

It consists of four main parts:
1. **The Universal Byte Reader ([tokenizer.py](src/model/tokenizer.py))**: Reads text as raw computer bytes instead of memorizing a massive 50,000-word dictionary.
2. **The Vision Eye ([vision.py](src/model/vision.py))**: Looks at an image, understands height and width, and compresses the picture into 16 dense visual thoughts.
3. **The Council of Experts ([transformer.py](src/model/transformer.py) & [model.py](src/model/model.py))**: An 8-layer Transformer brain where 4 specialized mini-brains (Experts) take turns solving different parts of each thought, keeping compute light.
4. **The Trial-and-Error Teacher ([grpo_rl.py](src/training/grpo_rl.py))**: A modern reinforcement learning algorithm (GRPO, from DeepSeek) that lets the model practice logic and learn from its mistakes without needing a second expensive AI to grade it.

---

## 🧩 Step-by-Step: How Each Piece Was Built and Why

### 1. Reading Text: Why We Used Raw Bytes (Vocabulary = 260)

* **The Standard Way**: Models like ChatGPT use a dictionary of 32,000 to 100,000 words and word fragments (called Byte-Pair Encoding or BPE).
* **The Problem**: In a small model, just storing a 50,000-word dictionary would take up 15 to 20 million parameters—eating up our entire model budget before we even added a single layer of thinking!
* **What We Did**: We taught our model to read **raw computer bytes**.
  * Every single character in the digital world (letters, numbers, emojis, Arabic, Japanese) is made of numbers between `0` and `255`.
  * We assigned numbers `0` to `255` directly to those bytes.
  * We added only 4 special marker tags:
    * `256: <PAD>` (empty space filler)
    * `257: <BOS>` (marks the start of a sentence)
    * `258: <EOS>` (marks the end of a thought)
    * `259: <IMG>` (placeholder slot where an image will be plugged in)
* **The Result**: Our entire vocabulary is locked at exactly **260 numbers**. It can read any language on Earth, never encounters an "unknown word" error, and takes up less than 0.4% of the model’s memory.

---

### 2. Giving the AI an Eye: The Vision Pipeline

Many toy tutorials take an image, flatten it out into a straight line of numbers, and dump it into the text brain. That doesn't work well because images aren't sentences—in an image, what is *above* or *below* an object is just as important as what is to the left or right.

Here is how we built the vision system:

1. **Chopping the image into patches**: We take a $64 \times 64$ color image and cut it into an $8 \times 8$ grid of 64 small squares (patches), like puzzle pieces.
2. **2D Spatial Awareness (Axial RoPE)**: Standard text AI only knows words moving from left to right. We created a 2D coordinate system where the model rotates numbers along both an $(X)$ axis and a $(Y)$ axis simultaneously. Now the AI knows exactly where every puzzle piece sits in 2D space.
3. **The 2×2 Smart Zoom (Downsampler)**: 64 patches is still too much clutter for a small text model to process. We designed a downsampler that takes neighboring $2 \times 2$ clusters of patches and merges them together. This shrinks the image from **64 pieces down to just 16 high-density concept tokens**—a 75% reduction in sequence length!
4. **The Projector (Translator)**: A small neural translator (using SwiGLU gating) converts those 16 visual tokens directly into the language of the text brain ($D = 256$).
5. **Image Splicing**: In the text prompt, wherever the user writes `<IMG>`, our code cuts out the dummy text slot and inserts the real vision token right into the stream of thought.

---

### 3. The Brain Trunk: The "Council of Experts" (MoE)

In a traditional neural network, every single neuron fires for every single word. If a model has 20 million parameters, all 20 million run for every letter. That makes models sluggish and power-hungry on a normal laptop CPU.

To solve this, we used **Mixture-of-Experts (MoE)**:
* Inside each layer of the brain, we created **4 separate mini-networks (Experts)**.
* When a thought arrives, a fast router evaluates it and picks **only the top 2 best-qualified experts** to process it. The other 2 experts stay asleep.
* **Why this matters**: The model has the memory and capacity of a large model, but only spends the compute of a much smaller one!

#### Alternating "Zoomed-In" and "Panoramic" Vision
Computing attention across every word gets exponentially more expensive as sentences get longer.
* **Even Layers (0, 2, 4, 6)**: Use a **sliding window** ($w=32$). They only look at the immediate neighborhood of recent words.
* **Odd Layers (1, 3, 5, 7)**: Take a **panoramic view**. They look back across the entire history to maintain big-picture context.
* This alternating rhythm keeps the model fast enough to run in real-time on an everyday CPU.

---

### 4. Reinforcement Learning: Teaching Logic Without a Boss (GRPO)

Pre-training makes an AI good at predicting what words usually come next, but it doesn't guarantee strict logical reasoning. It will often guess or ramble.

To teach the model strict logic, we implemented **Group Relative Policy Optimization (GRPO)**—the cutting-edge reinforcement learning technique made famous by DeepSeek-R1.

* **The Old Way (PPO)**: Requires training a second, expensive "Critic" AI that acts like a nagging teacher, constantly giving grades to the main AI. This doubles memory usage.
* **The GRPO Way (No Critic Needed)**:
  1. We give the model a logic puzzle: `"Eval: True and not False -> "`
  2. The model generates **8 different attempts** at the same time.
  3. We score all 8 attempts (+1.0 for the right answer, -0.5 for the wrong answer).
  4. We compare the 8 scores against their own group average. Whichever attempts did better than average get rewarded; the ones that did worse get penalized.
  5. The model adjusts its weights toward the winning attempts while checking against an unmoving "reference copy" of itself so it doesn't go crazy.

---

## 🛠️ The Real Engineering War Stories (Bugs, Blunders & Breakthroughs)

Building AI from scratch on your own machine means hitting real bugs that tutorials gloss over. Here are the 9 real war stories from our trenches:

### 1. The Zero-Output Ghost (Windows Process Silence)
* **The Symptom**: We wrote `pretrain.py`, hit Enter in PowerShell, and... nothing happened. No error, no training text, no warning. The terminal just spat us right back to the command prompt like nothing was there.
* **The Cause**: Two silent blockers collided. First, running Python from the root directory couldn't find sibling files inside `src/` without explicit paths, so it failed silently. Second, PyTorch's multithreaded data loader on Windows tried to spawn sub-processes that instantly deadlocked on CPU memory.
* **The Fix**: We injected runtime directory paths (`sys.path.append(...)`) and pinned `DataLoader(..., num_workers=0)` to force clean, single-process execution. The training loop sprang to life.

### 2. The Impossible "Loss of 145" Bug (The RoPE Math Disaster)
* **The Symptom**: The first test run reported a math error (loss) of **145.0**. In information theory, a network randomly guessing across a vocabulary of 260 byte-characters can only have a maximum initial error of $\ln(260) \approx 5.56$. A loss of 145 was mathematically impossible unless numbers were exploding into infinity.
* **The Cause**: Our positional rotation formula (1D RoPE) had a slicing bug that generated only 16 frequencies instead of the required 32. When multiplied during self-attention, numbers shot up to $+500$ and $-500$, completely saturating the network.
* **The Fix**: We rewrote the frequency generator from scratch to cover all 32 dimensions of each attention head. The error immediately collapsed to the expected $\sim 5.5$ on random initialization, and dropped to $0.2348$ by epoch 5.

### 3. The 15% Baseline & The Infinite Echo (`-> -> ->`)
* **The Symptom**: Before reinforcement learning, the model scored a miserable **15%** on basic True/False questions—worse than flipping a coin! When we looked at what it was actually saying:
  ```text
  Prompt : Eval: True and not False -> 
  Output : -> -> -> 20
  ```
* **The Cause**: Language models love echoing recent patterns. Because the prompt ended with an arrow `-> `, the model treated the arrow as a repetitive stammer rather than a question prompt.
* **The Fix**: In GRPO reinforcement learning, we added a strict penalty: $+1.0$ for getting the logic right, $+0.2$ for valid words, and a severe $-1.0$ penalty for repeating punctuation or rambling. Within 25 RL steps, the stutter vanished.

### 4. The 78.6% Parameter Freeze (Protecting the Vision Tower)
* **The Symptom**: During our first RL run, we updated all 17.5M parameters on logic questions. But the logic puzzles contained zero images.
* **The Cause**: Backpropagating pure text updates through the entire network bombarded the vision layers with meaningless random gradient noise. It caused "catastrophic forgetting"—destroying the model's visual comprehension.
* **The Fix**: We locked down all 1.8M parameters of the Vision Tower and the first 6 layers of the transformer trunk. We froze **13,750,220 parameters (78.6%)**, leaving only **3.7M parameters (21.4%)** in the top 2 layers and output head active to learn logic. The visual foundation remained untouched.

### 5. Group Rollout Starvation ($G=4 \to G=8$)
* **The Symptom**: Early GRPO training stalled out with negative rewards ($-0.50$) for 20+ steps with zero improvement.
* **The Cause**: We started by generating only 4 candidate answers per prompt ($G=4$). In early exploration, if all 4 attempts were wrong (e.g. `['3', '->', '20', '->']`), every attempt scored $-0.50$. The standard deviation became zero: the formula $A_i = \frac{R_i - \mu}{\sigma + \epsilon}$ had no difference to measure! Without contrast, the AI had no idea which way to steer.
* **The Fix**: We doubled the group size to $G=8$ candidates per prompt and bumped the temperature to $0.8$. With 8 varied attempts, at least one candidate hit a partial syntax bonus or correct answer, restoring the variance signal and driving accuracy up to 50.0% (and 75% after calibration).

### 6. The Green Canvas That Answered "False." (The `<IMG>` Trap)
* **The Symptom**: When we showed the model a solid green square and asked *"What is the primary tint?"*, it replied: `"False."`
* **The Initial Wrong Diagnosis**: We assumed the model was so obsessed with True/False logic puzzles that its brain had forgotten colors.
* **The Real Discovery**: It was a tokenizer bug! The prompt contained the text `"<IMG>"`. Our tokenizer turned that into five individual ASCII letters: `'<'`, `'I'`, `'M'`, `'G'`, `'>'` instead of the special image token ID `259`. Because token 259 was absent, the code that injects visual patches never triggered—**the model was never shown the picture!** It was answering blind.
* **The Fix**: We updated the tokenizer to parse `"<IMG>"` into token ID `259`. The moment the 16 visual tokens were injected into the sequence, the model immediately identified the canvas as **`green.`** with 100% precision.

### 7. The Interleaved Anchor Pass (Stopping Boolean Mode Collapse)
* **The Symptom**: Even with vision working, the model leaned toward answering `"False."` on ambiguous inputs because 76% of its training rewarded True/False words. The output probabilities for those two words were drowning out the rest of the dictionary.
* **The Cause**: "Mode Collapse"—when an AI discovers that spitting out a tiny set of safe words minimizes penalties across the board.
* **The Fix**: We added an Interleaved Replay Buffer: 70% of steps ran RL logic rollouts, and 30% of steps ran standard supervised learning on colors and math. This forced the upper layers to remember normal vocabulary while mastering logic.

### 8. The "15 + 27 = 60" Carry Barrier
* **The Symptom**: When asked `Calc: 15 + 27 = `, the model blurted out `60`.
* **The Analysis**: It didn't output gibberish—it output a two-digit integer in the right ballpark!
* **The Architectural Barrier**: Standard transformers generate text left-to-right. To output `42`, the model must write `'4'` before `'2'`. But addition works right-to-left: you compute $5 + 7 = 12$, write down $2$, and carry the $1$ over to get $1 + 1 + 2 = 4$. Without a scratchpad or thinking steps, asking a 17.5M model to resolve a multi-digit carry in a single instant forces it to approximate. The answer `60` proved it understood math syntax, but lacked the sequential computational steps to compute carries.

### 9. The "422" Stutter & Scratchpad Leak (The Battle for Exact Carry)
* **The Symptom**: When we introduced scratchpads (`[t_sum+u_sum] ans`), querying `Calc: 15 + 27 = ` yielded `[40+12] 422`.
* **The Double Failure Under the Hood**:
  1. **Scratchpad Leak**: Tens of 15 and 27 are $10$ and $20$ (sum = $30$). But the model output `[40+...]`. Why? Because $5 + 7 = 12$ creates a carry of $1$ into the tens digit ($1+2+1=4$), the model prematurely leaked the final answer's tens digit (`4`) into the initial scratchpad token (`40`).
  2. **Digit Stuttering (`42` $\to$ `422`)**: The model actually emitted `'4'` and `'2'`, completing `42`! But because the dataset target lacked a hard ending boundary (like `= {ans}.`) and many additions have 3 digits, the model chose to repeat `'2'` with 74% probability rather than stop, corrupting `42` into `422`.
* **The Fix**:
  1. We formatted the training generator with an explicit closing formula: `Calc: {a} + {b} = [{t_sum}+{u_sum}] = {ans}.`
  2. We configured the inference client to stop immediately upon hitting the terminal period (`.`).
  3. We ran multi-task calibration balancing carries, logic, and vision.
* **The Final Result**: `Calc: 15 + 27 = ` $\implies$ `[30+12] = 42.` in 270ms on CPU. Zero stutters, exact tens ($30$), exact units ($12$), and exact sum ($42.$)!

---

## 📈 The Journey: Real Training Numbers

We ran the model through real training on an ordinary CPU:

```
[ Phase 1: Pre-training ]
Start: Loss = 2.95  ──►  End of Epoch 5: Loss = 0.2348
(The model learned basic English, color words, and arithmetic syntax)

[ Phase 2: GRPO Reinforcement Learning ]
Baseline Logic Accuracy: 15.0%
Step 25: First structured breakthrough (outputting valid "-> False" format)
Step 200: Consistently deriving correct logic chains
Final Aligned Accuracy: 50.0%  (+35.0% Net Gain!)
```

---

## 🚀 The Real-World Application: The "Jarvis" Edge Assistant

We didn't build this to sit in an academic paper. We integrated it into **Jarvis**, an instant local OS assistant for your machine:

```
[ User Request ]
       │
   ┌───┴──────────────────────────────────────┐
   ▼                                          ▼
[ Known Direct Command? ]             [ Complex Thought? ]
("lock screen", "mute volume")        ("Explain quantum physics")
   │                                          │
   │ (< 0.15 milliseconds)                    │ (~130 milliseconds)
   ▼                                          ▼
[ Instant Local Execution ]           [ Micro-VLM Neural MoE Pass ]
(Ran on your laptop without lag)              │
                                      ┌───────┴───────┐
                                      ▼               ▼
                               [ Local Action ]  [ Escalate to Cloud LLM ]
```

1. **Instant Reflexes (< 0.15 ms)**: Routine commands like *"mute audio"* or *"lock screen"* fire on an instant reflex path with zero lag.
2. **Local Neural Brain (~130 ms)**: If the prompt is conversational or ambiguous, our 17.5M model reads the sentence on CPU, interprets the intent, and either executes it or flags it to escalate to a large cloud model.

---

## 🏃 Try It in 60 Seconds

You don't need a GPU or special drivers. Any standard laptop with Python works:

```bash
# 1. Activate the environment
.\venv\Scripts\activate   # Windows
# source venv/bin/activate # Mac/Linux

# 2. Run the client demo (tests both text logic and vision!)
python src/serve/client_example.py

# 3. Test the Jarvis edge latency benchmark
python src/serve/benchmark.py
```

### What You Will See:
```text
Prompt   : Eval: not (False or False) -> 
Response : True
Latency  : ~115 ms

Prompt   : Visual [64x64 Green Canvas] Question: Primary tint? Answer:
Response : green.
Latency  : ~250 ms
```

---

## 🏆 Key Takeaways

| What People Usually Think | What We Proved With Micro-VLM |
| :--- | :--- |
| *"You need billions of parameters to make a VLM."* | A clean **17.5M architecture** can see, read, and reason on your laptop CPU. |
| *"You must use huge 50,000-word tokenizers."* | A **260-byte tokenizer** handles all languages, codes, and images without OOV bugs. |
| *"Reinforcement learning needs a giant server farm."* | **Critic-free GRPO** can boost reasoning accuracy from 15% to 50% right on a home CPU. |
| *"Every neuron must fire on every word."* | **Mixture-of-Experts** only wakes up 2 out of 4 experts, keeping the laptop cool and fast. |

---

*Want to dive into the exact PyTorch code, attention masks, and mathematical formulations? Head over to the [Technical Developer README](README.md).*


 ███            █████████  ██████████ ██████   ██████ █████ ██████   █████ █████
░░░███         ███░░░░░███░░███░░░░░█░░██████ ██████ ░░███ ░░██████ ░░███ ░░███
  ░░░███      ███     ░░░  ░███  █ ░  ░███░█████░███  ░███  ░███░███ ░███  ░███
    ░░░███   ░███          ░██████    ░███░░███ ░███  ░███  ░███░░███░███  ░███
     ███░    ░███    █████ ░███░░█    ░███ ░░░  ░███  ░███  ░███ ░░██████  ░███
   ███░      ░░███  ░░███  ░███ ░   █ ░███      ░███  ░███  ░███  ░░█████  ░███
 ███░         ░░█████████  ██████████ █████     █████ █████ █████  ░░█████ █████
░░░            ░░░░░░░░░  ░░░░░░░░░░ ░░░░░     ░░░░░ ░░░░░ ░░░░░    ░░░░░ ░░░░░


Tips for getting started:
1. Ask questions, edit files, or run commands.
2. Be specific for the best results.
3. Create GEMINI.md files to customize your interactions with Gemini.
4. /help for more information.


╭───────────────────────────────────────────────────────────────────────────────────────────╮
│ Gemini CLI update available! 0.1.5 → 0.1.15                                               │
│ Run npm install -g @google/gemini-cli to update                                           │
╰───────────────────────────────────────────────────────────────────────────────────────────╯


> Please give me a detailed summary of this repository.

 ╭───────────────────────────────────────────────────────────────────────────────────────────╮
 │ ✔  ReadFile README.md                                                                     │
 ╰───────────────────────────────────────────────────────────────────────────────────────────╯
✦ This repository contains the code for the Hierarchical Reasoning Model (HRM), a novel
  recurrent neural network architecture designed for complex reasoning tasks.

  Key Features:


   * Architecture: Inspired by the human brain, HRM uses two interdependent recurrent
     modules: a high-level module for abstract planning and a low-level module for detailed
     computations. This allows it to achieve significant computational depth with training
     stability and efficiency.
   * Performance: HRM, with only 27 million parameters, shows exceptional performance on
     reasoning tasks like Sudoku, maze solving, and the Abstraction and Reasoning Corpus
     (ARC) benchmark. It can be trained on a small number of examples (as few as 1000)
     without pre-training or Chain-of-Thought (CoT) data.
   * Efficiency: It can perform sequential reasoning tasks in a single forward pass, which is
     more efficient than the iterative nature of LLMs using CoT.
   * Datasets: The repository provides scripts to build and use several datasets, including:
       * ARC (Abstraction and Reasoning Corpus)
       * Sudoku
       * Mazes
       * A C# dataset (inferred from file names)
   * Functionality: The repository includes scripts for:
       * Dataset generation: dataset/build_*.py
       * Training: pretrain.py, train_csharp_hrm.py
       * Evaluation: evaluate.py, arc_eval.ipynb
       * Visualization: puzzle_visualizer.html
   * Reproducibility: The README.md provides detailed instructions for setting up the
     environment, installing dependencies (including specific CUDA and PyTorch versions), and
     running experiments. It also links to pre-trained model checkpoints.
   * Experiment Tracking: The project is integrated with Weights & Biases (W&B) for
     experiment tracking.


  In essence, this repository offers a complete toolkit for researchers and developers to
  explore, train, and evaluate the Hierarchical Reasoning Model on a variety of complex
  reasoning tasks. The model itself is presented as a significant advancement in AI
  reasoning, particularly for tasks requiring structured, multi-step problem-solving.

> What can you tell me about the novel architecture, and how does it differ from a typical 
  LLM architecture?

 ╭───────────────────────────────────────────────────────────────────────────────────────────╮
 │ ✔  ReadFile models/hrm/hrm_act_v1.py                                                      │
 ╰───────────────────────────────────────────────────────────────────────────────────────────╯
✦ Based on the information in the repository, the Hierarchical Reasoning Model (HRM) has a
  novel architecture that is fundamentally different from a typical Large Language Model
  (LLM). Here's a breakdown of its key architectural features and how they contrast with
  standard LLMs:

  HRM's Novel Architecture


  The HRM's architecture is inspired by the hierarchical and multi-timescale processing
  observed in the human brain. Its core is built around two interdependent recurrent modules:


   1. High-Level Module (`H_level`): This module is responsible for "slow" abstract planning.
      It operates on a longer timescale, taking the processed output from the low-level module
       to build a high-level understanding and plan.
   2. Low-Level Module (`L_level`): This module handles "rapid" detailed computations. It
      operates on a faster timescale, performing multiple computation cycles for each single
      cycle of the high-level module. It takes input from the high-level module as guidance
      for its detailed processing.


  This creates a feedback loop where the H_level plans and the L_level executes, and the
  results of the execution inform the next planning step. This entire reasoning process is
  encapsulated within a single "forward pass" of the model, where the model internally
  iterates through its reasoning steps.


  Another key feature is the use of Adaptive Computation Time (ACT). The model learns to
  decide how many computational steps are necessary for a given problem, rather than using a
  fixed number of layers for every input. This is achieved through a Q-learning mechanism
  that determines when to "halt" the computation.


  How HRM Differs from a Typical LLM


  | Feature                  | Hierarchical Reasoning Model (HRM)
                                                                   | Typical Large Language
  Model (LLM)
               |
  | ------------------------ | --------------------------------------------------------------
  ---------------------------------------------------------------- |
  -------------------------------------------------------------------------------------------
  ----------------------------------- |
  | Core Architecture    | Dual recurrent modules (High-level and Low-level) with internal
  transformer blocks. Explicitly hierarchical and recurrent.       | A deep stack of
  identical transformer decoder blocks. It's a feed-forward architecture (for a single token
  generation step). |
  | Processing           | Multi-timescale processing with nested loops. The high-level
  module updates slower than the low-level one.                       | Uniform processing.
  Each layer processes the input in the same way.
                  |
  | State Management     | Maintains an explicit recurrent state that is passed between
  computation steps, allowing for a persistent "memory".              | Maintains a key-value
   cache for the attention mechanism to avoid re-computation for previous tokens in a
  sequence.             |
  | Reasoning Mechanism  | Internal, structured reasoning process within a single "forward
  pass" through the recurrent updates of the H and L modules.    | Emergent reasoning through
   auto-regressive generation of text (e.g., Chain-of-Thought). Reasoning is externalized as
  text.   |
  | Computational Depth  | Achieves computational depth through recurrent updates and
  Adaptive Computation Time (ACT).                                    | Achieves
  computational depth by stacking a large number of transformer layers.
                            |
  | Training & Size      | Trained on smaller, task-specific datasets without the need for
  large-scale pre-training. It has a much smaller parameter count (27 million). | Typically
  pre-trained on massive amounts of text data, then fine-tuned. They are generally very large
   (billions of parameters). |


  In summary, while a typical LLM is a massive, feed-forward transformer that reasons by
  generating text, the HRM is a smaller, recurrent model with a specialized, hierarchical 
  architecture that performs reasoning internally through a structured, multi-timescale 
  computational process.



> How does the HRM's architecture differ from a "thinking" LLM model?

✦ That's an excellent question that gets to the heart of the difference in their design
  philosophies. When you say a "thinking" LLM, you're likely referring to models that use
  techniques like Chain-of-Thought (CoT) or similar methods to generate intermediate
  reasoning steps before providing a final answer.


  The fundamental difference is this: "Thinking" LLMs simulate a reasoning process, while 
  the HRM is architecturally designed to perform one.

  Here’s a more detailed comparison:


  | Aspect | "Thinking" LLM (e.g., with Chain-of-Thought) | Hierarchical Reasoning Model
  (HRM) |
  | :--- | :--- | :--- |
  | Mechanism of "Thought" | External & Text-Based: The "thinking" is an inference-time
  technique where the model is prompted to generate its reasoning steps as explicit text.
  These text-based thoughts are then fed back into the model as part of the context to
  guide it towards the final answer. | Internal & Vector-Based: The "thinking" is the core
   computational process of the model's architecture. It happens through the iterative,
  multi-step interaction between the high-level (planning) and low-level (computation)
  modules. The "thoughts" are represented as high-dimensional vectors (the internal states
   z_H and z_L), not as human-readable text. |
  | Architectural Basis | Standard LLM Architecture: These models use the same underlying
  transformer architecture as non-thinking LLMs. The "thinking" is a process layered on
  top, not a change to the core model structure. | Custom Reasoning Architecture: The
  HRM's architecture is explicitly designed for this internal reasoning process. The
  dual-recurrent, multi-timescale structure is the thinking mechanism. It's not an add-on;
   it's the foundation. |
  | Process | Sequential Generation: The process is auto-regressive. The model generates
  one word (or token) at a time, building out its chain of thought sequentially as a
  single stream of text. | Internal Iteration: The process is a series of internal updates
   to its hidden states. For a single problem, it performs many internal computation
  cycles between its H and L modules before producing a final output. |
  | Efficiency | High Latency & Data Needs: Generating long chains of thought can be slow
  and computationally expensive. Training these models to "think" effectively often
  requires massive datasets of human-generated reasoning examples. | High Efficiency: The
  reasoning happens in a single forward pass (encompassing all the internal iterations),
  which is more computationally efficient. It's designed to learn complex reasoning from a
   very small number of examples (as few as 1000, per the README). |


  In analogy:


   * A "thinking" LLM is like a person who has to "think out loud" by writing down every
     single step of their reasoning on a piece of paper to solve a problem. The paper
     itself is part of the process.
   * The HRM is like a person who can "think internally" in their head, manipulating
     abstract concepts and plans without having to speak or write them down. The reasoning
     is a silent, internal process.


  Therefore, the HRM's architecture represents a more direct and integrated approach to
  building a reasoning machine, whereas "thinking" LLMs are a clever adaptation of a
  powerful language-prediction architecture to simulate a reasoning process.

> Fascinating. Could you walk me through the code where this "thinking internally" happens, 
  and help me understand what it's doing?

✦ Of course. Let's walk through the core piece of code from models/hrm/hrm_act_v1.py that
  implements the "internal thinking" process.

  The magic happens inside the forward method of the HierarchicalReasoningModel_ACTV1_Inner
  class.

  The Setup


  First, let's identify the key actors in this process:


   * carry.z_H: The hidden state of the High-level (H) module. This is the "slow, abstract
     plan."
   * carry.z_L: The hidden state of the Low-level (L) module. This is the "rapid, detailed
     computation."
   * input_embeddings: The puzzle or problem, converted into a numerical representation
     (embedding) that the model can understand.
   * self.H_level: The High-level reasoning module.
   * self.L_level: The Low-level reasoning module.
   * self.config.H_cycles: The number of times the H-module will update.
   * self.config.L_cycles: The number of times the L-module will update for each H-module
     update.


  The Core Reasoning Loop

  Here is the most critical section of the code. This is the "thinking" part.



    1 # from lines 200-212 of models/hrm/hrm_act_v1.py
    2 
    3         # Forward iterations
    4         with torch.no_grad():
    5             z_H, z_L = carry.z_H, carry.z_L
    6 
    7             for _H_step in range(self.config.H_cycles):
    8                 for _L_step in range(self.config.L_cycles):
    9                     if not ((_H_step == self.config.H_cycles - 1) and
      (_L_step == self.config.L_cycles - 1)):
   10                         z_L = self.L_level(z_L, z_H + input_embeddings,
      **seq_info)
   11 
   12                 if not (_H_step == self.config.H_cycles - 1):
   13                     z_H = self.H_level(z_H, z_L, **seq_info)
   14 
   15         assert not z_H.requires_grad and not z_L.requires_grad



  Let's break down what's happening in this nested loop:


   1. Outer Loop (`for _H_step...`): This is the "slow" thinking process of the high-level
      planner. It iterates H_cycles times.
   2. Inner Loop (`for _L_step...`): This is the "fast" thinking process of the low-level
      computation module. For each single step of the H-planner, this inner loop runs
      L_cycles times.
   3. Low-Level Computation (`z_L = self.L_level(...)`):
       * The L-module (self.L_level) takes its own previous state (z_L) as input.
       * Crucially, it also receives guidance from the H-module's plan (z_H) and the original
          problem (input_embeddings). This combination (z_H + input_embeddings) acts as the
         context or instruction for the detailed computation.
       * The L-module then updates its state. You can think of this as the model doing
         rapid, detailed work, like trying out a small calculation or a micro-step in the
         puzzle.
   4. High-Level Planning (`z_H = self.H_level(...)`):
       * After the L-module has completed all its rapid iterations, the H-module
         (self.H_level) takes the final state of the L-module (z_L) as its input.
       * It processes this information to update its own high-level plan (z_H). This is like
         the planner observing the results of the detailed work and updating its overall
         strategy.
   5. The Feedback Loop: The updated plan (z_H) is then fed back to the L-module at the
      start of the next outer loop iteration, providing new guidance for the next round of
      detailed computations. This cycle of Plan -> Execute -> Observe -> Revise Plan is the
      essence of the internal reasoning process.

  Why torch.no_grad()?


  You'll notice this entire thinking loop is wrapped in with torch.no_grad():. This is a
  performance optimization. It tells the program not to keep track of all the
  intermediate calculations for backpropagation (the process of learning). If it tracked
  every single step in the nested loop, the memory requirements would be enormous.


  So, how does the model learn? It does so in the very next two lines:



   1 # from lines 215-216 of models/hrm/hrm_act_v1.py
   2 
   3         # 1-step grad
   4         z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
   5         z_H = self.H_level(z_H, z_L, **seq_info)



  After the full iterative "thinking" process is complete, the model performs just one 
  final step of the L-module and H-module updates with gradient tracking enabled. This is
  a technique similar to Truncated Backpropagation Through Time (TBPTT). It allows the
  model to learn and adjust its weights based on the final "thought" it had, making the
  training process computationally feasible while still capturing the result of the deep,
  iterative reasoning process.


  Finally, the output of the H-module (z_H) is used to produce the actual answer to the
  puzzle.


  In summary, this code creates a structured, internal dialogue between a high-level
  planner and a low-level executor, allowing the model to "think through" a problem by
  iteratively refining its internal state before producing an answer.

> Interesting. I see that the number of cycles for each level is determined by a config 
  property (self.config.H_cycles and self.config.L_cycles) - where are those values being 
  set? How does the model know how many iterations are going to be required at any given 
  level?

 ╭────────────────────────────────────────────────────────────────────────────────╮
 │ ✔  SearchText 'cycles' in **/*.yaml within config                              │
 │                                                                                │
 │    Found 2 matche(s)                                                           │
 ╰────────────────────────────────────────────────────────────────────────────────╯
 ╭──────────────────────────╮
 │ ✔  ReadFile pretrain.py  │
 ╰──────────────────────────╯
✦ That's an excellent and insightful question. It touches on a key distinction between a
  fixed architectural choice and a learned, dynamic process. You're right to notice that
  H_cycles and L_cycles are being read from a config.

  Let's break it down into two parts.


  1. Where are H_cycles and L_cycles set?

  You are correct, these are static hyperparameters, not values that change dynamically
  based on the input. They define the fixed computational depth of a single reasoning
  step.

  My search found that their default values are set in the architecture's configuration
  file:


  File: /Users/garrettolson/Repos/HRM/config/arch/hrm_v1.yaml


   1 # ... other parameters
   2 H_cycles: 2
   3 L_cycles: 2
   4 # ... other parameters



  This means that by default, for every one cycle of the high-level planner, the
  low-level computation module will perform two cycles. The high-level planner will do
  this a total of two times.


  These values are loaded by the pretrain.py script when the model is created. As seen in
  the README.md, these can also be overridden from the command line for specific
  experiments, for example:


   1 # This command from the README overrides the default L_cycles
   2 ... pretrain.py ... arch.L_cycles=8 arch.halt_max_steps=8


  2. How does the model know how many iterations are required?


  This is the crucial part. The model doesn't change `H_cycles` or `L_cycles`. Instead,
  it decides how many times to run the entire reasoning block.


  Think of H_cycles and L_cycles as defining the "depth of a single thought." The model's
  architecture has a fixed number of internal cycles, much like a standard LLM has a fixed
   number of layers.


  The dynamic part—how the model adjusts the total amount of computation for a given
  problem—is handled by the Adaptive Computation Time (ACT) mechanism, which acts as a
  wrapper around the core reasoning loop.

  Let's look at the code for the wrapper class, HierarchicalReasoningModel_ACTV1:



    1 # from lines 258-287 of models/hrm/hrm_act_v1.py
    2 
    3     def forward(self, carry: HierarchicalReasoningModel_ACTV1Carry, batch:
      Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1Carry,
      Dict[str, torch.Tensor]]:
    4         # ...
    5 
    6         # Forward inner model (this is the reasoning loop we discussed)
    7         new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self
      .inner(new_inner_carry, new_current_data)
    8 
    9         # ...
   10 
   11         with torch.no_grad():
   12             # Step
   13             new_steps = new_steps + 1
   14             is_last_step = new_steps >= self.config.halt_max_steps
   15 
   16             halted = is_last_step
   17 
   18             # if training, and ACT is enabled
   19             if self.training and (self.config.halt_max_steps > 1):
   20                 # Halt signal
   21                 halted = halted | (q_halt_logits > q_continue_logits)


  Here's what's happening:


   1. The `forward` method of the `inner` model is called. This runs the complete,
      fixed-cycle (H_cycles and L_cycles) reasoning process once.
   2. A decision is made. The model uses a specialized q_head to output two values:
      q_halt_logits and q_continue_logits.
   3. To halt or not to halt? The model checks if q_halt_logits > q_continue_logits. If the
      "halt" score is higher than the "continue" score, it means the model is confident it
      has "thought" enough and has reached a solution.
   4. Repeat or Finish: If the model decides to continue, this entire forward method of the
      wrapper class is called again in the next step, running the whole internal dialogue
      again with the updated state. This continues until the model decides to halt, or it
      hits the maximum number of steps allowed (self.config.halt_max_steps).


  In short:


   * `H_cycles` / `L_cycles` (Static): The number of internal iterations in one reasoning
     step. This is a fixed part of the architecture.
   * ACT Steps (Dynamic): The number of times the entire reasoning step is repeated. This
     is a learned behavior where the model can choose to finish early.

> I see. How, then, does the model determine its confidence score (q_halt_logits) to see if 
  it has come up with an acceptible answer?

✦ Excellent question. The model doesn't have an innate sense of confidence. Instead, it
  learns to predict its own success through a process inspired by reinforcement learning,
  specifically Q-learning.


  The confidence score, q_halt_logits, is the output of a small, dedicated neural network
  layer called q_head. Let's trace how it works.

  1. The Decision-Making Layer (q_head)

  Inside the HierarchicalReasoningModel_ACTV1_Inner class, there's a simple linear layer
  defined:



   1 # from line 116 of models/hrm/hrm_act_v1.py
   2 self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)



  This is the "decision-maker." It takes a high-dimensional vector (size hidden_size) and
  outputs just two numbers, which are our logits: one for "halt" and one for "continue".

  2. Input to the Decision-Maker


  What does the q_head look at to make its decision? It looks at the final, high-level
  plan from the reasoning loop.


   1 # from line 221 of models/hrm/hrm_act_v1.py
   2 # Q head
   3 q_logits = self.q_head(z_H[:, 0]).to(torch.float32)



  After the model has finished its internal "thinking" (H_cycles and L_cycles), the final
  state of the high-level planner (z_H) represents the model's complete understanding of
  the problem and its proposed solution. A specific part of this state (z_H[:, 0], likely
  corresponding to a special classification token) is fed into the q_head.

  So, the decision to halt is based on the most abstract and comprehensive summary of the
  reasoning process.

  3. Learning to be Confident (Q-Learning)


  This is the most important part. The q_head isn't pre-programmed; its weights are
  learned during training. The model is trained to make q_halt_logits a good predictor of
  whether halting is the best action to take to get the right answer.

  Here’s a simplified view of the learning process:


   1. State: The model's current understanding (z_H) is its "state".
   2. Action: It can take one of two "actions": HALT or CONTINUE.
   3. Q-Value: The q_head predicts the "Q-value" for each action. This value represents the
      expected future reward (i.e., the probability of getting the puzzle right) if it takes
      that action from the current state.
       * q_halt_logits is the predicted reward for halting now.
       * q_continue_logits is the predicted reward for continuing to think for another step.
   4. Training Signal: The model learns by comparing its prediction for q_continue_logits
      with a "target" value. The code calculates this target in the
      HierarchicalReasoningModel_ACTV1 wrapper:




   1     # from lines 289-292 of models/hrm/hrm_act_v1.py
   2     next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry,
     new_current_data)[-1]
   3 
   4     outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step,
     next_q_halt_logits, torch.maximum(next_q_halt_logits,
     next_q_continue_logits)))

      This target essentially tells the model: "The value of continuing from your current
  state should be equal to the best possible value you can get in the next state (whether
  that's by halting or continuing again)."


   5. Learning: A loss function (defined elsewhere in the project) will compare the model's
      prediction (q_continue_logits) with this calculated target (target_q_continue). The
      difference (the error) is used to update the weights of the q_head via
      backpropagation.

  Over thousands of training examples, the q_head gets better and better at predicting
  whether more computation is likely to improve the answer.


  In essence, the model's confidence (q_halt_logits) isn't a simple score. It's a learned 
  Q-value that represents the model's prediction of its own success if it stops thinking 
  and submits its current answer.









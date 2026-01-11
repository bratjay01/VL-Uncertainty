**Task 1: Evaluation of VL_Uncertainty**

VL_Uncertainty was evaluated using the InternVL2-1B model on the LLaVABench benchmark. The code executed successfully and achieved a hallucination detection accuracy of 63.33%. The corresponding JSON log file is available in the /exp folder. No major issues were encountered during the setup or execution of the codebase. The Flash_attn module was omitted due to hardware limitations, as the experiments were conducted on a laptop.

**Task 2: Logical Inconsistency Integration**

The logical inconsistency detection method was successfully integrated into the VL_Uncertainty framework. Experimental results and logs for this task are stored in the /exp folder. The logical inconsistency method can be found in /util directory. Add these arguments to the existing command in the file "run_LLaVABench" to run logical reasoning.
```bash
--enable_logical_inconsistency --inconsistency_threshold 0.2
```

**Task 3**

I chose to implement weights for contradictions involving the main claim under the assumption that the contradictions between the initial claim and verification answers are more informative about hallucination than contradictions between two verification answers. If the model contradicts itself on the main claim, that's a strong signal of instability.
Right now, all contradiction pairs are treated equally. This means a contradiction between two verification statements (e.g., "existence question answer" vs "attribute question answer") carries the same weight as a contradiction between the main claim and a verification answer. When the inconsistency_rate treats all pairs equally, so rare but highly important contradictions (claim ↔ verification) might be diluted by many neutral verification↔verification pairs.
The weighted approach directly targets hallucination when the model changes its core assertion under probing. 
An improvement of approx 7% in the hallucination detection accuracy is observed from 38.3 to 45.( I only ran the code once for each experiment, hence the metric values cant be that reliable. As I did not take an average of multiple runs owing to time constraints) 


**Task 4**

**Core Problem: Self-consistent hallucinations and confirmation bias in self-probing methods.**

*(All the limitations are based on the results when the InternVL2-1B model was utilized. It is assumed that these limitations remain consistent with increase in model size for the sake of this assignment.)*

The LVLM can generate a coherent but incorrect internal narrative that remains logically consistent under probing. As a result, logical inconsistency metrics fail to detect hallucinations in these cases.

This issue arises due to two main factors:

Weak verification prompts: Automatically generated verification questions (e.g., existence or attribute queries) often fail to challenge the hallucinated concept. Instead, they reinforce it. For example, if an image of berries is incorrectly identified as plums, verification questions such as “Are there plums?” or “How many plums are there?” further strengthen the incorrect assumption and degrade performance.

Self-referential probing loop: The sequential dependency of the probing process causes verification questions to be derived from the hallucinated initial answer, introducing confirmation bias. Consequently, the quality of the verification probes is directly dependent on the quality of the initial response.

As a result, self-probing methods may confidently validate incorrect predictions without exposing hallucinations.

If I were to continue working on this method, I would work on:
1.  Adversarial probing strategy: Generate questions specifically designed to challenge the initial answer:

"If this is [claimed object], what should we see?" then verify if those visual properties exist
Counterfactual questions: "What would indicate this is NOT [claimed object]?"
Comparative questions: "Is this more like [similar object A] or [similar object B]?"

2. Ablation studies: Systematically evaluate the contribution of each component (weighted contradictions, prompt types, external grounding)

3. Parallel architecture structure:

Instead of sequential dependencies, generate multiple independent reasoning paths from the start ( Image grounding path, counterfactual path, logical reasoning etc). This has confirmation bias because each step depends on the previous (potentially hallucinated) output. Cross-path contradictions are stronger signals than within-path. Even if one path has weak questions, other paths might provide independent validation.

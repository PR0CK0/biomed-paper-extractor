## Task 1 — Figure Digitization (VLM)

* Need to specify the exact type of figure... pie, line, scatter, histogram, bar, distribution, table, diagram, abstract, etc... These are your "fine tune" datasets for a better VLM setup.
* Visually complex images with lots of text fail to finish with certain models, especially smaller ones
* In general, models with 5B+ parameters succeed at outputting digitized image information; smaller ones (< 3B params) can fail on complex images, especially those with multiple panes and lots of text

* This is all a very complex task centering around the balancing of VLM speed and accuracy... finding the "middle ground" model is difficult and requires much investigation across a diverse paper set — not a small handful of papers
* PICKING the right model is the hard part... really it depends on the types of figures you encounter. The larger Qwen models are exceptional at extracting JSON from papers, but they can be slow for larger, complex images.

* Fine-tuning is appropriate but requires curating a dataset of "difficult" images across categories (complex pie charts, dense graphs, multi-pane figures, etc.). This can take weeks to curate effectively for LoRA. Fine-tuning will not necessarily help immediately — it simply requires more testing on a much larger and more diverse paper set to determine model performance.

* Figures are processed sequentially (one at a time) to keep the app simple and avoid GPU memory contention. Parallelizing VLM inference across figures would dramatically cut total processing time — a paper with 12 figures at ~2s each takes ~24s serially but could theoretically finish in ~2s with full parallelism. The tradeoff is complexity around concurrent GPU access, partial result streaming, and error handling per figure.

* It may be beneficial to use a fast supervised ML model like YOLO (tuned for diagrams) to pre-pass and extract individual panes from multi-pane images (e.g., images with parts A, B, C, etc.). Larger VLMs can detect and extract individual panes themselves, but it is a slower process.

## Task 2 — NER

* NER is a much simpler task and really just depends on the term set you want to see. It is fast — 1-2s per paper.
* GLiNER is useful because you specify your set of terms up-front, while the other models have rigid pre-defined term sets.
* The scispaCy NER models are nice because they automatically link entities to UMLS CUIs — though this can be done for the other models with a lookup table.
* NER is nowadays something of a fast and largely solved task; model selection really just depends on the set of terms you want to output.

## Dynamic Task Selection

* Currently the app runs both VLM and NER together — it would be useful to add a toggle so users can run just one task (e.g., only NER for fast entity extraction without touching figures, or only VLM when NER output isn't needed).
* Separating the tasks also opens the door to adding new task types in the future (e.g., citation graph extraction, methods section parsing, statistical result scraping) without requiring changes to existing task logic.
* A task selection UI (checkboxes or a multi-select) is a natural fit here — run selected tasks, skip the rest. This keeps the pipeline modular and lets users avoid unnecessary compute.
* Each task could be defined as a self-contained module with a standard interface (input: paper, output: structured result), making it straightforward to plug in new tasks or swap out implementations.

## General

* It is possible to digitize any image with very good accuracy — it just requires a larger model and more inference time/cost. This is fundamentally a speed vs. accuracy tradeoff.
* Claude was used to construct this. It strung together a lot of code written in previous projects. In the timeframe given, manually constructing this piece was not feasible.

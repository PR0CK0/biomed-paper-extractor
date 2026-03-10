* Need to specify the exact type of figure... pie, line, scatter, histogram, bar, distribution, table, diagram, abstract, etc... These are your "fine tune" datasets for a better VLM setup.
* Visually complex images with lots of text fail to finish with certain models, especially smaller ones

* This is all a very complex task centering around the balancing of VLM speed and accuracy... finding the "middle ground" model is difficult and requires much investigation across a diverse paper set
* PICKING the right model is the hard part... really it depends on the types of figures you encounter. So might need a very fast pre-pass model to do boundaries or general detection to send into the heavy lifting model.

* NER is the simpler task and really just depends on the term set you want to see. Lots of options there, but it's very fast, like 1-2s per paper.

* Figures are processed sequentially (one at a time) to keep the app simple and avoid GPU memory contention. Parallelizing VLM inference across figures would dramatically cut total processing time — a paper with 12 figures at ~2s each takes ~24s serially but could theoretically finish in ~2s with full parallelism. The tradeoff is complexity around concurrent GPU access, partial result streaming, and error handling per figure.
* Need to specify the exact type of figure... pie, line, scatter, histogram, bar, distribution, table, diagram, abstract, etc... These are your "fine tune" datasets for a better VLM setup.
* Visually complex images with lots of text fail to finish with certain models, especially smaller ones

* This is all a very complex task centering around the balancing of VLM speed and accuracy... finding the "middle ground" model is difficult and requires much investigation across a diverse paper set
* PICKING the right model is the hard part... really it depends on the types of figures you encounter. So might need a very fast pre-pass model to do boundaries or general detection to send into the heavy lifting model.

* NER is the simpler task and really just depends on the term set you want to see. Lots of options there, but it's very fast, like 1-2s per paper.
# DANCER+BART for scientific papers summarization

DANCER is a simple yet efficient approach for scientific papers summarization. It first breaks a long paper into several sections thanks to their well structures. In the original paper, the authors employ a simple Pointer-Generator network to abstractively summarize each section, which are concatenated in the final stage to create a complete summary. This architecture is called DANCER+LSTM.

DANCER+BART employs the same idea of divide-and-conquer, yet replace the Pointer-Generator by the state-of-the-art BART in hope to improve the quality of partial summaries.

You can find its details in this [blog](https://medium.com/@cinnamonai/bootcamp-tech-blog-4-long-document-summarization-6bc25e3add94) and my [presentation](https://www.facebook.com/watch/?v=676262983225151) ([slide](https://drive.google.com/file/d/1dwCDQXQs5bZnlEI_CENDMcIf39_lQHZH/view?usp=sharing)).

```
@article{Gidiotis2020ADA,
  title={A Divide-and-Conquer Approach to the Summarization of Academic Articles},
  author={Alexios Gidiotis and Grigorios Tsoumakas},
  journal={ArXiv},
  year={2020},
  volume={abs/2004.06190}
}
```

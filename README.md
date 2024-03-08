# Abstract
In the current era of online publication, the publication rate of online news articles has grown exponentially. The classification of news articles is crucial for easier comprehension of online content. Without proper categorization, readers often struggle to navigate through the vast amounts of news available to them. This paper is devoted to the single-label semantic classification on the combined dataset of BBC All Time and Reuters News. We conducted comparative analyses between SWIFT and established models such as BERT, alBERT, and DistilBERT. Our experiments demonstrate that SWIFT surpasses these generic models on our English News Dataset, yielding superior classification performance, particularly in the categorization of news titles. Comparison with the state-of-the-art research shows we can consider SWIFT as a baseline for future investigations of analysis of news datasets.

## Keywords
NLP, Text Classification, BERT, DistilBERT, alBERT, Transformers, Semantic Understanding

![Proposed Methodology Architecture](images/model.png)
*Fig. 1: Proposed Methodology Architecture*

## Introduction
Topical news classification involves categorizing news articles into thematic classes. A key aspect of this task is single-label news classification, which has seen various solutions using standard text features and deep learning techniques. However, these methods often have huge variations in results across different languages and datasets, and applications limiting their generalized use. [1, 2]

Recently, the BERT language model [3] has emerged as a promising approach capable of achieving high-quality results across various semantic text processing tasks, including news classification. BERT has demonstrated its effectiveness in analyzing English news across diverse datasets. While much attention has been given to BERT's performance in English news analysis, there is still a need for further enhancing its capabilities across different datasets and tasks.

The paper follows a structured approach. In Section I, we look into the current literature on topical classification, exploring methodologies employing BERT alongside other prevalent text features. Section II outlines our research methodology, the tailored text dataset, feature extraction techniques, and experimental design.

In Section III, we present the outcomes of the experiments conducted with varied text models. Finally, in Section IV the comparative results across the proposed and existing models summarize the key findings of our study and suggest avenues for future exploration.

### Related Work
News classification, a fundamental task in Natural Language Processing (NLP), involves assigning labels or classes to text based on its content. [4] It begins with a dataset where class assignments are known, and the aim is to correctly predict the target class (i.e., category) for each news item. Unlike regular texts, news articles are continuously generated, making their categorization challenging. However, this classification facilitates easier access and navigation through a diverse range of articles in real time for users. Machine learning is the primary approach for implementing news classification, with supervised learning models such as Naïve Bayes Classifier [5], K-Nearest Neighbors [6], and Support Vector Machines [7] being commonly utilized. While these methods offer basic accuracy, the complexity arises when a news article may belong to more than one category, necessitating careful consideration. Techniques like Hierarchical Multilabel Classification [8], Semi-supervised learning [9], Bayesian Networks [10], and clustering algorithms have been employed to address such challenges.

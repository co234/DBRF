# DBRF
The repository for the implementation of "De-biased Representation Learning for Fairness"


# Abstract
Existing fair representation learning methods for discrimination removal always assume the observed labels are reliable and require the learned representations to be predictable. However, when labels are contaminated by human bias, the predictiveness is improperly measured, which does not guarantee the fairness of predictions based on the learned representations. To circumvent this issue, we explore the following research question: \emph{Can we make the representations predictable to the ideal labels when we only have access to unreliable labels?} To this end, we propose a \textbf{D}e-\textbf{B}iased \textbf{R}epresentation Learning for \textbf{F}airness (DBRF) framework that disentangles the sensitive information from non-sensitive attributes whilst keeping the learned representations predictable to the ideal fair labels instead of the observed biased ones. We formulate the de-biased learning framework through information-theoretic concepts such as mutual information and information bottleneck to penalize the unfair samples. The key idea is that DBRF advocates not to use unreliable labels for supervision when the sensitive information benefits the prediction of unreliable labels. Experimental results over both synthetic and real-world data demonstrate that DBRF effectively learns de-biased representations towards ideal labels.


# Instructions
Please run the command line below for detailed guide:
```
python vae_model/train.py --help 
```
Run code:
```
./train.sh
```

See the sh file for more details.




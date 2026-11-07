%%
%% This is file `sigconf.tex',
%%
%% Commands for TeXCount
%TC:macro \cite [option:text,text]
%TC:macro \citep [option:text,text]
%TC:macro \citet [option:text,text]
%TC:envir table 0 1
%TC:envir table* 0 1
%TC:envir tabular [ignore] word
%TC:envir displaymath 0 word
%TC:envir math 0 word
%TC:envir comment 0 0

\documentclass[sigconf]{acmart}

\AtBeginDocument{%
  \providecommand\BibTeX{{%
    Bib\TeX}}}

\setcopyright{none}
\settopmatter{printacmref=false}
\acmConference[12S4054]{Proyek Sistem Rekomendasi}{Semester Gasal 2026/2027}{Laguboti, Indonesia}

\begin{document}

\title{The Impact of User Interaction History on Collaborative, Content-Based, and Hybrid Recommendation Performance}

\author{Eunike Purba}
\email{iss23021@students.del.ac.id}
\affiliation{%
  \institution{Institut Teknologi Del}
  \city{Laguboti}
  \country{Indonesia}
}

\author{Anastasya Siahaan}
\email{iss23046@students.del.ac.id}
\affiliation{%
  \institution{Institut Teknologi Del}
  \city{Laguboti}
  \country{Indonesia}
}

\author{Yolanda Saragih}
\email{iss23050@students.del.ac.id}
\affiliation{%
  \institution{Institut Teknologi Del}
  \city{Laguboti}
  \country{Indonesia}
}

\author{Theresia Silaban}
\email{iss23051@students.del.ac.id}
\affiliation{%
  \institution{Institut Teknologi Del}
  \city{Laguboti}
  \country{Indonesia}
}

\renewcommand{\shortauthors}{Purba et al.}

\begin{abstract}
This study examines how user interaction history relates to recommendation performance across user activity groups. Using the Amazon Reviews 2023 Musical Instruments dataset with the provided temporal training and warm-validation split, we compare four recommendation approaches: a popularity baseline, a TruncatedSVD-based latent-factor model, TF-IDF content-based filtering, and a hybrid model combining the two latter approaches. Users are grouped by the number of training interactions into Low (1--5), Medium (6--8), and High (9 or more) activity groups. All approaches are evaluated using Recall@10 and NDCG@10 under a shared evaluation protocol. In the current experiments, the hybrid model achieves the highest overall Recall@10 (0.014567), while the popularity baseline achieves the highest overall NDCG@10 (0.009222). Group-level results show that performance varies with user activity and evaluation metric. These findings provide an empirical comparison of recommendation approaches under different levels of user interaction history. The results are limited to the current warm-validation setting and do not establish statistical significance.
\end{abstract}

\begin{CCSXML}
<ccs2012>
 <concept>
  <concept_id>10002951.10003317.10003347.10003350</concept_id>
  <concept_desc>Information systems~Recommender systems</concept_desc>
  <concept_significance>500</concept_significance>
 </concept>
</ccs2012>
\end{CCSXML}

\ccsdesc[500]{Information systems~Recommender systems}

\keywords{recommender systems, user-history sparsity, collaborative filtering, TF-IDF, hybrid recommendation}

\maketitle

% =============================================================================
% 2. INTRODUCTION
% =============================================================================

\section{Introduction}

Recommendation systems help users discover relevant items by ranking alternatives according to their likely interests. Their performance depends partly on the information available about each user. Users with extensive interaction histories provide more evidence of their preferences, whereas users with only a few interactions provide less information for estimating their interests. This limitation is commonly associated with user-history sparsity and may affect recommendation approaches differently.

Collaborative filtering uses patterns in user--item interactions to estimate preferences. In this study, the collaborative baseline uses Truncated Singular Value Decomposition (TruncatedSVD) to obtain latent representations from the user--item rating matrix. Content-based filtering instead represents items using their textual attributes and recommends items according to their similarity to a user's positive interaction history. The study implements this approach using Term Frequency--Inverse Document Frequency (TF-IDF). A hybrid model combines the scores from the latent-factor and content-based components, while a popularity-based method provides a reference baseline.

The effect of interaction history is important because the four approaches rely on different sources of information. Popularity uses aggregate positive interactions, the latent-factor approach learns from the user--item interaction matrix, and TF-IDF content-based filtering relies on item metadata and a user's positive history. The hybrid approach combines interaction-based and content-based signals. Comparing these methods across users with different amounts of training history can help identify whether one approach performs better than another under the evaluated conditions.

This study uses the Amazon Reviews 2023 Musical Instruments dataset with the provided temporal data split. The training data contain 427,957 interactions from 54,483 users. The experiments use the provided warm-validation data and retain the original split rather than shuffling or reallocating interactions. Users are grouped by the number of training interactions: Low (1--5), Medium (6--8), and High (9 or more). Recommendation performance is measured using Recall@10 and Normalized Discounted Cumulative Gain at 10 (NDCG@10) under a shared evaluation procedure.

The study addresses the following research questions:

\begin{itemize}
    \item RQ1: How does the number of user interactions relate to the performance of collaborative, content-based, and hybrid recommendation approaches?
    \item RQ2: Which recommendation approach achieves the most consistent performance across user groups with different levels of interaction history?
\end{itemize}

We hypothesize that models relying on interaction data, such as TruncatedSVD, improve as the number of training interactions grows, whereas the popularity baseline, which does not use individual histories, remains comparatively stable across groups. These hypotheses are examined descriptively in this study.

The contribution of this study is an empirical comparison of popularity-based, TruncatedSVD-based, TF-IDF content-based, and hybrid recommendation approaches across user-history groups. Rather than proposing a new recommendation algorithm, the study examines differences in Recall@10 and NDCG@10 across the defined activity groups using the same warm-validation protocol.

\section{Related Work}

\subsection{Recommendation Approaches}

Recommender systems are commonly grouped into collaborative,
content-based, and hybrid approaches \cite{adomavicius2005,cano2017}.
Collaborative filtering relies only on past user behavior, such as
ratings or purchases, and does not require item profiles. It is
often more accurate than content-based techniques but struggles with
new users and items \cite{koren2009}. Latent factor models, including
matrix factorization, characterize users and items by factors inferred
from rating patterns \cite{koren2009}. Content-based filtering instead
describes items by their attributes, builds a profile of the user's
interests, and compares items against that profile \cite{pazzani2007}.
For textual item descriptions, a standard representation is the
TF-IDF vector space model with cosine similarity \cite{pazzani2007}.

\subsection{Hybrid Recommendation}

Hybrid recommenders combine two or more strategies to benefit from
their complementary strengths, and the taxonomy of Burke
\cite{burke2002} distinguishes several hybridization classes. A
systematic review of 76 studies found that collaborative filtering
combined with content-based filtering is the most common pairing,
that weighted hybrids are the most frequent class, and that
cold-start and data sparsity are the most frequently addressed
problems \cite{cano2017}. The hybrid model in this study follows the
weighted class by linearly combining normalized scores.

\subsection{User Interaction History and Evaluation}

Sparse user histories are a central difficulty for collaborative
methods. Applying SVD directly to an incomplete rating matrix is
problematic, and filling missing entries by imputation can distort
the data \cite{koren2009}. Whether content-based or hybrid methods
mitigate this depends on the data and must be tested empirically,
which motivates grouping users by history length in this study.
Evaluation outcomes also depend on the task, data, and metrics
chosen \cite{herlocker2004}. We therefore apply Recall@10 and
NDCG@10 \cite{jarvelin2002} under one shared protocol for all models.

\section{Dataset and Problem Formulation}

\subsection{Dataset Description}

This study uses the Amazon Reviews 2023 Musical Instruments dataset with the provided 5-Core subset and absolute-timestamp split \cite{hou2024}. The training set contains 427,957 interactions from 54,483 users. The full validation set contains 41,284 interactions, while the warm-validation subset used in the experiments contains 30,360 interactions from 11,659 evaluable users. The test set contains 42,595 interactions and is reserved for final evaluation. It is not used for model selection or parameter tuning.

Each interaction contains a user ID, item ID, rating, and timestamp. Ratings of 4 or 5 are treated as positive interactions when constructing the popularity ranking and the positive user profiles for content-based filtering. Item metadata, including title, main category, categories, features, description, and basic content text, is used to represent item characteristics in the TF-IDF model.

The experiments use the provided candidate-item set, which contains 22,753 items. Items already observed in a user's training history are excluded from that user's recommendations. Users are grouped according to their number of training interactions: Low (1--5 interactions), Medium (6--8 interactions), and High (9 or more interactions). These fixed groups allow recommendation performance to be compared across different levels of user interaction history.

\subsection{Training Data Statistics}

The training set contains 54,483 users and 22,753 items, with a sparsity of 0.9997 and a mean rating of 4.45. The number of training interactions with ratings 1 to 5 is 17,486, 14,335, 28,578, 64,120, and 303,438, respectively. User activity is the total number of training interactions of a user, regardless of rating. Table~\ref{tab:groups} shows the resulting groups.

\begin{table}[t]
\caption{User activity groups defined on training interactions.}
\label{tab:groups}
\centering
\small
\begin{tabular}{lrrrr}
\toprule
Group & Range & Train users & Mean & Eval. users \\
\midrule
Low    & 1--5        & 22,117 & 4.25  & 6,105 \\
Medium & 6--8        & 18,197 & 6.76  & 2,364 \\
High   & $\geq 9$   & 14,169 & 14.89 & 3,190 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Problem Formulation}

The task is to rank candidate items for each evaluable user based on the user's training interaction history. Four recommendation approaches are compared: a popularity-based baseline, a Truncated Singular Value Decomposition (TruncatedSVD) latent-factor baseline, a TF-IDF content-based filtering baseline, and a hybrid model combining the latent-factor and content-based scores. All models use the same training data, candidate-item set, warm-validation subset, and evaluation procedure to support a consistent comparison.

For the popularity baseline, candidate items are ranked by the number of positive training interactions, defined as ratings of 4 or 5. The collaborative baseline estimates user-item preference scores using TruncatedSVD applied to the training interaction matrix. The content-based baseline represents item metadata using TF-IDF and ranks candidates by cosine similarity to each user's profile, constructed from positively rated training items. The hybrid model combines the normalized TruncatedSVD and TF-IDF scores with equal weights of 0.5.

The study evaluates recommendation quality using Recall@10 and Normalized Discounted Cumulative Gain at 10 (NDCG@10). Recall@10 measures the proportion of relevant validation items retrieved in the top ten recommendations, while NDCG@10 accounts for the positions of relevant items in the ranking. For the group analysis, users are categorized by the number of training interactions into Low (1--5), Medium (6--8), and High (9 or more). The comparison examines how the four approaches perform overall and across these user-activity groups. The test set is reserved for final evaluation and is not used in the current warm-validation comparison.

\begin{table}[t]
\caption{Interaction Data Used in the Experiments}
\label{tab:interaction_data}
\centering
\begin{tabular}{lrl}
\hline
Split           & Interactions & Purpose \\
\hline
Training        & 427,957      & Model fitting \\
Full validation & 41,284       & Reference split \\
Warm validation & 30,360       & Model evaluation \\
Test            & 42,595       & Final evaluation only \\
\hline
\end{tabular}
\end{table}

The dataset consists of training, validation, and test interactions from the Amazon Reviews 2023 Musical Instruments dataset. As shown in Table~\ref{tab:interaction_data}, the training set contains 427,957 interactions and is used to fit the recommendation models. The full validation set contains 41,284 interactions, of which 30,360 interactions belong to the warm-validation subset used for model evaluation. The test set contains 42,595 interactions and is reserved for final evaluation only. The experiments use the provided warm-validation data to compare the performance of the popularity-based, TruncatedSVD-based, TF-IDF content-based, and hybrid recommendation approaches. The test set is not used for model selection or parameter tuning.

\section{Method}

\subsection{Popularity Baseline}

The popularity baseline ranks candidate items according to the number of positive interactions in the training data, where an interaction is considered positive if its rating is at least 4. For each candidate item, the number of positive interactions is counted. Candidate items are then sorted in descending order of positive interaction counts, with item identifiers in ascending order used to break ties. The resulting global ranking is shared across users and is saved for evaluation. During evaluation, the ranking is assessed using the common evaluation procedure applied to all recommendation models.

\subsection{Collaborative Filtering Baseline}

The collaborative filtering baseline uses Truncated Singular Value Decomposition (TruncatedSVD) as a latent-factor approximation \cite{koren2009}. The training interactions are aggregated by taking the mean rating for each user--item pair and represented as a sparse user--item matrix. TruncatedSVD is applied to obtain user and item latent representations \cite{pedregosa2011}. The number of components is set to the minimum of 32 and one less than the smaller matrix dimension, with seven iterations and a random state of 42. No centering or normalization is applied to the matrix before decomposition, and a user not found in the matrix receives an empty ranking.

For each evaluation user, candidate-item scores are computed as the dot product between the user's latent representation and each candidate item's latent representation. Candidate items already observed by the user in the training data are excluded. The remaining items are ranked by descending predicted score, with item identifiers in ascending order used to break ties. Up to ten highest-ranked items are returned for evaluation.

\subsection{TF-IDF Content-Based Filtering}

The content-based filtering baseline represents items using their textual metadata. Available fields, including title, main category, categories, features, description, and basic content text, are concatenated into a single text representation. TF-IDF vectors are constructed using up to 100,000 features, English stop-word removal, a minimum document frequency of two, and unigram and bigram features \cite{pedregosa2011}. Duplicate item metadata entries are removed before vectorization. The vectorizer is fitted on all rows of the item metadata table, not only on training items.

For each evaluation user, a profile is constructed by averaging the TF-IDF vectors of items rated at least four in the training data. The resulting profile is normalized, and cosine similarity is computed between the user profile and each candidate item's TF-IDF vector. Candidate items are ranked in descending order of similarity, with item identifiers in ascending order used to break ties. Items already observed in the user's training history are excluded, and up to ten remaining items are recommended. Users without positive training items that have available metadata receive an empty recommendation list.

\subsection{Hybrid Recommendation Model}

The hybrid model combines the TruncatedSVD-based latent-factor approach and TF-IDF content-based filtering. The latent-factor component uses the sparse user--item rating matrix constructed from the training data, while the content-based component uses item metadata, including titles, categories, features, descriptions, and available basic content text.

For each evaluation user, the latent-factor score for each candidate item is computed as the dot product of the corresponding user and item representations. The content-based score is calculated using cosine similarity between the user's profile and the candidate item's TF-IDF vector. The user profile is obtained by averaging the TF-IDF vectors of items rated at least four in the training data.

Before combination, each component's scores are independently normalized using min--max normalization. The hybrid score is calculated as

\begin{equation}
s_{\mathrm{hybrid}}(u,i)
= \alpha\hat{s}_{\mathrm{MF}}(u,i)
+ (1-\alpha)\hat{s}_{\mathrm{CBF}}(u,i),
\end{equation}

where $u$ denotes a user, $i$ denotes a candidate item, $\alpha$ is the MF weight, and the normalized component scores are denoted by $\hat{s}_{\mathrm{MF}}$ and $\hat{s}_{\mathrm{CBF}}$. The initial value $\alpha=0.5$ is used in the current experiments; the optimal value will be selected from $\{0.3, 0.5, 0.7\}$ using warm-validation Recall@10 before the final experiment. The min--max normalization is applied per user, separately to the latent-factor and content-based score vectors over the candidate items, before items already seen by the user are masked.

Candidate items already observed by a user in the training data are excluded. The remaining items are ranked by descending hybrid score, with item identifiers in ascending order used to break ties. The ten highest-ranked items are selected for recommendation, and the resulting rankings are evaluated using the same evaluation procedure as the other baselines.

% =============================================================================
% 5. EXPERIMENTAL SETUP  — updated for CP3
% =============================================================================

\section{Experimental Setup}

The experiments use the provided training and warm-validation splits without reshuffling or
creating new splits. The training set is used to construct the recommendation models. The
warm-validation subset is used to compare ranking performance and to select hyperparameters.
The test set is reserved for final evaluation and is not used for model selection or parameter
tuning.

\subsection{Evaluation Protocol}

All four approaches use the provided warm-start candidate-item set containing 22,753 items.
Items observed in a user's training history are excluded from that user's recommendations.
Positive interactions are defined as ratings of 4 or 5. Evaluable users are taken from the
provided \texttt{evaluable\_users\_validation\_warm.csv}; a user with no remaining relevant
validation item after seen-item removal is not counted. The candidate set and evaluation
procedure are identical for all four approaches.

A validation item is considered relevant for a user if the user rated it 4 or 5.
Let $R_u$ be the set of relevant validation items of user $u$ and $L_u$ the ranked list of
ten recommended items. Then

\begin{equation}
\mathrm{Recall@10}(u)=\frac{|L_u\cap R_u|}{|R_u|},\quad
\mathrm{NDCG@10}(u)=\frac{\mathrm{DCG@10}(u)}{\mathrm{IDCG@10}(u)},
\end{equation}

where $\mathrm{DCG@10}(u)=\sum_{k=1}^{10}\mathrm{rel}_k/\log_2(k+1)$ with binary relevance
$\mathrm{rel}_k$ and $\mathrm{IDCG@10}$ uses $\min(|R_u|,10)$ relevant items at the top
positions. Reported scores are macro-averages over the 11,659 evaluable users.

User activity groups are determined from the number of training interactions using fixed
\texttt{pd.cut} bins, consistently applied across all models via \texttt{make\_user\_groups()}
in \texttt{baseline\_common.py}: Low (1--5 interactions), Medium (6--8), and High (9 or more).

\subsection{Leakage Rules}

To ensure a fair comparison free of data leakage, the following rules are enforced throughout
all experiments.

\begin{enumerate}
  \item Models are trained exclusively on training-set interactions; validation and test
        interactions do not enter the training process.
  \item The official temporal split is not reshuffled; validation and test interactions do
        not leak into training.
  \item User profiles and model parameters are built from training data only. Aggregate
        item fields \texttt{average\_rating} and \texttt{rating\_number} are not used as
        features because they may encode post-cutoff information.
  \item The candidate protocol is identical for all compared models; no candidate set is
        constructed to favour a particular approach.
  \item Hyperparameter tuning and model selection use only the validation set. The test set
        is not accessed before Week~13.
\end{enumerate}

\subsection{Hyperparameter Search Space}

Hyperparameter selection is performed by evaluating each candidate value on the warm-validation
set and choosing the configuration with the highest Recall@10. All search-space values and
fixed parameters are defined centrally in \texttt{src/config.py}, and every run is logged
to \texttt{results/experiment\_log.csv} with its run identifier, timestamp, git commit hash,
parameter values, and validation metrics to support full reproducibility. The global random
seed is fixed at 42. The run identifier follows the format
\texttt{<model>\_<param><value>\_seed<seed>} (e.g.\ \texttt{mf\_components64\_seed42}).
Table~\ref{tab:search_space} lists the parameters and search ranges.

\begin{table}[t]
\caption{Hyperparameter search space. Selection criterion: highest Recall@10 on
warm-validation. Fixed parameters: \texttt{random\_state=42}; \texttt{n\_iter=7} (MF);
\texttt{ngram\_range=(1,2)}, \texttt{min\_df=2} (TF-IDF); min--max normalization per
user (Hybrid).}
\label{tab:search_space}
\centering
\small
\begin{tabular}{llll}
\toprule
Model                & Parameter              & Search Space          & Status   \\
\midrule
Popularity           & ---                    & No tuning             & Fixed    \\
Matrix Factorization & \texttt{n\_components} & \{32, 64, 128\}       & To tune  \\
TF-IDF CBF           & \texttt{max\_features} & \{50\,000, 100\,000\} & To tune  \\
Hybrid               & $\alpha$ (MF weight)   & \{0.3, 0.5, 0.7\}    & To tune  \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Experiment Matrix}

Table~\ref{tab:experiment_matrix} lists all planned experiment runs. Tuning experiments
(E2--E4) are executed on the warm-validation set. The best configuration from each group
is used in the final comparison and the Track-A group analysis (E5).

\begin{table}[t]
\caption{Experiment matrix. All runs report Recall@10 and NDCG@10 on warm-validation.}
\label{tab:experiment_matrix}
\centering
\small
\begin{tabular}{llp{3.2cm}l}
\toprule
ID      & Model/Variant                   & Purpose                                  & Metric       \\
\midrule
E1      & Popularity                      & Shared baseline                          & R@10, N@10   \\
E2a--c  & MF ($n$=32/64/128)              & Tune collaborative baseline              & R@10, N@10   \\
E3a--b  & TF-IDF CBF (50k/100k)           & Tune content-based baseline              & R@10, N@10   \\
E4a--c  & Hybrid ($\alpha$=0.3/0.5/0.7)   & Tune hybrid; best E2 \& E3 config        & R@10, N@10   \\
E5      & Track-A Analysis                & Per-group analysis (Low/Med/High)        & R@10, N@10   \\
\bottomrule
\end{tabular}
\end{table}

\section{Results and Discussion}

The warm-validation results show different performance patterns across the four recommendation approaches. The hybrid model achieves the highest overall Recall@10 of 0.014567, followed by popularity at 0.013499, TruncatedSVD at 0.013094, and TF-IDF content-based filtering at 0.012568. In terms of NDCG@10, popularity achieves the highest score of 0.009222, followed by the hybrid model at 0.009165, TruncatedSVD at 0.007917, and TF-IDF content-based filtering at 0.007304.

These results indicate that the model with the highest Recall@10 is not necessarily the model with the highest NDCG@10. The hybrid approach retrieves a larger proportion of relevant items within the top ten, whereas the popularity baseline achieves slightly better ranking quality according to NDCG@10. The observed difference is descriptive and has not been tested for statistical significance.

Performance also varies across user-activity groups. The hybrid approach achieves the highest Recall@10 for the Medium group, while TruncatedSVD achieves the highest Recall@10 for the High group. Popularity achieves the highest Recall@10 for the Low group. These findings suggest that the relative performance of recommendation approaches depends on both the evaluation metric and the amount of interaction history available for each user. Table~\ref{tab:group_results} lists the group-level scores.

\begin{table*}[t]
\caption{Warm-validation results by user activity group. Bold marks the highest value in each column; Range is the highest minus the lowest group score of a model.}
\label{tab:group_results}
\centering
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrrrrrrrr}
\toprule
 & \multicolumn{4}{c}{Recall@10} & \multicolumn{4}{c}{NDCG@10} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Model        & Low    & Medium & High   & Range  & Low    & Medium & High   & Range  \\
\midrule
Popularity   & \textbf{0.0155} & 0.0133 & 0.0097 & 0.0058 & \textbf{0.0107} & 0.0084 & 0.0070 & 0.0037 \\
TruncatedSVD & 0.0125 & 0.0128 & \textbf{0.0144} & 0.0018 & 0.0074 & 0.0078 & 0.0090 & 0.0015 \\
TF-IDF CBF   & 0.0124 & 0.0149 & 0.0112 & 0.0037 & 0.0072 & 0.0091 & 0.0062 & 0.0030 \\
Hybrid       & 0.0137 & \textbf{0.0175} & 0.0141 & 0.0038 & 0.0082 & \textbf{0.0106} & \textbf{0.0099} & 0.0024 \\
\bottomrule
\end{tabular}
\end{table*}

The current findings are limited to the provided warm-validation setting. They do not establish that any approach will perform best on the reserved test set or in a deployed recommendation system. Final conclusions should be confirmed through the planned test evaluation after the experimental configuration is fixed.

\section{Limitations / Responsible Recommendation Considerations}

\textit{Replace this section with your Limitations / Responsible Recommendation Considerations.}

\section{Conclusion}

\textit{Replace this section with your Conclusion.}

\section{AI Usage Declaration}

\textit{Replace this section with your AI Usage Declaration.}

\section{Contribution}

\textit{Replace this section with the contribution statement for your group.}

\bibliographystyle{ACM-Reference-Format}
\bibliography{sample-base}

\end{document}

\endinput
%%
%% End of file `sigconf.tex'.

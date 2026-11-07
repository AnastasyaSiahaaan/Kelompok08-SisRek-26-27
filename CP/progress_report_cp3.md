%%
%% This is file `sigconf.tex',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% samples.dtx  (with options: `all,proceedings,sigconf')
%% 
%% IMPORTANT NOTICE:
%% 
%% For the copyright see the source file.
%% 
%% Any modified versions of this file must be renamed
%% with new filenames distinct from sigconf.tex.
%% 
%% For distribution of the original source see the terms
%% for copying and modification in the file samples.dtx.
%% 
%% This generated file may be distributed as long as the
%% original source files, as listed above, are part of the
%% same distribution. (The sources need not necessarily be
%% in the same archive or directory.)
%%
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
%%
%% The first command in your LaTeX source must be the \documentclass
%% command.
%%
%% For submission and review of your manuscript please change the
%% command to \documentclass[manuscript, screen, review]{acmart}.
%%
%% When submitting camera ready or to TAPS, please change the command
%% to \documentclass[sigconf]{acmart} or whichever template is required
%% for your publication.
%%
%%
\documentclass[sigconf]{acmart}
%%
%% \BibTeX command to typeset BibTeX logo in the docs
\AtBeginDocument{%
  \providecommand\BibTeX{{%
    Bib\TeX}}}

%% Rights management information.  This information is sent to you
%% when you complete the rights form.  These commands have SAMPLE
%% values in them; it is your responsibility as an author to replace
%% the commands and values with those provided to you when you
%% complete the rights form.
\setcopyright{none}
\settopmatter{printacmref=false}
\acmConference[12S4054]{Proyek Sistem Rekomendasi}{Semester Gasal 2026/2027}{Laguboti, Indonesia}

%%
%% Submission ID.
%% Use this when submitting an article to a sponsored event. You'll
%% receive a unique submission ID from the organizers
%% of the event, and this ID should be used as the parameter to this command.
%%\acmSubmissionID{123-A56-BU3}

%%
%% For managing citations, it is recommended to use bibliography
%% files in BibTeX format.
%%
%% You can then either use BibTeX with the ACM-Reference-Format style,
%% or BibLaTeX with the acmnumeric or acmauthoryear sytles, that include
%% support for advanced citation of software artefact from the
%% biblatex-software package, also separately available on CTAN.
%%
%% Look at the sample-*-biblatex.tex files for templates showcasing
%% the biblatex styles.
%%
%%
%% The majority of ACM publications use numbered citations and
%% references.  The command \citestyle{authoryear} switches to the
%% "author year" style.
%%
%% If you are preparing content for an event
%% sponsored by ACM SIGGRAPH, you must use the "author year" style of
%% citations and references.
%% Uncommenting
%% the next command will enable that style.
%%\citestyle{acmauthoryear}


%%
%% end of the preamble, start of the body of the document source.
\begin{document}

%%
%% The "title" command has an optional parameter,
%% allowing the author to define a "short title" to be used in page headers.
\title{The Impact of User Interaction History on Collaborative, Content-Based, and Hybrid Recommendation Performance}

%%
%% The "author" command and its associated commands are used to define
%% the authors and their affiliations.
%% Of note is the shared affiliation of the first two authors, and the
%% "authornote" and "authornotemark" commands
%% used to denote shared contribution to the research.

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


%%
%% By default, the full list of authors will be used in the page
%% headers. Often, this list is too long, and will overlap
%% other information printed in the page headers. This command allows
%% the author to define a more concise list
%% of authors' names for this purpose.
\renewcommand{\shortauthors}{Purba et al.}

%%
%% The abstract is a short summary of the work to be presented in the
%% article.
\begin{abstract}

This study examines how user interaction history relates to recommendation performance across user activity groups. Using the Amazon Reviews 2023 Musical Instruments dataset with the provided temporal training and warm-validation split, we compare four recommendation approaches: a popularity baseline, a TruncatedSVD-based latent-factor model, TF-IDF content-based filtering, and a hybrid model combining the two latter approaches. Users are grouped by the number of training interactions into Low (1--5), Medium (6--8), and High (9 or more) activity groups. All approaches are evaluated using Recall@10 and NDCG@10 under a shared evaluation protocol. In the current experiments, the hybrid model achieves the highest overall Recall@10 (0.014567), while the popularity baseline achieves the highest overall NDCG@10 (0.009222). Group-level results show that performance varies with user activity and evaluation metric. These findings provide an empirical comparison of recommendation approaches under different levels of user interaction history. The results are limited to the current warm-validation setting and do not establish statistical significance.

\end{abstract}

%%
%% The code below is generated by the tool at http://dl.acm.org/ccs.cfm.
%% Please copy and paste the code instead of the example below.
%%
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


%%
%% Keywords. The author(s) should pick words that accurately describe
%% the work being presented. Separate the keywords with commas.
\keywords{recommender systems, user-history sparsity, collaborative filtering, TF-IDF, hybrid recommendation}
%% A "teaser" image appears between the author and affiliation
%% information and the body of the document, and typically spans the
%% page.

%%
%% This command processes the author and affiliation and title
%% information and builds the first part of the formatted document.
\maketitle


% =============================================================================
% VISIBLE COURSE GUIDANCE -- DELETE BEFORE FINAL SUBMISSION
% =============================================================================

% =============================================================================
% 2. INTRODUCTION
% Recommended minimum: ~0.75 page (about 4--5 focused paragraphs).
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
\end{itemize} We hypothesize that models relying on interaction data, such as TruncatedSVD, improve as the number of training interactions grows, whereas the popularity baseline, which does not use individual histories, remains comparatively stable across groups. These hypotheses are examined descriptively in this study.

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

The experiments use the provided candidate-item set, which contains 22,753 items. Items already observed in a user's training history are excluded from that user's recommendations. Users are grouped according to their number of training interactions: Low (1–5 interactions), Medium (6–8 interactions), and High (9 or more interactions). These fixed groups allow recommendation performance to be compared across different levels of user interaction history.

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
Low & 1--5 & 22,117 & 4.25 & 6,105 \\
Medium & 6--8 & 18,197 & 6.76 & 2,364 \\
High & $\geq 9$ & 14,169 & 14.89 & 3,190 \\
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
Split & Interactions & Purpose \\
\hline
Training & 427,957 & Model fitting \\
Full validation & 41,284 & Reference split \\
Warm validation & 30,360 & Model evaluation \\
Test & 42,595 & Final evaluation only \\
\hline
\end{tabular}
\end{table}
The dataset consists of training, validation, and test interactions from the Amazon Reviews 2023 Musical Instruments dataset. As shown in Table~\ref{tab:interaction_data}, the training set contains 427,957 interactions and is used to fit the recommendation models. The full validation set contains 41,284 interactions, of which 30,360 interactions belong to the warm-validation subset used for model evaluation. The test set contains 42,595 interactions and is reserved for final evaluation only. The experiments use the provided warm-validation data to compare the performance of the popularity-based, TruncatedSVD-based, TF-IDF content-based, and hybrid recommendation approaches. The test set is not used for model selection or parameter tuning.

\section{Method}
\subsection{Popularity Baseline}
The popularity baseline ranks candidate items according to the number of positive interactions in the training data, where an interaction is considered positive if its rating is at least 4. For each candidate item, the number of positive interactions is counted. Candidate items are then sorted in descending order of positive interaction counts, with item identifiers in ascending order used to break ties. The resulting global ranking is shared across users and is saved for evaluation. During evaluation, the ranking is assessed using the common evaluation procedure applied to all recommendation models.
\subsection{Collaborative Filtering Baseline}
The collaborative filtering baseline uses Truncated Singular Value Decomposition (TruncatedSVD) as a latent-factor approximation \cite{koren2009} . The training interactions are aggregated by taking the mean rating for each user--item pair and represented as a sparse user--item matrix. TruncatedSVD is applied to obtain user and item latent representations \cite{pedregosa2011} . The number of components is set to the minimum of 32 and one less than the smaller matrix dimension, with seven iterations and a random state of 42. No centering or normalization is applied to the matrix before decomposition, and a user not found in the matrix receives an empty ranking.

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
= 0.5\hat{s}_{\mathrm{MF}}(u,i)
+ 0.5\hat{s}_{\mathrm{CBF}}(u,i),
\end{equation}

where \(u\) denotes a user, \(i\) denotes a candidate item, and the normalized component scores are denoted by \(\hat{s}_{\mathrm{MF}}\) and \(\hat{s}_{\mathrm{CBF}}\). Thus, both components receive equal initial weights, which have not yet been tuned on the validation set. The min--max normalization is applied per user, separately to the latent-factor and content-based score vectors over the candidate items, before items already seen by the user are masked. A user without a positive profile has zero content-based scores, and a user without latent factors has zero latent-factor scores. Because TruncatedSVD factorizes the matrix with unobserved entries treated as zero, it approximates latent-factor modeling rather than fitting a regularized model on observed ratings only \cite{koren2009}.

Candidate items already observed by a user in the training data are excluded. The remaining items are ranked by descending hybrid score, with item identifiers in ascending order used to break ties. The ten highest-ranked items are selected for recommendation, and the resulting rankings are evaluated using the same evaluation procedure as the other baselines.


\section{Experimental Setup}
The experiments use the provided training and warm-validation splits. The training set is used to construct the recommendation models, while the warm-validation subset is used to compare their ranking performance. The test set is reserved for final evaluation and is not used for model selection or parameter tuning.

All four approaches use the provided candidate-item set containing 22,753 items. Items observed in a user's training history are excluded from that user's recommendations. Positive interactions are defined as ratings of 4 or 5 for the popularity ranking and the positive user profiles used by content-based filtering. User activity groups are determined from the number of training interactions: Low (1--5), Medium (6--8), and High (9 or more).

A validation item is considered relevant for a user if the user rated it 4 or 5. Let $R_u$ be the set of relevant validation items of user $u$ and $L_u$ the ranked list of ten recommended items. Then
\begin{equation}
\mathrm{Recall@10}(u)=\frac{|L_u\cap R_u|}{|R_u|},\quad
\mathrm{NDCG@10}(u)=\frac{\mathrm{DCG@10}(u)}{\mathrm{IDCG@10}(u)},
\end{equation}
where $\mathrm{DCG@10}(u)=\sum_{k=1}^{10}\mathrm{rel}_k/\log_2(k+1)$ with binary relevance $\mathrm{rel}_k$. Reported scores are averages over the 11,659 evaluable users. The evaluable users are taken from the provided warm-validation user list; relevant validation items are those rated at least 4 that are candidate items not already seen in the user's training history, and a user with no remaining relevant item is not counted. Recall@10 is divided by the full number of remaining relevant items, and the ideal DCG uses $\min(|R_u|,10)$ relevant items. The same candidate set and procedure are used for all four approaches. To answer RQ2, consistency is measured as the difference between the highest and lowest group-level score of each model.

The hybrid model combines independently min--max-normalized TruncatedSVD and TF-IDF content-based scores with initial equal weights of 0.5. The TruncatedSVD baseline uses up to 32 components, seven iterations, and random state 42. The TF-IDF representation uses up to 100,000 features, English stop-word removal, a minimum document frequency of two, and unigram and bigram features.

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
Model & Low & Medium & High & Range & Low & Medium & High & Range \\
\midrule
Popularity & \textbf{0.0155} & 0.0133 & 0.0097 & 0.0058 & \textbf{0.0107} & 0.0084 & 0.0070 & 0.0037 \\
TruncatedSVD & 0.0125 & 0.0128 & \textbf{0.0144} & 0.0018 & 0.0074 & 0.0078 & 0.0090 & 0.0015 \\
TF-IDF CBF & 0.0124 & 0.0149 & 0.0112 & 0.0037 & 0.0072 & 0.0091 & 0.0062 & 0.0030 \\
Hybrid & 0.0137 & \textbf{0.0175} & 0.0141 & 0.0038 & 0.0082 & \textbf{0.0106} & \textbf{0.0099} & 0.0024 \\
\bottomrule
\end{tabular}
\end{table*}

The current findings are limited to the provided warm-validation setting. They do not establish that any approach will perform best on the reserved test set or in a deployed recommendation system. Final conclusions should be confirmed through the planned test evaluation after the experimental configuration is fixed.

\section{Limitations / Responsible Recommendation Considerations}
\paragraph{Template guidance (delete before submission).}
Minimum target: \textbf{0.25 page}. Discuss limitations that affect validity or
generalizability, such as using only one Amazon category, offline evaluation,
the rating $\geq 4$ relevance convention, metadata quality, or track-specific
trade-offs. Also discuss responsible-recommendation implications relevant to the
track, such as popularity concentration, diversity, calibration, or reinforcing
past preferences. ``Our laptop was slow'' is not a scientific limitation unless
it materially changed the experimental design.

\textit{Replace this guidance with your Limitations / Responsible Recommendation Considerations.}

\section{Conclusion}
\paragraph{Template guidance (delete before submission).}
Minimum target: \textbf{0.25 page}. In one or two compact paragraphs, restate the
\section{Contribution}

\textbf{Eunike Purba:} Implemented the centralized configuration module (\texttt{src/config.py}) defining all hyperparameter search spaces, default values, and random seed for reproducibility.

\textbf{Anastasya Siahaan:} Updated the three baseline scripts (\texttt{src/mf\_baseline.py}, \texttt{src/tfidf\_cbf\_baseline.py}, \texttt{src/hybrid\_baseline.py}) to read parameters from \texttt{config.py}; prepared the experiment matrix (\texttt{results/experiment\_matrix.csv}) and the progress report (\texttt{CP/progress\_report\_cp3.md}); verified all baselines run correctly with the new configuration.

\textbf{Yolanda Saragih:} Implemented the experiment logging utility (\texttt{src/experiment\_log.py}) that records each run's identifier, timestamp, git commit hash, parameters, and validation metrics to \texttt{results/experiment\_log.csv}.

\textbf{Theresia Silaban:} Implemented the hyperparameter tuning script (\texttt{src/tune\_hyperparameters.py}), which evaluates all search-space configurations for MF, TF-IDF CBF, and Hybrid on the warm-validation set and reports the best configuration per model.er.

\textit{Replace this guidance with your Conclusion.}

\section{AI Usage Declaration}
\paragraph{Template guidance (delete before submission).}
Minimum target: about \textbf{0.10 page}. State factually whether and how
generative AI tools were used, for example for brainstorming, language editing,
coding assistance, debugging, or not at all. The authors remain responsible for
originality, correctness, citations, analysis, code, and experimental results.

\paragraph{Example wording (delete before submission).}
\textit{Generative AI tools were used for language editing and debugging
assistance. They were not used to fabricate experimental results or references.
All AI-assisted outputs were reviewed and verified by the authors.}

\textit{Replace this guidance with your AI Usage Declaration.}

\section{Contribution}
\paragraph{Template guidance (delete before submission).}
Minimum target: about \textbf{0.15 page}. State the contribution of each group
member using concrete roles such as data preparation, baseline implementation,
track-specific method, experimentation, evaluation, visualization, literature
review, manuscript drafting/revision, reproducibility, and presentation. Avoid
``all members contributed equally'' unless it is accurate and individual roles
can still be explained during Week 15 questioning.

\paragraph{Example wording (delete before submission).}
\textit{Student A: data preparation and MF implementation; Student B: CBF and
visualization; Student C: track-specific experiment and evaluation; Student D:
literature review, manuscript revision, and reproducibility checks.}

\textit{Replace this guidance with the contribution statement for your group.}


\paragraph{References guidance (delete before submission).}
Minimum target: approximately \textbf{0.75 page}. For this course manuscript,
aim for roughly \textbf{8--15 strong references}. Prefer peer-reviewed or
authoritative scholarly sources and cite them because they support a claim.
Avoid using blogs, Medium posts, random tutorials, or ChatGPT as scholarly
references for algorithmic/scientific claims. Use BibTeX and include DOI when
available. The Rosni Lumbantoruan examples cited in the Related Work guidance
above are stored in \texttt{sample-base.bib}; delete any example citation you do
not actually use in the final manuscript.

\bibliographystyle{ACM-Reference-Format}
\bibliography{sample-base}

\end{document}
\endinput
%%
%% End of file `sigconf.tex'.
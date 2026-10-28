<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/banner-dark.png">
  <img alt="The University of Chicago, MS in Applied Data Science. ADSP 32014 course project." src=".github/assets/banner-light.png">
</picture>

# Conversation Analysis: What Makes a Chatbot Win a Human Preference Vote

> **Example project from Autumn 2025.** This is a copy of [RayNG123/Bayes_Final_Proj](https://github.com/RayNG123/Bayes_Final_Proj), shown with the authors' work and commit history intact. Their own description of the code is [REPORT.md](REPORT.md); their analysis and its interpretation are in the markdown cells of [main.ipynb](main.ipynb). Course staff wrote this README and made the edits listed under [Changes by course staff](#changes-by-course-staff), including the section on [what to learn from this project](#what-to-learn-from-this-project).

When two chatbots answer the same prompt side by side and a person picks the better one, which properties of the answers and of the models predict the vote? The team took the Chatbot Arena conversations released by LMSYS, built features for each side of every battle, and fit a Bayesian logistic regression in PyMC on the differences between the two sides. A second model lets the effects vary by the topic of the prompt.

## Summary

The data is 23,294 Arena battles from April to June 2023 with a declared winner, after ties are dropped, between 20 models of the time, from GPT-4 and Claude to 7-billion-parameter open models. For each side the notebook records a hand-coded model size, whether the model is open or proprietary, and whether its maker is a company, a university or a community; counts of turns and of tokens per turn; three formatting flags, code blocks, bullets and headings, read from the assistant's messages by a small OpenAI model; and the topic of the first prompt, one of 25 clusters found by K-means on OpenAI embeddings and named by a second OpenAI model. The modeling design is the right one for a head-to-head vote: every feature enters as the standardized difference between side A and side B, the outcome is whether A won, and the intercept measures position bias. With Normal priors and NUTS sampling, seven of eleven coefficients have 95 percent intervals clear of zero: being proprietary, writing longer answers, model size, bullets and code blocks help; a community or corporate maker, relative to an academic one, hurts. A second model crosses every feature with the 25 topics, 275 coefficients, and recovers some sensible stories, proprietary models dominate questions about US presidents and lose on sexualized roleplay because they refuse, alongside spikes the authors themselves call noise and overfitting.

## Team and contributions

Team "Hidden Layer Heroes", Autumn 2025.

- **Yusen Wu** ([@RayNG123](https://github.com/RayNG123)): all five commits, the worker modules, the notebook and its write-up.
- **Yifei Zhou**: named as co-author in the authors' README; their part of the work is not visible in the commit history, which was assembled on December 7, 2025.

## Getting started

There is no environment file. The authors list openai, numpy, pandas, scikit-learn, tqdm, matplotlib and nltk; the notebook also needs datasets, huggingface_hub, pymc, arviz and numpyro.

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install openai numpy pandas scikit-learn tqdm matplotlib nltk datasets huggingface_hub pymc arviz numpyro jupyter
mkdir -p results out
```

You need an OpenAI key in `OPENAI_API_KEY` and a Hugging Face account that has accepted the dataset's terms; the notebook calls `login()` at the top. Then run `main.ipynb` top to bottom. Be aware of the cost before you do: the formatting flags are two OpenAI calls per battle, about 46,000 in all, and the topics need an embedding of every prompt. The notebook caches the embeddings and writes the feature tables to `results/`, which is ignored by git, so the Bayesian part cannot be rerun from this repository without repeating those calls. The two models sample in a few minutes each with the numpyro backend.

## Data

Data files are never committed to this repository.

- **Source:** [lmsys/chatbot_arena_conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations) on Hugging Face: 33,000 conversations with pairwise human votes, collected on Chatbot Arena between April and June 2023, with the two models' identities, the full conversations and the vote.
- **Terms:** Creative Commons per the dataset card; the dataset is gated, so you accept its terms on Hugging Face before `load_dataset` will work. The conversations contain whatever users typed, including offensive material.
- **How to get it:** the notebook downloads it with `load_dataset` after a Hugging Face login.
- **Where to put it:** nowhere; it is read from the Hugging Face cache. The notebook's derived tables go to `results/`.

## Project structure

| Path | Contents |
|---|---|
| `REPORT.md` | The authors' README: the toolkit's modules and a typical workflow. There is no separate report; the analysis is narrated in the notebook. |
| `main.ipynb` | Loading, feature engineering, the formatting classifier, embedding and clustering, cluster labelling, the two PyMC models and their interpretation. |
| `utils.py` | Message parsing, first-prompt extraction, turn and token-length features. |
| `format_workers.py` | The formatting classifier: prompts gpt-5-nano for three binary flags over the assistant's messages, in a multiprocessing pool. |
| `embedding_workers.py` | Cached text-embedding-3-large embeddings and K-means topic clustering. |
| `cluster_workers.py` | Inertia sweep, archetype selection nearest each centroid, cluster labelling with gpt-5-mini. |

## Method

**Features per side.** Model size in billions of parameters from a hand-written table; open or proprietary; maker category, corporate, academic or community; number of turns, user turns and assistant turns; average user and assistant turn length in tokens; three formatting flags from the assistant's messages; the topic cluster of the first prompt.

**Formatting flags.** The assistant's messages, last 4,000 characters, go to gpt-5-nano with a prompt asking for a JSON object with `any_code_blocks`, `any_bullets` and `any_headings`, in a pool of 200 workers. Failures default to zero.

**Topics.** The first user prompt is embedded with text-embedding-3-large; K-means with 25 clusters after an inertia sweep from 5 to 95; the 30 prompts nearest each centroid are shown to gpt-5-mini, which returns a label of at most four words. The labels range from "Python code requests" and "Travel itineraries" to "German language requests" and "Sexualized roleplay requests".

**Pooled model.** Features are standardized with a scaler fit on both sides stacked, then differenced, A minus B. With $x$ the difference vector,

$$\text{logit}\,P(\text{A wins}) = \alpha + x^\top \beta, \qquad \alpha \sim \mathcal{N}(0, 1), \quad \beta_j \sim \mathcal{N}(0, 1).$$

NUTS through numpyro, four chains, 2,000 tuning and 2,000 draws, target acceptance 0.9. Effects are read from 95 percent highest-density intervals.

**Interaction model.** The same, with the difference vector multiplied by the one-hot topic of the prompt, so each feature gets a coefficient per topic, 275 in all, with $\beta_j \sim \mathcal{N}(0, 0.5)$ priors and no pooling across topics.

### What is Bayesian here

The models are Bayesian generalized linear models: a logistic likelihood, Normal priors on the coefficients that act as mild regularization, and a posterior explored by Hamiltonian Monte Carlo rather than a point estimate with standard errors. The authors use the posterior the way the course asks: an effect counts when its 95 percent interval excludes zero, and the interval's width is part of the result. Two choices in the design are worth copying. Differencing the two sides' features turns a paired comparison into a regression without losing the pairing, and the intercept becomes an estimate of position bias, the tendency to prefer whichever answer is shown first. The interaction model is where the course's hierarchical material was needed and not used: 275 coefficients with independent priors are free to chase noise in small topics, and the authors see exactly that. A partial-pooling prior, $\beta_{j,\text{topic}} \sim \mathcal{N}(\beta_j, \tau_j)$, would have shrunk each topic's effect toward the global one in proportion to how little data the topic has, which is the fix the course teaches for this situation.

## Results

From the notebook. Coefficients are on standardized differences, so a unit is one standard deviation of a feature across both sides.

| Feature, A minus B | Posterior mean | 95% HDI |
|---|---|---|
| Proprietary model | 0.725 | 0.688 to 0.761 |
| Average assistant turn length | 0.423 | 0.390 to 0.458 |
| Model size | 0.190 | 0.156 to 0.221 |
| Any bullets | 0.132 | 0.104 to 0.161 |
| Any code blocks | 0.126 | 0.088 to 0.165 |
| Community maker | −0.121 | −0.141 to −0.099 |
| Corporate maker | −0.295 | −0.325 to −0.267 |

Turn counts, user turn length and headings have intervals that include zero. In the interaction model, the largest positive effects are being proprietary on questions about US presidents, and almost everything in the word-puzzle and video-game clusters; the largest negative ones are being proprietary on sexualized roleplay and on travel itineraries. The notebook's own reading is that the first group is sensible and the second is noise.

## What to learn from this project

Course staff added this section. The design and the inference are sound; the points below are about what the features measure, what was left unchecked, and the lesson the second model was trying to teach.

- **Pool partially instead of not at all.** The interaction model's noise is not a mystery to be noted, it is the symptom hierarchical priors exist to treat. One extra level, topic effects drawn around the global effect, would have kept the real differences and shrunk the spurious ones, and a model comparison with LOO would have shown whether topics add anything.
- **Put the model's identity in the model.** Every feature here is a proxy for which of 20 models is answering, and model identity is never a term. A per-model effect, or the Bradley–Terry model that Arena itself uses for its leaderboard, would separate "GPT-4 and Claude were the best models in mid-2023" from "proprietary models win". As fit, the proprietary coefficient mostly says the former.
- **Validate features you got from an LLM.** Code fences, bullets and headings are exact regular-expression matches. Asking a language model for them cost 46,000 calls, introduced errors that default silently to zero, and was never checked against a sample. When an LLM is the right tool for a feature, measure its accuracy on a hundred hand-labelled rows first.
- **Don't invent data.** The size table gives GPT-4 a thousand billion parameters and Claude 175 billion. Neither was public. The size coefficient rests on guesses and overlaps with the proprietary flag.
- **Read coefficients on the scale you fit.** The notebook turns 0.725 into "about 2.1 times the odds", but the feature is a standardized difference; per raw unit, open versus proprietary, the odds ratio is closer to five. The interpretation also names Gemini, which did not exist in the data.
- **Show that the sampler and the model worked.** No R-hat or effective sample sizes are reported, no posterior predictive check, no held-out accuracy, no comparison between the two models. Each is one call in ArviZ.
- **Keep the derived data.** The feature table is the project's expensive artifact and is ignored by git. Committing it, without the raw conversations, would make the Bayesian part reproducible for free.
- **Small things.** The authors' README reports no results; the whole history is one day; the second author's contribution is invisible; the pool of 200 workers shares one OpenAI client created at import time.

What the project does well: a question students care about, asked of data that is about language models; the paired-difference design with position bias as the intercept; priors stated and intervals used honestly; a reasonable topic pipeline, inertia sweep, archetypes nearest the centroids, labels a reader can check; and a second model whose failure the authors noticed and said out loud rather than reporting its best-looking rows.

## About this project

This project was built in Bayesian Machine Learning with Generative AI Applications (ADSP 32014), part of the Master of Science in Applied Data Science at the University of Chicago, in Autumn 2025. It is shown here as an example for later cohorts.

## Changes by course staff

Course staff forked the authors' repository and made these edits. The authors' code and notebook are unchanged.

- Renamed the authors' README to `REPORT.md`, unchanged, and added this README with the program banner.

## License

The authors have not added a license, so their work is shown here under GitHub's terms for public repositories. Ask them before reusing it.

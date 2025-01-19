# Reference recognition in scientific papers using ChatGPT

Steps taken for each experiment:

1. Extract text from paper
2. Remove references
3. [Optional] Remove reference markers
4. Use one of the following base prompts:

    For text where reference markers (e.g: `[2]`, `[2-5]` etc.) are still present: 
    
    ```text
    I have this scientific paper text from a conference, but I lost all its
    references. Could you take a look and attempt to find every single reference
    from this paper? Meaning, for each "[x]" or "[x, y]" or "[x-x+n]" mark, try to
    figure out what papers were being cited.
    
    Please keep in mind that this paper is from a renowned scientific conference,
    meaning it is likely that all of its references come from academic works such
    as scientific journals and scientific conferences.
    
    I will now proceed to give you parts of the text in separate messages, as it's
    too long for one message. After each part, please give me the references. Here
    is the first part:
    ```

    The purpose of the second paragraph is to make sure that the chatbot won't
    start using the new "Searching the web" feature to find references, from
    previous  experiments it was observed that most found references would end up
    being informal sources found on the web is this measure wasn't taken.

    For text without reference markers:

    ```text
    You are a highly experienced scholar that constantly reads scientific
    papers.
    I have this scientific paper text from a conference, but I lost all its
    references. Could you take a look and attempt to find papers that are
    referenced by this paper? I'm interested in the name of the paper, its
    authors and the reason why you think so (preferably quoting the original text).
    I'm only interested in papers you are 100% sure exist.
    
    Please keep in mind that this paper is from a renowned scientific conference,
    meaning it is likely that all of its references come from academic works such
    as scientific journals and scientific conferences.
    
    I will now proceed to give you parts of the text in separate messages, as it's
    too long for one message. After each part, please give me information about
    the papers you think are cited. Keep in mind that not all excerpts I send you
    need to contain references, please only tell me about papers you're sure are
    referenced.

    Here is the first part:
    ```

5. Proceed to give parts of the paper to ChatGPT (has to be done in batches as
   the text is too long for one message for most papers)

## Paper 1 with reference markers

The first experiment uses a **fairly recent paper** currently submitted for peer
review, this means that there's almost **no chance** that ChatGPT has already been
**trained on its text**, so all references should be proper guesses.

- [Link to paper](https://openreview.net/forum?id=zogaeVpbaE#discussion)
- [Link to ChatGPT conversation](https://chatgpt.com/share/67890704-faa0-800e-ab8f-e8e9c7e0e74f)

It seems that the more the paper itself talks about the papers being cited the
better the results. This is especially true for this paper, as a lot of its
citations are done **in mass** (e.g: `Recent work has attempted to bridge this data
gap by constructing models that are trained on less data, including on
naturalistic data from children [2–5]`) with no further clarifications about
the findings of the cited papers.

The chatbot was however able to find quite a few referenced papers given
enough context, doing exceptionally well on papers that introduce **named
entities** (e.g: `ViLT`, `FLAVA`, `BLIP` etc.) or **popular papers** such as
`Representational similarity analysis – connecting the branches of
systems neuroscience` (around 4k citations).

One of the issues that came up was the **small context window** of GPT4o, this
caused the model to completely forget about the task at hand and simply
summarize the given text around the halfway mark, at which point it had to be
reintroduced to the task.

Some isolated, but unsatisfactory mistakes:

- there are two cases in which the model guessed a correct reference, but for the
wrong marker.  
- it guessed that a paper related to the Microsoft COCO dataset was a correct
reference for multiple different markers
- there are two cases (`[38]` and `[41]`) in which the correct authors are
guessed, but a fictional paper is given as a result (seems to be because the
authors are mentioned by name near the reference marker)
- one of the recognized named entities (`THINGS dataset`, [44]) has the wrong
authors and a fairly different title

Final result: the model guessed 12 references out of 72 if we use a loose
definition of "found" (accepting correct reference but wrong marker, slightly
different names etc.) and 6 out of 72 if we're being strict.

## Paper 1 without reference markers

- [Link to paper](https://openreview.net/forum?id=zogaeVpbaE#discussion)
- [Link to ChatGPT conversation](https://chatgpt.com/share/67894103-85c4-800e-855e-5e358f476e05)

The main differences from the previous attempt are:

- a lot less references were found (mass citations could be a cause)
- even with more detailed instructions the model often gives up on finding an
actual paper and instead just starts doing named entity recognition (which is
a fair strategy)

4 out 72 references were found (in a strict sense of perfect title and author
match)

All the papers that were recognized are fairly old (and were almost certainly
used in training the model) compared to most papers that are being referenced
(which are really new, since LLM benchmarking is a pretty new and niche field).

This leads me to believe that result might be a bit better if we choose an
older paper that has references which were already indexed. Therefore, the
following two areas would be good ways to investigate further:

- an obscure paper that is older
- a popular paper that is older

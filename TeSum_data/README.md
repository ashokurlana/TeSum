## TeSum Corpus

### Download

https://ltrc.iiit.ac.in/showfile.php?filename=downloads/teSum/

The corpus is released under Creative Commons Attribution-NonCommercial 4.0 International License.

### Overview
The data comes in `.jsonl` format, where each JSON object correspond to all the data extracted from a single article. Each JSON object has the following fields:
```
{id,
 url,
 title,
 cleaned_text,
 summary,
 article_sentence_count,
 article_token_count,
 summary_token_count,
 title_token_count,
 compression,
 abstractivity,
 }
```
where `$cleaned_text` and `$summary` of the correspond to the text (document) release as list of sentences, and the source URL for each document `${url}` is available.

Unless you have a specific need, we request you to respect the split decision to prevent test data leakage, and for fair comarision. You could refer to our paper for a detailed explanation.



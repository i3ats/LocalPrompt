---
# Configuration for the Jekyll template "Just the Docs"
parent: Decisions
nav_order: 0001
title: ADR Template
---

<!-- we need to disable MD025, because we use the different heading "ADR Template" in the homepage (see above) than it is foreseen in the template -->
<!-- markdownlint-disable-next-line MD025 -->

# Use NLTK to extract keywords for search

## Context and Problem Statement

Extract keywords from prompt.

## Decision Drivers

* Speed
* Relevance of keywords

## Considered Options

* Use NLTK
* Use GPT-Neo

## Decision Outcome

Chosen option: "Use NLTK," because it is fast and can get the relevant keywords.

## Validation

* Using a series of Unit tests to verify the output.

## Pros and Cons of the Options

### Use NLTK

* Good, because it is fast
* Bad, because it might require the User to reformulate the prompt a few times

### Use GPT Neo

* Good, because is does the job
* Good, because it can imply context
* Bad, because it is relatively slow

## More Information

Remember that this is only the keyword search. It's not even doing the thing we want it to do yet.
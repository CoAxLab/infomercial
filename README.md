# Informercial

`infomercial` contains code for [Curiosity eliminates the exploration-exploitation dilemma](https://www.biorxiv.org/content/10.1101/671362v8), bioArxiv _671362v8_ (2020).

In this paper we present an alternative interpretation of the classic but intractible exploration-exploitation dilemma. We prove the key to finding a tractable solution is to do an unintuitive thing--to explore without considering reward value.

# Research summary 
The exploration-exploitation dilemma is summarized by a simple question: “Should I exploit an available reward, or explore to try out a new uncertain action?” Unfortunately, it’s been proven that this dilemma, when stated as a mathematical problem, is intractable and so can’t be solved directly. This fundamentally limits our ability to predict optimal naturalistic behavior during foraging and exploration, and to optimally drive learning in artificial agents.

To overcome this field-wide limitation, we took a fresh look at the dilemma. Our goal was simple: when one mathematical problem can’t be solved, it’s often good to find another related problem that can be and use that to make progress on both.

We show, for the first time, that nearly any dilemma problem can also be viewed as competition, between exploiting known rewards or exploring to learn a world model--a simplified concept of memory borrowed from computer science. We prove this competition can be perfectly solved using the simplest of all value learning algorithms: a deterministic greedy policy. To ensure this solution is as broad as possible we also derived a new universal theory of information value which complements--but is independent of--Shannon’s Information Theory.

# Code dependencies
- A standard anaconda install
- pytorch (>= 4.1)

# Install
`pip install . -e` following cloning of this repo.

# Experiments
All experiments can be (re)run from the top-level Makefile, found in the `infomercial` repo. 

For analysis see `./notebooks/`

For paper figures see `./figures/`

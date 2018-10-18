# Information games

`infomercial` contains a library and experiments that study a kind of game theory where information is worth having for its own sake. 

Shannon’s formulation holds the key to how we define the value information. This definition naturally allows for several kinds of information games, all of which are general sum. 

Existing evolutionary approaches offer the tools to study information games formally, which lets us show how information play can alter both Nash and Pareto equilibrium of normal form games, like the prisoner's dilemma. 

In the end it is deep learning that really let’s us put information games into general and useful practice.

# Dependencies

- A standard anaconda install
- pytorch (>= 4.1)

# Install

`pip install . -e` following cloning of this repo.

All experiments can be (re)run from the top-level Makefile, found in the `infomercial` repo. 
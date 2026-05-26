# Europa Redux
An arithmetic reasoning model - not by scratchpad, but by structure.

## Key Modifications
The fixed encoding scheme should continue, and be extended by agent-driven development. The major changes from the prior Europa work, for now, will be as follows:

### Data / Formatting
- Math problems will be fixed at two inputs -> one output. 
- Max number will be reduced from 9E8 to 9E6; digit-placing reduced to six places accordingly, and math problem kind should be adjusted as well
- Re-introduction of the <ans> operator token following the = sign in regular arithmetic problems.
- Introduction of boolean true/false problems via 'true' and 'false' vocabulary components. Modification of the generator to generate many true and false examples for integration into training. 
- Include '<' and '>' in vocabulary with opposing (1 vs -1) meanings for comparison questions, and generate some in the dataset 
- Remove the '-' inside parenthesis for negative numbers. We will represent negative numbers inside of '(xxxxxx)' parentheses. while positive numbers will be represented inside of '{xxxxxx}' curly brackets. Each pair of tokens should have opposing meanings as well. 
- There should be no separators anywhere, and we should not have a '<sep>' token. We can reserve the '<pad>' token for later use but for now it should be kept unused if possible. 

### Model Structure
- Instead of one transformer, we will train two models: one bidirectional encoder model, as custom as we'd like, for embedding the tokens; and then one decoding-specialist transformer for producing outputs. 
- Ideally, we should train them separately, by trying to produce a combination evaluation target / residual probing strategy that gives us good signals for how we're doing on the bidirectional encoder. This is more or less non-negotiable unless totally impossible. 

# Model card

## Intended use

Rank pass options for retrospective video review and demonstrate how tracking context can augment event data. This prototype is not suitable for live tactical recommendations.

## Target

A pass receives a pressure label when an opponent is within eight metres of the pass target at the end frame. The prediction uses only information available at the start frame. A separate `turnover_within_5s` field is retained for downstream research but is not the training target.

## Inputs

Pass length and direction; nearest-defender distance at origin and target; passing-lane clearance; nearest teammate support; team width; opponent compactness; and pass origin.

## Architecture

A small PyTorch multilayer perceptron (32 and 16 hidden units, GELU activations, dropout) with class-weighted binary cross-entropy and early stopping. A balanced logistic regression is trained as a required baseline. The nonlinear network must outperform this baseline before it is considered useful.

On the checked-in demo run, both models discriminate well, but the logistic baseline has marginally higher average precision. It is therefore the recommended model; the neural checkpoint is retained as a reproducible research artifact, not presented as the winner.

## Validation

The final 15% of the match timeline is the test set and the preceding 15% is validation. This is more realistic than a random row split but still represents only one anonymised match.

## Limitations

- One public match cannot establish external validity.
- Eight metres is a research threshold, not a universal definition of pressure.
- Player identities and tactical roles are anonymised.
- Tracking errors and substitutions are not explicitly modelled.
- Deployment would require multi-match, team-held-out validation and calibrated probabilities.

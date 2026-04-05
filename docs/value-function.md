We need to clearly define a value function for our work, so we quantitatively can judge our improvements. One initial thought is:

"""
  The best value function for your “game” is not accuracy. Accuracy throws away the main thing you care about: how good your probabilities
  are. On your cleaned UFC table, the bookmaker baseline is already strong: accuracy 0.662, Brier score 0.2119, log loss 0.6116. So your main
  score should be:

  score = bookmaker_logloss - model_logloss

  Higher is better. Use Brier score as secondary. If you want a betting-flavored metric, use flat-stake or Kelly-style simulated return only
  as a tertiary metric on untouched test folds, not as the main objective, because ROI is much noisier.
"""

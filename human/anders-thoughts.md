10-fold CV. Industry standard.

What we could do for our "paper".
Hold out 10-20% of the data (last, chronologically).
On the existing data, extract as much insight as possible. Test different models. Use cross validation. Decide upon a model we wish to use.
Then test it against the final data, and see if we landed on the correct one. Obviously we wont necessarily get it right, because we may still get unlucky, especially if we test a lot of models on the final data. But we have won if we decided on the model we deemed the most likely to fit the data the best.

Use kelly criterion later?




### Feature engineering:
- Age in relation to Wins/Losses. Maybe this will say something. They are correlated
- Dif-features most relevant for everything
- Only Diffs would loose information. But adding the average selectively solves the problem, as they are orthogonal (so no coliniarity)

# STAT 441: Statistical Learning - Complete Course Summary

## 1. Core Concepts

### 1.1 Statistical Learning Fundamentals

**What is Statistical Learning?**
Statistical learning involves using data to learn a function f(x) that relates predictor variables (features) to an outcome variable (label/target). Unlike linear regression where the functional form is assumed to be linear, statistical learning aims to estimate unknown functions from data.

**Supervised vs. Unsupervised Learning**
- **Supervised Learning**: Outcome is known. Data consists of {(x₁, y₁), ..., (xₙ, yₙ)} where xᵢ are predictors and yᵢ are outcomes.
  - Classification: Categorical outcome (e.g., disease/no disease)
  - Regression: Continuous outcome (e.g., stock price, age)
- **Unsupervised Learning**: Outcome is unknown (e.g., clustering). Much harder than supervised learning due to lack of confirmation.

### 1.2 Bias-Variance Tradeoff

The expected mean squared error (MSE) can be decomposed as:
E[(ŷ - y₀)²] = Var(ŷ) + Bias(ŷ)² + Var(ε)

- **Bias** refers to the error between the learning model and the true function
- **Variance** refers to variation due to different training datasets
- **Irreducible Error** is Var(ε), the lower bound on MSE
- Model complexity creates a fundamental tradeoff: flexible models have low bias but high variance; inflexible models have high bias but low variance

### 1.3 Bayes Classifier

The **Bayes classifier** assigns each observation to the most likely class: predict class j such that Pr(Y=j|x) is maximized.

- This minimizes the test error on average
- For binary classification with threshold 0.5: predict y=1 if Pr(Y=1|x) > 0.5
- The **Bayes error rate** is the minimal achievable error rate (often unknown in practice)
- The **conditional Bayes error** at observation x is: 1 - max_j Pr(Y=j|x)

### 1.4 Interpretation vs. Prediction Tradeoff

- **Linear regression**: Highly interpretable, can explain how variables affect outcomes, but only predicts well when the model is correct
- **Statistical learning**: Excels at prediction, but interpretability is challenging ("black box" methods)
- There is an inherent tradeoff: optimize for prediction then interpret, OR restrict models for interpretability then optimize within that class

## 2. Practical Learning Aspects

### 2.1 Overfitting

**Overfitting** occurs when a model fits the random noise in a sample rather than the generalizable relationship. This typically happens when:
- The model is too flexible relative to the data
- There are too many parameters relative to observations

**Defense against overfitting**:
- Separate data into training and test sets
- Fit the model on training data
- Evaluate on test data (if overfitting occurs, test performance deteriorates)

### 2.2 Cross-Validation

**Purpose**: Reduce waste of training data while still estimating test error accurately

**k-fold Cross-Validation**:
1. Partition data into k random subsets of roughly equal size
2. For each of k iterations:
   - Use one subset as test data, remainder as training data
   - Fit the model and evaluate
3. Average the k evaluation results

**Leave-One-Out (LOO) Cross-Validation**: Extreme case where n models are fit on (n-1) observations predicting the single left-out observation. Computationally intensive but useful for algorithms like kNN that don't require model fitting.

**Three-Way Split** (ideal but often not used):
- Training data: for model fitting
- Validation data: for tuning parameters
- Test data: for final error estimation

### 2.3 Evaluation Measures for Classification

**Confusion Matrix**:
|  | Predicted True | Predicted False |
|---|---|---|
| Actual True | TP | FN |
| Actual False | FP | TN |

**Key Metrics**:
- **Accuracy**: (TP + TN) / N
- **Sensitivity** (True Positive Rate): TP / (TP + FN) - % of actual positives found
- **Specificity** (True Negative Rate): TN / (TN + FP) - % of actual negatives correctly identified
- **False Positive Rate**: 1 - Specificity
- **F-score**: Harmonic mean of precision and recall

**ROC Curves**: Display sensitivity vs. (1-specificity) to show the tradeoff between true positive rate and false positive rate as the classification threshold changes.

### 2.4 Data Preprocessing

**One-Hot Encoding**: Convert categorical variables into multiple binary indicators. For a categorical variable with k categories, create k binary variables (or k-1 to avoid perfect multicollinearity).

**Variable Scaling**: Scale continuous variables by standardizing: (x - mean) / std. Important for algorithms sensitive to scale (e.g., SVM, kNN).

**Transformations**: When outcomes span several orders of magnitude, log transform: y_new = log(y + constant).

## 3. Linear Classification Methods

### 3.1 Logistic Regression

**Model Specification**:
- For binary classification with outcomes 0/1
- Logistic regression models: log(p/(1-p)) = β₀ + β₁x₁ + ... + βₚxₚ
- This is equivalent to: p = expit(β₀ + β₁x₁ + ... + βₚxₚ) where expit(t) = eᵗ/(1+eᵗ)
- The logistic function (sigmoid) ensures 0 < p < 1

**Prediction**:
- Predicted probability: p̂ = expit(β₀ + β₁x₁ + ... + βₚxₚ)
- Classification: ŷ = 1 if p̂ > 0.5, else ŷ = 0
- Alternative thresholds (not just 0.5) can be used for different sensitivity/specificity tradeoffs

**Interpretation of Coefficients**:
- **Odds ratio**: exp(βⱼ) gives the multiplicative change in odds for a one-unit increase in xⱼ
- For a one-unit increase in xⱼ, the odds of y=1 multiply by exp(βⱼ)
- Example: if exp(β) = 1.16, a one-unit increase increases odds by 16%

**Estimation**: Maximum likelihood estimation using Newton-Raphson algorithm (non-linear equations in β)

**Newton-Raphson for Logistic Regression**:
```
β_new = β_old + (X'WX)^(-1) X'(y - p)
```
where W is a diagonal matrix with diagonal elements p(1-p) and p is the vector of predicted probabilities

**Advantage**: Logistic regression is relatively inflexible (linear decision boundaries), so less prone to overfitting; usually no cross-validation needed

### 3.2 Regularized Logistic Regression

**Problem**: When many x-variables exist or they're highly correlated, coefficients can become unstable

**L1 Regularization (Lasso)**:
- Minimize: -log(likelihood)/n + λ∑|βⱼ|
- Can shrink coefficients to exactly zero (useful for variable selection)
- Variable with zero coefficient is eliminated

**L2 Regularization (Ridge)**:
- Minimize: -log(likelihood)/n + λ∑βⱼ²
- Coefficients shrink toward zero but never reach exactly zero
- Better than L1 for prediction when many relevant variables exist

**L1 vs. L2**:
- L1: Better when many irrelevant variables exist (performs feature selection)
- L2: Usually better overall prediction accuracy, especially with many relevant variables
- L1 is numerically more challenging

**Choosing λ**: Via cross-validation or information criteria (AIC, BIC)

## 4. Instance-Based Learning

### 4.1 k-Nearest Neighbors (kNN)

**Algorithm**:
1. For new observation x₀, find the k nearest training observations
2. Estimate class probability: P(Y=j|X=x₀) = (# of class j among k neighbors) / k
3. Classify as the most frequent class (majority vote)

**Memory-based**: No training required; all computation happens at prediction time

**Decision Boundaries**: kNN accommodates highly nonlinear decision boundaries (unlike logistic regression's linear boundaries)

**Bias-Variance Tradeoff**:
- k=1: Low bias, high variance (unstable, sensitive to noise)
- Increasing k: Decreases variance, increases bias (more stable but less flexible)
- Optimal k found via cross-validation

**Distance Metrics**:
- **Euclidean distance** (default): √(∑(xᵢ - x'ᵢ)²)
- **Similarity metric**: s = 1 - d/d_max (converts distance to similarity)
- **Cosine similarity**: Common in text mining/document retrieval

**Leave-One-Out Cross-Validation**: Easy for kNN since no model fitting required

**Drawbacks**:
- Computationally expensive for large datasets (must compute distance to all training points)
- Poor for learning variable relationships (highly unstructured)

## 5. Bayesian Methods

### 5.1 Naive Bayes

**Key Idea**: Use Bayes rule with a simplifying independence assumption

**Bayes Rule**:
P(Y=k|X=x) = f_k(x)·P(Y=k) / f(x)

where f_k(x) is the class-conditional density and P(Y=k) is the prior

**The "Naive" Assumption**: Conditional on class k, assume x-variables are independent
- P(X=x|Y=k) = ∏ⱼ P(Xⱼ=xⱼ|Y=k)
- This assumption is almost always wrong but extremely convenient

**Prediction**: Select class k maximizing:
log(P(Y=k)) + ∑ⱼ log(fⱼ_k(xⱼ))

**Estimation**:
- **Categorical features**: P(Xⱼ=xⱼ|Y=k) = (# obs in class k with Xⱼ=xⱼ) / (# obs in class k)
- **Continuous features**: Typically assume Gaussian distribution, estimate mean and variance from data

**Numerical Considerations**: With many variables, probabilities can underflow. Use log-transform to avoid: log doesn't change which class maximizes the probability.

**Why it Works**: Despite the naive independence assumption, Naive Bayes often outperforms sophisticated methods. The bias from the wrong assumption may not hurt posterior probabilities much, especially near decision regions, and it saves variance.

**Advantages**:
- Computationally fast (O(n) complexity)
- Works well even with the obviously wrong assumption
- Fast for both training and testing

## 6. Tree-Based Methods

### 6.1 Classification and Regression Trees (CART)

**Core Idea**: Recursively partition feature space into rectangles, fitting a constant in each rectangle

**Regression Trees**:
- Minimize residual sum of squares (RSS) at each split
- Predicted value in leaf = average y in that leaf
- Nonlinear relationships and interactions are naturally captured

**Best Split**: Find variable j and split point s that minimize:
∑_{i∈R₁} (yᵢ-c₁)² + ∑_{i∈R₂} (yᵢ-c₂)²

where c₁, c₂ are average y values in the two regions

**Interaction Representation**:
- One split: main effect of one variable
- Two splits: (up to) 2-way interaction
- Multiple splits on different variables represent high-order interactions

**Classification Trees**:
- Use different splitting criteria: misclassification error, Gini index, or entropy/deviance
- Prediction: most frequently occurring class in the leaf
- **Gini Index**: G = 1 - ∑ₖ p̂ₖ²
- **Entropy/Deviance**: D = -∑ₖ p̂ₖ log(p̂ₖ)

**Stopping Criterion**:
- Option 1 (bad): Stop if RSS reduction < threshold (misses future big improvements)
- Option 2 (better): Grow tree fully, then prune

**Tree Pruning**: Remove splits to minimize:
Loss = ∑RSS + α|T|

where |T| is the number of leaves, α is tuning parameter (chosen via cross-validation). This regularizes bias-variance tradeoff.

**Handling Different Variable Types**:
- Ordered variables: treat like continuous (e.g., education levels)
- Indicator variables: one split
- Categorical variables: "one vs rest" split

**Advantages**: Highly interpretable, emulates human thinking
**Disadvantages**: Highly unstable to data perturbations, but valuable as building blocks for ensemble methods

### 6.2 Bootstrap and Bagging

**Bootstrap**: Sample with replacement from original data, usually with the same size as original

**Bagging (Bootstrap Aggregation)**:
1. For i = 1 to B:
   - Draw bootstrap sample i
   - Fit model on bootstrap sample
   - Make predictions on bootstrap sample
2. Average predictions: f̂_bag(x) = (1/B)∑ f̂_b(x)

**When Useful**: Most useful for high-variance learners (e.g., trees). Reduces variance, increases stability.

**Not Useful**: Low-variance methods like logistic regression don't benefit much.

**Classification**: Two options:
- Option 1: Majority vote
- Option 2: Average predicted proportions per class (Hastie et al. preference)

**Trade-off**: Bagging destroys interpretability of individual trees

### 6.3 Random Forests

**Key Difference from Bagging**: At each split, randomly select m out of p variables to consider (not all p)
- For classification: m ≈ √p
- For regression: m ≈ p/3
- Use different random subset at each split

**Algorithm**:
1. For b = 1 to B:
   - Draw bootstrap sample
   - Grow tree until desired node size
   - At each split: select m random variables, choose best split among these m
2. Prediction:
   - Regression: average over all trees
   - Classification: majority vote

**Tuning Parameters**:
- Number of trees (stop when test error settles)
- Number of splits (typically full trees)
- Number of variables m

**Out-of-Bag (OOB) Estimates**:
- Each bootstrap sample uses ~2/3 of observations, leaving ~1/3 unused
- Probability of not selecting one observation in n draws = (1-1/n)ⁿ ≈ 0.368
- OOB observations can estimate test error without separate test set
- Efficient use of data

**Variable Importance**:
- For each tree: compute OOB accuracy
- Permute values of variable j in OOB sample
- Recompute OOB accuracy
- Decrease in accuracy = importance of variable j
- Average over all trees

**Advantages**: One of best-performing methods; automatically handles variable interactions
**Disadvantages**: Reduced interpretability compared to single trees

### 6.4 Boosting

**Core Idea**: Combine many weak learners (e.g., shallow trees) sequentially, with later learners focusing on mistakes of earlier ones

#### AdaBoost (Adaptive Boosting)

**For Binary Classification** (y ∈ {-1, 1}):

1. Initialize weights: wᵢ = 1/N for all observations
2. For m = 1 to M:
   - Fit classifier Gₘ(x) using weights wᵢ
   - Compute weighted error: err_m = ∑wᵢ·I(yᵢ ≠ Gₘ(xᵢ))
   - Compute: αₘ = log((1-err_m)/err_m) ≥ 0
   - Update: wᵢ ← wᵢ·exp(αₘ·I(yᵢ ≠ Gₘ(xᵢ)))
3. Final prediction: G(x) = sign(∑ᵐ αₘGₘ(x))

**Weight Updates**:
- Correctly classified observations: weight unchanged
- Incorrectly classified observations: weight multiplied by exp(αₘ)
- Larger αₘ when err_m is small (strong classifier gets higher weight)

#### Forward Stagewise Additive Modeling

**General Framework**:
- Fit additive model: f(x) = ∑ᵐ bₘ(x; γₘ)
- Sequential fit: minimize loss at each step without changing previous terms

**Loss Functions**:
- Exponential loss → AdaBoost
- Binomial deviance → Logit Boost (more robust)
- Squared error → gradient boosting for regression

#### MART Algorithm (Gradient Boosting)

**Initialization**: f₀ = log(p/(1-p)) where p = proportion of class 1

**For each iteration m = 1 to M**:
1. Compute pseudo-residuals: rᵢ = yᵢ - p̂ᵢ (for binary classification)
   - More generally: rᵢ = -∂L/∂f where L is the loss function
2. Fit regression tree with J leaves to pseudo-residuals
3. For each terminal node j:
   - Compute fitted value: γⱼ = argmin_γ ∑Loss(yᵢ, f_{m-1}(xᵢ) + γ)
4. Update: fₘ(x) = f_{m-1}(x) + γⱼ

#### XGBoost (Extreme Gradient Boosting)

**Improvements over MART**:
1. **Regularized Boosting**: Add L2 penalty on tree coefficients
2. **Engineering improvements**: Faster computation via quantile sketching, parallel computing

**Objective Function**:
Loss = ∑L(yᵢ, fₘ(xᵢ)) + 0.5λ∑βⱼ² + α|T|

- L: loss function
- λ: penalty on large coefficients
- α: penalty for number of leaves (encourages pruning)

**Changes from MART**:
- XGBoost fits regularized trees with specified depth
- Allows pruning via gain threshold
- Initialization: f₀ = 0.5 (constant)

**Splitting Criterion**: Maximize "similarity" based on first and second derivatives of loss
- For Gaussian outcomes: similarity = (∑ residuals)² / (# residuals + λ)

**Tuning Parameters**:
- Number of iterations M
- Number of leaves per tree J (or tree depth)
- Shrinkage (learning rate) v ∈ {0.1, 0.01, 0.001}
- Bagging fraction (typically 50%)

**Key Insight**: Shrinking (reducing step size) substantially increases M and runtime but reduces overfitting

#### Boosting Loss Functions

For 2-class classification with y ∈ {-1, 1}:
- **Misclassification**: I(yf < 0) - not differentiable
- **Exponential**: exp(-yf) - very steep for large negative yf
- **Binomial Deviance**: log(1 + exp(-2yf)) - nearly linear for large negative yf (more robust)
- **Squared Error**: (y-f)²

Binomial deviance is more robust because it's less heavily influenced by large negative margins.

## 7. Support Vector Machines (SVM)

### 7.1 Linear SVM

**Goal**: Find hyperplane that separates two classes with maximum margin

**Formulation**: For separable data, find β minimizing:
½||β||² subject to yᵢ(β₀ + xᵢ'β) ≥ 1 for all i

For non-separable data (soft margin SVM), allow some misclassifications.

**Support Vectors**: Points on or near the decision boundary that determine the hyperplane

**Key Property**: Solution depends only on inner products ⟨xᵢ, xⱼ⟩, not on observations themselves

### 7.2 SVM with Kernels

**Motivation**: Data may not be linearly separable in original space

**Feature Expansion**: Rather than explicitly adding quadratic/higher-order terms, use kernel functions

**Kernel Trick**: The solution depends only on K(xᵢ, xⱼ) = ⟨h(xᵢ), h(xⱼ)⟩ where h expands the feature space implicitly

**Common Kernels**:
- **Linear**: K(x, x') = ⟨x, x'⟩ (no expansion, just linear SVM)
- **Polynomial**: K(x, x') = (γ⟨x, x'⟩ + β₀)^d 
  - d = polynomial degree (typically 2 or 3)
  - Expands to all polynomial terms of degree d
- **Radial Basis Function (RBF)**: K(x, x') = exp(-γ||x - x'||²)
  - γ > 0 is tuning parameter
  - Corresponds to infinite-dimensional feature space
  - Most popular choice
- **Sigmoid**: K(x, x') = tanh(γ⟨x, x'⟩ + β₀)
  - Not commonly used (theoretical issues)

### 7.3 SVM for Classification

**Prediction**:
f(x) = β₀ + ∑ᵢ∈S αᵢyᵢK(xᵢ, x)

where S is the set of support vectors and αᵢ are Lagrange multipliers

**SVM Scores**: Not probabilities; positive scores suggest y=1, negative suggest y=-1

**Platt's Method**: Convert scores to probabilities using logistic regression on the score
- Fits: log(p/(1-p)) = β + βf(x)
- Useful for applications requiring probabilities

**Sensitivity to Scaling**: Continuous variables should be standardized: (x - mean)/std

**Multi-class SVM**: Two approaches
- **One-vs-All**: K SVM classifiers (class k vs all others), predict class with highest score
- **One-vs-One**: K(K-1)/2 pairwise classifiers, predict class with most wins

### 7.4 SVM as Regularized Regression

SVM with linear kernel can be written as:
Loss = ∑max(0, 1 - yᵢfᵢ) + λ||β||²

- Loss function: hinge loss max(0, 1 - yf)
- Penalty: L2 regularization
- Similar to logistic regression (which uses binomial deviance loss) but with different loss function
- Explains why SVM and logistic regression often give similar predictions

**Advantages**: Often best-performing method; deterministic
**Disadvantages**: Requires scaling; model is often uninterpretable

## 8. Additional Methods

### 8.1 Lasso and Ridge Regression

**Problem**: When many x-variables exist, regression coefficients can be unstable

**Ridge Regression (L2 Penalty)**:
- Criterion = RSS + λ∑βⱼ²
- Shrinks coefficients toward zero but never to exactly zero
- All variables retained

**Lasso (L1 Penalty)**:
- Criterion = RSS + λ∑|βⱼ|
- Can shrink coefficients to exactly zero (variable selection)
- Numerically more challenging

**Choosing λ**: Via cross-validation

**Geometric Interpretation** (p=2 case):
- Ridge: circular constraint region (coefficients rarely reach corners where they'd be zero)
- Lasso: diamond-shaped constraint region (corners correspond to one coefficient being zero, making variable elimination possible)

## 9. Course Glossary

| Term | Definition |
|---|---|
| **Feature/Variable** | Input x to model |
| **Label/Outcome** | Target y we want to predict |
| **Class** | Category of outcome (e.g., "spam" is one class) |
| **Classification** | Predicting categorical outcome |
| **Regression** | Predicting continuous outcome |
| **Supervised Learning** | Outcome is known (regression/classification) |
| **Unsupervised Learning** | Outcome is unknown (clustering) |
| **Training Data** | Data used to fit the model |
| **Test Data** | Data used to evaluate model performance |
| **Tuning Parameter** | Parameter chosen via cross-validation (e.g., k in kNN, C in SVM) |
| **Bias** | Error from model not capturing true relationship |
| **Variance** | Error from model being sensitive to random variations in training data |
| **Overfitting** | Model fits training noise rather than true relationship |
| **Regularization** | Adding penalty term to prevent overfitting |
| **Cross-Validation** | Using multiple train/test splits to estimate error |
| **Support Vector** | Training point near or on decision boundary |
| **Weak Learner** | Classifier barely better than random guessing |
| **Ensemble** | Combining multiple learners for prediction |

## 10. Summary Table: Methods Covered

| Method | Type | Linear? | Interpretable? | Key Use |
|---|---|---|---|---|
| Logistic Regression | Linear | Yes | Very | Binary classification, interpretability |
| Regularized Logistic | Linear | Yes | Very | High-dimensional problems |
| k-Nearest Neighbors | Instance-based | No | No | Nonlinear boundaries, no assumptions |
| Naive Bayes | Bayesian | Yes | Moderate | Fast training, many variables |
| CART (Trees) | Tree-based | No | Very | Interpretability, interactions |
| Bagging/Bootstrap | Ensemble | No | Low | Reduce variance of trees |
| Random Forests | Ensemble | No | Low | Best overall performance, automatic interactions |
| Boosting (MART) | Ensemble | No | Low | Prediction accuracy |
| XGBoost | Ensemble | No | Low | Fast, scalable gradient boosting |
| SVM (Linear) | Linear | Yes | Moderate | High-dimensional data, text |
| SVM (Kernel) | Nonlinear | No | Low | Nonlinear boundaries |
| Lasso/Ridge | Regularization | Yes | Very | Feature selection (Lasso), prediction |

## 11. Key Takeaways

1. **No Free Lunch**: No single method is best for all problems. Choice depends on data characteristics, goals (interpretability vs. prediction), and computational constraints.

2. **Bias-Variance Tradeoff**: Complex models reduce bias but increase variance. This fundamental tradeoff affects all learning algorithms.

3. **Regularization is Essential**: Adding penalties (L1/L2, tree pruning, shrinkage) prevents overfitting and improves test performance.

4. **Ensemble Methods Work**: Combining multiple weak learners (bagging, boosting, random forests) significantly improves prediction.

5. **Cross-Validation is Your Friend**: Use it to choose tuning parameters and estimate test error. It's more efficient than simple train/test split.

6. **Preprocessing Matters**: Scaling variables, encoding categorical features, and transformations can substantially impact model performance.

7. **Start Simple**: Begin with interpretable methods (logistic regression, simple trees). Add complexity only if needed.

8. **Interpretation vs. Prediction**: Linear models are interpretable but limited; flexible models predict better but are harder to interpret.
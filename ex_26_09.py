# %% [markdown]
# # Exercise class on 26/09/24

# %%
# The usual imports
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
import pandas as pd

# %% [markdown]
# ## Task 1
# - Graph the function $y = sin(x^2).$
# - Differentiate it using numpy!
# - Add some noise to $y.$ Does the derivative get noisier?
# - What about the integral?
#
# Reminder:
# - `np.linspace`: generate an array of evenly spaced numbers
# - `plt.plot(x, y)`: plots the points $(x[i], y[i])$

# %%
x = np.linspace(-5, 5, 1000)
y = np.sin(np.pow(x, 2))
plt.plot(x, y)

# %% [markdown]
# How to differentiate? Want an array $dy$ such that
# $$
# dy[i] = \frac{y[i + 1]  - y[i]}{h}
# $$
# where $h$ is the step size of $x$.
# Let's do it the vectorized way!

# %%
def d(y):
    return y[1:] - y[:-1]

x = np.linspace(-2, 2, 1000)
y = np.sin(np.pow(x, 2))

plt.plot(x[1:], d(y)/d(x)) # Just dy/dx!
plt.plot(x, y)
plt.show()

# %% [markdown]
# What if it's noisy?

# %%
x = np.linspace(-2, 2, 1000)
y = np.sin(np.pow(x, 2)) + 0.01 * np.random.randn(x.shape[0])
plt.plot(x[1:], d(y)/d(x))
plt.plot(x, y)
plt.show()

# %% [markdown]
# Yikes! Why does that happen?
#
# "Physicist's intuition": noise is "higher frequency" than our signal,
# and higher frequency signals have big derivatives. Remember,
#
# $$
# \frac{d}{dx} \sin(\lambda x) = \lambda \cos(x).
# $$
#
# What if I want to integrate my function over its domain?
# Remember that
# $$
# \int_a^b f(x) \, dx \approx \sum_{i = 0}^N f(x_i) (x_{i + 1}- x_i)
# $$
# when $x_i$ is a fine grid over the domain $[a, b].$ This is called a Riemann sum.

# %%
def integral(y, x):
    return np.sum(y[:1] * d(x)) # "Sum of y times dx", just like Liebniz's notation…

integral(y, x)
# %% [markdown]
# That was much easier than integrating symboically.
#
# By the way, what do you think happens to the integral if we add noise to $y$?
# Do you think it blows up like the derivative does?

# %% [markdown]
# ## Task 2
# - Load the iris dataset.
# - Plot it
# - Find a nice classifier for the setosa species
# - Do PCA

# %%
iris = datasets.load_iris()
dir(iris) # Show the attributes of an object

# %%
print(iris.data.shape) # iris.data is a matrix with 150 rows and 4 columns

# %%
print(iris.target) # 150 values in [0, 1, 2]
print(iris.target_names) # 3 strings

# %% [markdown]
# Quiz: how do we turn `iris.target` and `iris.target_names` into an array of feature names?
# Remember that everything's "vectorized"---including indexing.

# %%
iris.target_names[iris.target][:10] # Show the first 10 values.


# %% [markdown]
# Let's load everything into a pandas `DataFrame`. (A `DataFrame` is a good model for a dataset: rows and labelled columns with possibly different types.)

# %%
def get_iris_df():
    df = pd.DataFrame(iris.data, columns=iris.feature_names) # Specify data and attribute names
    target = iris.target
    names = iris.target_names
    df['target'] = iris.target_names[iris.target] # Add a new "target" attribute to the data frame
    return df

df = get_iris_df()
df.head() # Show the first few rows

# %% [markdown]
# How can we visualize our dataset? One good solution: scatter plots for each pair of feature.

# %%
import seaborn as sb
df = get_iris_df()
sb.pairplot(df, hue='target')

# %% [markdown]
# By inspecting the data visually, we find a good "linear decision boundary" for the setosa class.
# This means that we have a number---a linear function of our four numerical attributes---that
# gives us information about the `target` attribute.
#
# If we organize these numerical attributes into a matrix  $X \in \mathbb R^{b \times d}$ with $b = 150$ rows
# and $d = 4$ columns, we can express this linear feature as
# $$
# X v
# $$
# for some vector $v.$

# %%
feature = df.iloc[:, :4] @ np.array([0, 0, 1, 1]) # Matrix multiplication
# (You can also convert a `DataFrame` to an array with `DataFrame.to_numpy()` if you need.)
sb.histplot(df, x=feature, hue='target')

# %% [markdown]
# This is the histogram of the variable we get if we project the points of the following scatter plot onto the diagonal.

# %%
sb.scatterplot(df, x='petal width (cm)', y='petal length (cm)', hue='target')
plt.show()

# %% [markdown]
# Now, what if we want to _autonatically_ find a good direction to project our data onto?
#
# We're now doing basic "machine learning"!
#
# Let's step back. The simplest possible model for the distribution of our vector $x$ of numerical attributes is just
# $$
# x \approx v
# $$
# for some constant vector $v.$ We can choose $v$ by minimizing the function
# $$
# \sum_i \lVert x_i - v \rVert^2
# $$
# where $x_i$ runs over our data. This function is minimized by making $v$
# the mean of the data. (Exercise: differentiate the objective and prove this!)

# %%
x = df.iloc[:,:4] # Select only the numeric attributes
x.mean(axis=0)


# %% [markdown]
# We set up a model and "trained" it on the data! It was a simple model, and there was a closed-form
# expression for its "optimal" parameters. Our model says:
#
# > I think all data points equal this vector (the mean), plus some noise.
#
# Now let's do a little better. Maybe we think that there exists a latent
# one-dimensional variable $f$, unknown to us, so that our data is well
# described by
#
# $$
# x \approx f v_1 + v_0
# $$
#
# for some $v_1$ and $v_0.$ How can we choose $v_1$ and $v_0$ optimally?
#
# If we use an squared loss like above, it turns out that $v_0$ will be the mean
# and $v_1$ will be the _eigenvector_ of the matrix
# $$
# X^T X
# $$
# with largest eigenvalue, where $X$ is the data matrix after we've subtracted away the
# average value of each feature.
#
# The eigenvectors of this matrix are called the _principal components_ of our distribution,
# and their eigenvalues are called singular values. Given $X,$ we can directly compute
# principal components using a matrix factorization called singular value decomposition,
# which is more numerically stable than the eigenvalue problem for $X^T X.$
#
# **You don't have to know this for the course**, but you **do** have to be aware of the
# basic idea of principal componennt analysis (PCA.)
#
# **Hard exercise**: define the loss function for PCA and differentiate it with respect
# to its vector-valued parameters (the principal components).

# %%
def main_principal_component(x):
    zero_bias = x - x.mean(axis=0)
    cov = zero_bias.T @ zero_bias # For "covariance", empirical covariance of features
    return np.linalg.eigh(cov).eigenvectors[:, -1]

x = df.iloc[:,:4]
v = main_principal_component(x)
v # Unit vector giving "direction of largest variance", meaning that
  # how much data points deviate from the mean is best explained by a factor of $v$
  # in the sense of sum of squared error.
# %% [markdown]
# Let's check that $v$ has unit norm:


# %%
print((v ** 2).sum()) # using just basic operations
print(np.linalg.norm(v)) # numpy's norm function

# %% [markdown]
# Let's plot the histogram of our data projected onto its principal component.

# %%
sb.histplot(df, x=(x @ v), hue='target')
# %% [markdown]
# Actually, this is not as useful for classifying the data as
# its projection onto $[0, 0, 1, 1]$!
#
# It turns out that, while the projection onto $v$ has more variance,
# the projection onto $[0, 0, 1/\sqrt 2, 1/\sqrt 2]$ is more useful for classification. Fancy algorithms still require you to use your head and remember what you are _trying to do_.

# %%
from math import sqrt
print((x @ v).var()) # v was chosen to maximize this!
print((x @ np.array([0, 0, 1/sqrt(2), 1/sqrt(2)])).var())


# %% [markdown]
# Let's try finding the top 2 principal components.

# %%
def principal_components(x, k):
    zero_bias = x - x.mean(axis=0)
    cov = zero_bias.T @ zero_bias
    return np.linalg.eigh(cov).eigenvectors[:, -k:] # Learn slices for index!

components = principal_components(x, 2)
components

# %% [markdown]
# Let's plot the projection of our data onto these directions.

# %%
projection = pd.DataFrame(x.to_numpy() @ components, columns=["v2", "v1"])
projection['target'] = df['target']
sb.scatterplot(projection, x='v1', y='v2', hue='target')
plt.show()

# %% [markdown]
# We could also have done that with `sklearn`. In Python there are tools for everything.
#
# %%
from sklearn.decomposition import PCA

# Declare up our "model."
model = PCA(n_components=2)
# Give model our first 4 features (the numerical attributes) as training inputs
# and let it "learn" its parameters. (What are its parameters?)
model.fit(df.iloc[:, :4])
# Transform the data under the learned projection and plot it.
sb.scatterplot(pd.DataFrame(model.transform(x)), x=0, y=1)

# %% [markdown]
# The graph looks basically the same. Remember, the optimal "principal directions" are
# not quite unique: flipping (negating) them does nothing to the loss!
#
# Something to think about: what happens if we apply PCA to data where
# coordinates have very different scales? Can you predict the output of the
# following code?

# %%
data = df.iloc[:, :4]
scaled_data = data * np.array([1, 100, 1, 1]) # Blow up the second feature
model = PCA(n_components=1)
model.fit(scaled_data)
# model.components_ # uncomment and evaluate to see

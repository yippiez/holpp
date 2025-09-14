# NOTE: This file contains perceptron definitions and related utilities.
# NOTE: This file is meant to be used as context or with ipython for exploration and experimentation.

import jax.numpy as jnp 
import matplotlib.pyplot as plt

def perceptron(x, w, b):
    """Standard perceptron activation function."""
    return jnp.where(jnp.dot(w, x) + b > 0, 1, 0)


def display_perceptron(w, b, xlim=(-10, 10), ylim=(-10, 10), resolution=100):
    """Display the decision boundary of a perceptron."""
    x = jnp.linspace(xlim[0], xlim[1], resolution)
    y = jnp.linspace(ylim[0], ylim[1], resolution)
    X, Y = jnp.meshgrid(x, y)
    Z = jnp.array([[perceptron(jnp.array([xi, yi]), w, b) for xi in x] for yi in y])

    fig =plt.figure()
    ax = fig.add_subplot()
    ax.imshow(Z, extent=(xlim[0], xlim[1], ylim[0], ylim[1]), origin='lower', cmap='coolwarm', alpha=0.5)
    
    # Decision boundary line
    if w[1] != 0:
        slope = -w[0] / w[1]
        intercept = -b / w[1]
        x_vals = jnp.array(xlim)
        y_vals = slope * x_vals + intercept
        ax.plot(x_vals, y_vals, color='black')
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title('Perceptron Decision Boundary')

    fig.show()

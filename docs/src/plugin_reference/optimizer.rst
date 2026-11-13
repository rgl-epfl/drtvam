.. _optimizer:

Optimizers
==========

Optimizers implement the update rule used to modify the optimized parameters at
each step of the optimization process. Dr.TVAM uses the `Optimizer` sub-classes
from Mitsuba 3, therefore any optimizer available in Mitsuba 3 can be used in
Dr.TVAM. We additionally provide two variants of the L-BFGS optimizer, which we
will detail shortly.

Optimizers in mitsuba are instatiated as regular objects:

.. code-block:: python

    opt = mi.ad.Adam(lr=1e-2)

They maintain a dictionary-like interface to set and access optimization
parameters. Marking a parameter as optimizable is done like so:

.. code-block:: python

    opt['x'] = mi.TensorXf(1.0)

From that point on, ``opt['x']`` will be considered as a differentiable
quantity, and will get a gradient after backpropagation. Please see the
corresponding `tutorial
<https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/use_optimizers.html>`_
in the Mitsuba 3 documentation for more details.

Mitsuba 3 provides the following optimizers:

* `SGD <https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.ad.SGD>`_
* `Adam <https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.ad.Adam>`_

Additionally, we provide two variants of the L-BFGS optimizer:

L-BFGS (``LBFGS``)
------------------

This optimizer implements the classic limited-memory version of the BFGS
algorithm. It is a quasi-Newton method that approximates the Hessian matrix of
the loss function using past gradients.

The following additional parameters should be provided:

.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Key
      - Type
      - Description

    * - ``m``
      - ``int``
      - The number of past gradients to store. Typical values aare between 5 and
        10. A higher value will require more memory, as more gradients are
        stored, which can be prohibitive for large problems. Defaults to 5.

    * - ``line_search_fn``
      - function
      - The L-BFGS update determines the step size by performing a backtracking
        line search, which evaluates the loss for varying step sizes until one
        satisfying the `Wolfe conditions
        <https://en.wikipedia.org/wiki/Wolfe_conditions>`_ is found. The line
        search function should take a dictionary of updated parameters as input,
        and return the loss value for those parameters.

    * - ``wolfe``
      - ``bool``
      - Whether to use the Wolfe conditions for the line search or only the
        simpler Armijo rule. Defaults to False (i.e. Armijo rule).

    * - ``search_it``
      - ``int``
      - The maximum number of iterations for the line search. Defaults to 20.

    * - ``init_alpha``
      - ``float``
      - The initial step size of the backtracking line search. It is halved
        until the line search conditions are satisfied. Defaults to 10 if
        ``wolfe`` is True, and 1 otherwise.


Linear L-BFGS (``Linear LBFGS``)
--------------------------------

The main use case of Dr.TVAM is to optimize patterns for printing. In that case,
the forward model is linear with respects to the patterns, which enables a nice
performance optimization: the line search in L-BFGS requires evaluating the
forward model at each step, which can be expensive. If the operation is linear,
we only need to compute it once for the search direction :math:`d`:

.. math::
   \mathcal{L}(f(x + \alpha d)) = \mathcal{L}(f(x) + \alpha f(d))

Then, only the loss function :math:`\mathcal{L}` needs to be evaluated at each
line search step, which is much faster than evaluating the full forward model.

This optimizer requires the following additional parameters:

.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Key
      - Type
      - Description

    * - ``m``
      - ``int``
      - The number of past gradients to store. Typical values aare between 5 and
        10. A higher value will require more memory, as more gradients are
        stored, which can be prohibitive for large problems. Defaults to 5.

    * - ``render_fn``
      - function
      - The line search function from L-BFGS is now split in two parts. The
        ``render_fn`` evaluates the forward model, given a dictionary of
        parameters. It should return the recorded dose in the medium, i.e. the
        output of the rendering operation.

    * - ``loss_fn``
      - function
      - A function that takes as argument the recorded dose in the medium, and
        returns the loss value. This is the function that will be evaluated in
        the backtracking line search.

    * - ``search_it``
      - ``int``
      - The maximum number of iterations for the line search. Defaults to 20.

    * - ``init_alpha``
      - ``float``
      - The initial step size of the backtracking line search. It is halved
        until the Armijo condition is satisfied. Defaults to 1.


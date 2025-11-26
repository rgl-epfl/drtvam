.. _loss:

Loss function
===================

Loss functions are a central element of any inverse problem formulation. We
define ours as children of the base ``Loss`` class, which provides a common
interface to evaluate the loss. The ``Loss`` class overrides the ``__call__``
method, which allows to call it as if it were a regular function:

.. code-block:: python

    loss_fn = L2Loss({'reduction': 'sum'})
    loss = loss_fn(y, target, patterns)

As for other plugins, the loss parameters are passed as a dictionary to the
constructor of the loss class.

The following parameters are common to all loss functions: 

.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Key
      - Type
      - Description

    * - ``reduction``
      - ``str``
      - Most objective functions compute a loss *per voxel*, which is then
        aggregated over the entire volume. The ``reduction`` parameter specifies
        how the per-voxel losses are aggregated. Possible values are ``sum`` and
        ``mean``. Defaults to ``sum``.

Reference value
---------------

The reference expected by the loss object differs depending on the choice of
discretization (see :ref:`film`). In the case of binary discretization, the
reference is a single channel binary mask, while in the case of surface-aware
discretization, the reference is a 2-channel tensor, containing the fractional
volumes of the "inside" and "outside" sub-regions of each voxel. For both cases,
the first 3 dimensions of the reference tensor should match the resolution of
the film.

Supported objective functions
-----------------------------

L2 Loss (``L2Loss``)
^^^^^^^^^^^^^^^^^^^^

This loss function computes the squared error between the predicted and target
values.
It requires no additional parameter. We do not recommend using this loss as 
the ``ThresholdedLoss`` provides better results and allows for finer control.


.. _thresholded-loss:
Thresholded loss (``ThresholdedLoss``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This loss function uses two thresholds to penalize different parts of the
predicted volume:

* Non-object voxels whose recorded dose is *higher* than a lower threshold
  ``tl`` are penalized
* object voxels whose recorded dose is *lower* than an upper threshold ``tu >
  tl`` are penalized. 
* Finally, object voxels whose recorded dose is *higher* than ``1.0`` are also
  penalized, to avoid some regions being overexposed. 
* Additionally, a sparsity loss can be added to increase the overall energy 
  efficiency of the patterns. We prefer homogenous patterns without 
  spiking values since those reduce the overall energy of the patterns.

With ``'sum'`` reduction, its full expression is:


.. math::

   L = w_{\text{object}} \cdot \sum_{i\in\text{Object}} \operatorname{ReLU}\left(t_u - v_i\right)^K + w_{\text{void}} \cdot \sum_{i\notin\text{Object}} \operatorname{ReLU}\left(v_i - t_l\right)^K\\ + w_{\text{limit}}\cdot \sum_{i\in\text{Object}} \operatorname{ReLU}\left(v_i - 1\right)^K + w_{\text{sparsity}} \cdot \sum_{p \in \text{patterns}} p^M
   
where $v_i$ is the intensity value at voxel $i$.
This objective function was introduced by Wechsler et al. `[2024]
<https://opg.optica.org/oe/fulltext.cfm?uri=oe-32-8-14705&id=548744>`_.

It requires the following additional parameters:

.. list-table::
    :widths: 10 10 80
    :header-rows: 1

    * - Key
      - Type
      - Description

    * - ``K``
      - ``int``
      - The power to which to raise the error. Defaults to ``2``.
    
    * - ``M``
      - ``int``
      - The power to which to raise the pattern values. Defaults to ``4``.

    * - ``tl``
      - ``float``
      - The lower threshold, in ``[0, 1]``. Defaults to ``0.9``.

    * - ``tu``
      - ``float``
      - The upper threshold, in ``[0, 1]``. Defaults to ``0.95``.

    * - ``weight_object``
      - ``float``
      - Weight of the object term. Defaults to ``1``.

    * - ``weight_void``
      - ``float``
      - Weight of the void term. Defaults to ``1``.

    * - ``weight_limit``
      - ``float``
      - Weight of the overpolymerization term. Defaults to ``1``.

    * - ``weight_sparsity``
      - ``float``
      - Weight of the sparsity term. Defaults to ``0``.


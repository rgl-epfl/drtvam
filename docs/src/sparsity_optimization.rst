.. _sparsity_tips:


Tips to reduce experimental printing time (sparsity optimization)
=================================================================


In TVAM, the printing time depends heavily on the optimized patterns.
This is because the DMD has a certain power per pixel available. If in inside the set of pixels a single very bright pixel occurs, all other pixels are dimmed down with relative intensity to this pixels.
Hence, we desire to have patterns which are more homogeneous in brightness and do not have very bright spots.
In mathematical terms, this means that we want to reduce the sparsity of the patterns (i.e. removing the bright spots)


To tune the sparsity of the patterns during optimization, we recommend to use the :ref:`ThresholdedLoss <thresholded-loss>` which allows to add a sparsity loss term.
To control the influence of the sparsity term, the parameter ``weight_sparsity`` is used.


There is two metrics we are going to use to evaluate the sparsity of the patterns. 
The first one is defined as the energy efficiency of the patterns.

It is given by the ratio of the projected intensity and the total available intensity on the DMD.
The total availabe intensity of the DMD is simply 1 times numbers of pixels in horizontal and vertical direction.
The projected intensity is given by the sum of all pattern values divided by the maximum pattern value.


.. math::
   
   \text{Pattern Energy Efficiency} = \frac{\frac{\sum_{p \in \text{patterns}} p}{\text{max}(p)}}{N_{\text{pixels}}}


This metric is good to evaluate how much energy is projected. However, the efficiency gets also higher if wasteful intensity is projected into the void.
Hence, another metric is the normalized threshold. It is basically the best maximum threshold (for example 0.9) divided by the maximum pixel value in the patterns.
It is still a bit of an abstract quantity which depends absolutely on many parameters. But if one only compares different sparsity weights, it gives a indication about the printing time reduction.


Simple example with low efficiency
----------------------------------
We can see that the patterns are very sparse and have bright spots. This leads to a low energy efficiency of only 0.94%.

.. raw:: html

   <details>
   <summary><a>Simple example</a></summary>

.. code-block:: json

    {
        "vial": {
            "type": "cylindrical",
            "r_int": 7.5,
            "r_ext": 8.0,
            "ior": 1.54,
            "medium": {
                "ior": 1.49,
                "phase": {"type": "rayleigh"},
                "extinction": 0.1,
                "albedo": 0.0
            }
        },
        "projector": {
            "type": "collimated",
            "n_patterns": 300,
            "resx": 200,
            "resy": 200,
            "pixel_size": 20e-3,
            "motion": "circular",
            "distance": 20
        },
        "sensor": {
            "type": "dda",
            "scalex": 3,
            "scaley": 3,
            "scalez": 3,
            "film": {
                "type": "vfilm",
                "resx": 150,
                "resy": 150,
                "resz": 150
            }
        },
        "target": {
            "filename": "benchy.ply",
            "size": 3.0
        },
        "loss": {
            "type": "threshold",
            "tl": 0.6,
            "tu": 0.9
        },
        "transmission_only": true,
        "regular_sampling": true,
        "filter_radon": true,
        "n_steps": 40,
        "spp": 1,
        "spp_ref": 1,
        "spp_grad": 1 
    }
    

.. raw:: html

   </details>


.. image:: resources/sparsity/histogram_sparsity_0.0.png
  :width: 400

.. image:: resources/sparsity/patterns_sparsity_0.0.png
  :width: 300



Simple example with different thresholds
----------------------------------------
One way to reduce the sparsity of the patterns is to increase the lower threshold ``tl`` which decreases contrast unfortunately.
But in this case we can get already a better energy efficiency of 4.22%.

We can run the command:

.. code-block:: bash

   drtvam -Dloss.tl=0.8 example_config.json


.. image:: resources/sparsity/histogram_tl_0.8.png
  :width: 400

.. image:: resources/sparsity/patterns_tl_0.8.png
  :width: 300


Variation of sparsity weight
----------------------------


weight_sparsity=0.1
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   drtvam -Dloss.weight_sparsity=0.1 example_config.json

.. image:: resources/sparsity/histogram_sparsity_0.1.png
  :width: 400

.. image:: resources/sparsity/patterns_sparsity_0.1.png
  :width: 300


weight_sparsity=0.5
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   drtvam -Dloss.weight_sparsity=0.5 example_config.json

.. image:: resources/sparsity/histogram_sparsity_0.5.png
    :width: 400

.. image:: resources/sparsity/patterns_sparsity_0.5.png
    :width: 300


weight_sparsity=1.0
~~~~~~~~~~~~~~~~~~~~
.. code-block:: bash

   drtvam -Dloss.weight_sparsity=1.0 example_config.json

.. image:: resources/sparsity/histogram_sparsity_1.0.png
    :width: 400

.. image:: resources/sparsity/patterns_sparsity_1.0.png
    :width: 300

weight_sparsity=5.0
~~~~~~~~~~~~~~~~~~~~
.. code-block:: bash

   drtvam -Dloss.weight_sparsity=5.0 example_config.json

.. image:: resources/sparsity/histogram_sparsity_5.0.png
    :width: 400

.. image:: resources/sparsity/patterns_sparsity_5.0.png
    :width: 300


weight_sparsity=10.0
~~~~~~~~~~~~~~~~~~~~~
.. code-block:: bash

   drtvam -Dloss.weight_sparsity=10.0 example_config.json

.. image:: resources/sparsity/histogram_sparsity_10.0.png
    :width: 400

.. image:: resources/sparsity/patterns_sparsity_10.0.png
    :width: 300

weight_sparsity=40.0
~~~~~~~~~~~~~~~~~~~~~
.. code-block:: bash

   drtvam -Dloss.weight_sparsity=40.0 example_config.json

.. image:: resources/sparsity/histogram_sparsity_40.0.png
    :width: 400

.. image:: resources/sparsity/patterns_sparsity_40.0.png
    :width: 300




Overall results
----------------
As we can see, increasing the sparsity weight improves the energy efficiency and the normalized threshold significantly. The heigher the sparsity weight, the more homogeneous the patterns become.
In this scenario, the sparsity weight of 5.0 is a good trade-off between histogram separation and sparsity.
Only for high values like 10.0 or 40.0, the pattern quality degrades slightly (as seen by the IoU metric).
If we decide to go with the value of 5.0, we see almost a 10x increase in energy efficiency compared to no sparsity regularization. That means, the final print will required 10x less laser power (or printing time) to print the object since the patterns are more efficient in converting that laser power to absorbed dose.

+------------------+------+-----------------+------------------------+
| sparsity_weight  | tl   | pattern energy  | normalized threshold   |
|                  |      | efficiency      |                        |
+==================+======+=================+========================+
| 0.0              | 0.7  | 0.0094          | 0.274                  |
+------------------+------+-----------------+------------------------+
| 0.1              | 0.7  | 0.0373          | 1.026                  |
+------------------+------+-----------------+------------------------+
| 0.5              | 0.7  | 0.0568          | 1.546                  |
+------------------+------+-----------------+------------------------+
| 1.0              | 0.7  | 0.0688          | 1.842                  |
+------------------+------+-----------------+------------------------+
| 5.0              | 0.7  | 0.1059          | 2.723                  |
+------------------+------+-----------------+------------------------+
| 10.0             | 0.7  | 0.1264          | 3.256                  |
+------------------+------+-----------------+------------------------+
| 40.0             | 0.7  | 0.1701          | 4.418                  |
+------------------+------+-----------------+------------------------+
| 0.0              | 0.8  | 0.0422          | 1.121                  |
+------------------+------+-----------------+------------------------+


What else?
----------
If further sparsity reduction is required, one can also try to play with the power ``M`` of the sparsity term. 
Also, ``weight_void`` or ``weight_object`` can be adjusted to influence the pattern contrast.
Of course, more contrast is desirable, but it also leads to more sparse patterns. So there is always a trade-off to find.

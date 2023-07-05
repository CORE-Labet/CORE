
CORE
--------

**CORE** is a plug-and-play conversational agent for any recommender system built upon `PyTorch <https://pytorch.org>`_.
CORE is

- **Comprehensive**: CORE provides data manager, offline trainer, and online checker.
- **Flexible**: CORE could be integrated into any recommender system.
- **Efficient**: CORE is built upon an online decision tree algorithm instead of heavily learning algorithms.


.. toctree::
   :caption: Documentation:
   :maxdepth: 2

   agent/agent.rst
   manager/manager.rst
   trainer/trainer.rst
   checker/checker.rst

.. toctree::
   :caption: Examples
   :maxdepth: 2

   examples/din.rst

.. toctree::
   :caption: API Documentation
   :maxdepth: 2

   api/api.rst


Citing
------

If you find CORE useful, please cite it in your publications.

.. code-block:: bibtex

      @software{Core,
      author = {Jiarui Jin, Xianyu Chen, Fanghua Ye, Mengyue Yang, Yue Feng, Weinan Zhang, Yong Yu, and Jun Wang},
      title = {Core: A Plug-and-Play Conversational Agent for Recommender System},
      year = {2023},
      publisher = {GitHub},
      journal = {GitHub repository},
      version = {0.1},
      howpublished = {\url{https://github.com/CORE-Labet/CORE}},
      }


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

AutoModels
-----------

In many cases, the architecture you want to use can be guessed from the name or the path of the pretrained model you are supplying to the ``from_pretrained`` method.

AutoClasses are here to do this job for you so that you automatically retreive the relevant model given the name/path to the pretrained weights/config/vocabulary:

Instantiating one of ``AutoModel``, ``AutoConfig`` and ``AutoTokenizer`` will directly create a class of the relevant architecture (ex: ``model = AutoModel.from_pretrained('bert-base-cased')`` will create a instance of ``BertModel``).


``AutoConfig``
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_transformers.AutoConfig
    :members:


``AutoModel``
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_transformers.AutoModel
    :members:


``AutoTokenizer``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_transformers.AutoTokenizer
    :members:

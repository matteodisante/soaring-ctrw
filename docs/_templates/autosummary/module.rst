{{ fullname | escape | underline }}

.. Summary tables below link to the full, inline documentation rendered by
   ``automodule`` further down the same page. Submodules recurse into their
   own pages. Everything is discovered from the live module, so new
   functions/classes/submodules appear automatically.

{% block summary %}
{% if functions or classes %}
.. currentmodule:: {{ fullname }}

{% if classes %}
.. rubric:: Classes

.. autosummary::
{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if functions %}
.. rubric:: Functions

.. autosummary::
{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endif %}
{% endblock %}

{% block modules %}
{% if modules %}
.. rubric:: Submodules

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ fullname }}.{{ item.split('.')[-1] }}
{%- endfor %}
{% endif %}
{% endblock %}

.. rubric:: Reference

.. automodule:: {{ fullname }}
   :members:
   :show-inheritance:
   :member-order: bysource

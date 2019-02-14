---
layout: page
permalink: partners.html
---

# partnertitle

{% for partner in site.data.partners %}

<p>{{partner.name}}</p>

{% endfor %}
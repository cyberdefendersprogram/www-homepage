---
layout: blog
title: Cyber Defenders Workshops 
order: 000
---
{% assign filtered_items = site.workshops | where_exp: "item", "item.order != 0 " %}
{% assign workshops = filtered_items | group_by: "workshop_name" %}

<h1>All Workshops</h1>
  {% for workshop in workshops %}
  <h2>{{ workshop.name }}</h2> 
  <ul>
    {% assign sorted_items = workshop.items | sort: "order" %}
    {% for item in sorted_items  %}
    <li><h3>{{ item.order }}
            <a href="{{ item.url | relative_url }}">
            {{ item.title | escape }}
            </a>
        </h3>
    </li>
    {% endfor %}
    </ul>
 {% endfor %}
